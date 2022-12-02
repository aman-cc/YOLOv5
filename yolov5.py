import math
import os
import time

import torch
import yaml
from tqdm import tqdm

import yolo


class YOLOv5:
    def __init__(self, args) -> None:
        self.data_dir = args["DATA_DIR"]
        self.batch_size = args["batch_size"]
        self.epochs = args["train_epochs"]
        self.use_cuda = args["use_cuda"]
        self.results = os.path.join(os.getcwd(), "results.json")
        self.dataset = args["dataset"]
        self.dali = args["dali"]
        self.expt_dir = args["EXPT_DIR"]
        self.period = args["period"]
        self.iters = args["iters"]
        self.seed = args["seed"]
        self.model_size = args["model_size"]
        self.img_sizes = args["img_sizes"]
        self.lr = args["lr"]
        self.momentum = args["momentum"]
        self.weight_decay = args["weight_decay"]
        self.print_freq = args["print_freq"]
        self.world_size = args["world_size"]
        self.distributed = args["distributed"]
        self.mosaic = args["mosaic"]
        self.amp = args["amp"]
        self.model_sizes = {
            "small": (0.33, 0.5),
            "medium": (0.67, 0.75),
            "large": (1, 1),
            "extreme": (1.33, 1.25),
        }

    def load_model(self, num_classes, warmup_iters, device):
        cuda = device.type == "cuda"
        self.num_classes = num_classes
        if cuda:
            yolo.get_gpu_prop(show=True)
        print(f"Device: {device}")

        self.model = yolo.YOLOv5(
            num_classes, self.model_sizes[self.model_size], img_sizes=self.img_sizes
        ).to(device)
        self.model.transformer.mosaic = self.mosaic
        self.warmup_iters = warmup_iters

        self.model_without_ddp = self.model
        params = {"conv_weights": [], "biases": [], "others": []}
        for n, p in self.model_without_ddp.named_parameters():
            if p.requires_grad:
                if p.dim() == 4:
                    params["conv_weights"].append(p)
                elif ".bias" in n:
                    params["biases"].append(p)
                else:
                    params["others"].append(p)

        self.accumulate = max(1, round(64 / self.batch_size))
        wd = self.weight_decay * self.batch_size * self.accumulate / 64
        self.optimizer = torch.optim.SGD(
            params["biases"], lr=self.lr, momentum=self.momentum, nesterov=True
        )
        self.optimizer.add_param_group(
            {"params": params["conv_weights"], "weight_decay": wd}
        )
        self.optimizer.add_param_group({"params": params["others"]})
        self.lr_lambda = (
            lambda x: math.cos(math.pi * x / ((x // self.period + 1) * self.period) / 2)
            ** 2
            * 0.9
            + 0.1
        )

        print("Optimizer param groups: ", end="")
        print(", ".join("{} {}".format(len(v), k) for k, v in params.items()))
        del params
        if cuda:
            torch.cuda.empty_cache()

        self.ema = yolo.ModelEMA(self.model)
        self.ema_without_ddp = self.ema.ema.module if self.distributed else self.ema.ema

    def load_weights(self, weights_file, device, pretrained=False):
        ckpt_path = os.path.join(self.expt_dir, weights_file)
        if not os.path.isfile(ckpt_path):
            raise FileNotFoundError(f"Model weight not found at {ckpt_path}")

        checkpoint = torch.load(ckpt_path, map_location=device)  # load last checkpoint
        if pretrained and self.num_classes != 80:  # Different dataset tuning
            # Remove some node's weights from loading
            remove_head_list = [
                "head.predictor.mlp.0.weight",
                "head.predictor.mlp.0.bias",
                "head.predictor.mlp.1.weight",
                "head.predictor.mlp.1.bias",
                "head.predictor.mlp.2.weight",
                "head.predictor.mlp.2.bias",
            ]
            for item_ in remove_head_list:
                checkpoint.pop(item_)
            self.model_without_ddp.load_state_dict(checkpoint, strict=False)
        else:
            self.model_without_ddp.load_state_dict(checkpoint["model"])
        self.ema = yolo.ModelEMA(self.model)
        self.ema_without_ddp = self.ema.ema.module if self.distributed else self.ema.ema

    def train(self, d_train, d_test, save_path, device):
        loss_list, mAP_list = [], []
        for epoch in tqdm(range(self.epochs)):

            A = time.time()
            self.lr_epoch = self.lr_lambda(epoch) * self.lr
            print(
                "lr_epoch: {:.4f}, factor: {:.4f}".format(
                    self.lr_epoch, self.lr_lambda(epoch)
                )
            )
            iter_train, loss = yolo.train_one_epoch(
                self.model, self.optimizer, d_train, device, epoch, self, self.ema
            )
            loss_list.append(loss)
            A = time.time() - A

            B = time.time()
            eval_output, iter_eval = yolo.evaluate(self.ema.ema, d_test, device, self)
            B = time.time() - B

            if yolo.get_rank() == 0:
                print("training: {:.2f} s, evaluation: {:.2f} s".format(A, B))
                yolo.collect_gpu_info(
                    "yolov5s",
                    [self.batch_size / iter_train, self.batch_size / iter_eval],
                )
                print(eval_output.get_AP())
                mAP_list.append(eval_output.get_AP().get('bbox AP', 0.0))

            with open('train_monitoring.yaml', 'w') as f:
                train_metrics = dict(loss=loss_list, mAP=mAP_list)
                yaml.dump(train_metrics, f)

        yolo.save_ckpt(
            self.model_without_ddp,
            self.optimizer,
            self.epochs,
            save_path,
            eval_info=str(eval_output),
            ema=(self.ema_without_ddp.state_dict(), self.ema.updates),
        )

    def evaluate(self, d_test, device):
        eval_output, iter_eval = yolo.evaluate(self.ema.ema, d_test, device, self)
        print(f"{eval_output.get_AP()}")
        return eval_output.get_AP().get('bbox AP')

    def infer(self, d_infer, device):
        self.model.eval()
        self.model.to(device)

        for p in self.model.parameters():
            p.requires_grad_(False)

        predictions, labels = {}, {}
        for i, data in enumerate(d_infer):
            images = data.images
            targets = data.targets

            with torch.no_grad():
                results, losses = self.model(images)
            # with torch.no_grad():

            # Batch-size is 1
            target_boxes = targets[0].get("boxes", [])
            target_labels = targets[0].get("labels", [])
            result_boxes = results[0].get("boxes", [])
            result_labels = results[0].get("labels", [])
            result_scores = results[0].get("scores", [])
            result_logits = results[0].get("logits", [])

            (
                target_boxes,
                target_labels,
                result_boxes,
                result_labels,
                result_scores,
                result_logits,
            ) = (
                item_.cpu().numpy().tolist() if item_ != [] else item_
                for item_ in (
                    target_boxes,
                    target_labels,
                    result_boxes,
                    result_labels,
                    result_scores,
                    result_logits,
                )
            )

            labels[i] = {"boxes": target_boxes, "objects": target_labels}
            predictions[i] = {
                "boxes": result_boxes,
                "objects": result_labels,
                "scores": result_scores,
                "pre_softmax": result_logits,
            }

        return {"labels": labels, "predictions": predictions}


if __name__ == "__main__":
    with open("./config.yaml", "r") as stream:
        args = yaml.safe_load(stream)

    yolo_obj = YOLOv5(args)

    if not os.path.isdir(args["DATA_DIR"]):
        raise Exception(
            "COCO data not download. Please download COCO using './download_coco.sh'"
        )
    splits = ("train2017", "val2017")
    file_roots = [os.path.join(yolo_obj.data_dir, "images", x) for x in splits]
    ann_files = [
        os.path.join(yolo_obj.data_dir, "annotations/instances_{}.json".format(x))
        for x in splits
    ]
    if not os.path.isdir(args["EXPT_DIR"]):
        os.makedirs(args["EXPT_DIR"], exist_ok=True)

    transforms = yolo.RandomAffine((0, 0), (0.1, 0.1), (0.9, 1.1), (0, 0, 0, 0))
    dataset_train = yolo.datasets(
        yolo_obj.dataset, file_roots[0], ann_files[0], train=True
    )
    dataset_test = yolo.datasets(
        yolo_obj.dataset, file_roots[1], ann_files[1], train=True
    )  # set train=True for eval
    if len(dataset_train) < yolo_obj.batch_size:
        raise Exception(
            f"Very low number of samples. Available samples: {len(dataset_train)} | Batch size: {yolo_obj.batch_size}"
        )

    sampler_train = torch.utils.data.RandomSampler(dataset_train)
    sampler_test = torch.utils.data.SequentialSampler(dataset_test)

    batch_sampler_train = yolo.GroupedBatchSampler(
        sampler_train, dataset_train.aspect_ratios, yolo_obj.batch_size, drop_last=True
    )
    batch_sampler_test = yolo.GroupedBatchSampler(
        sampler_test, dataset_test.aspect_ratios, yolo_obj.batch_size
    )

    num_workers = min(
        os.cpu_count() // 2, 8, yolo_obj.batch_size if yolo_obj.batch_size > 1 else 0
    )
    device = torch.device(
        "cuda" if torch.cuda.is_available() and yolo_obj.use_cuda else "cpu"
    )
    cuda = device.type == "cuda"
    data_loader_train = torch.utils.data.DataLoader(
        dataset_train,
        batch_sampler=batch_sampler_train,
        num_workers=num_workers,
        collate_fn=yolo.collate_wrapper,
        pin_memory=cuda,
    )

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test,
        batch_sampler=batch_sampler_test,
        num_workers=num_workers,
        collate_fn=yolo.collate_wrapper,
        pin_memory=cuda,
    )
    d_train = yolo.DataPrefetcher(data_loader_train) if cuda else data_loader_train
    d_test = yolo.DataPrefetcher(data_loader_test) if cuda else data_loader_test

    num_classes = len(dataset_train.classes)
    warmup_iters = max(1000, 3 * len(dataset_train))
    save_path = os.path.join(args["EXPT_DIR"], "ckpt")
    yolo_obj.load_model(num_classes, warmup_iters, device)
    yolo_obj.load_weights("yolov5s_official_2cf45318.pth", device, pretrained=True)
    yolo_obj.train(d_train, d_test, save_path, device)
    mAP = yolo_obj.evaluate(d_test, device)
    # yolo_obj.load_weights('ckpt', pretrained=False)
    results = yolo_obj.infer(d_test, device)
