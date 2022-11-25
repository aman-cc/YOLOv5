import time
import math
import yaml
import os
import torch
import yolo
import json
import numpy as np
import pandas as pd

from tqdm import tqdm

DALI = False

def getdatasetstate(args={}):
    return {k: k for k in range(100000)}

class YoloArgs:
    def __init__(self, args) -> None:
        self.batch_size = args['batch_size']
        self.data_dir = args['DATA_DIR']
        self.epochs = args['train_epochs']
        self.use_cuda = args['use_cuda']
        self.results = os.path.join(os.getcwd(), "results.json")

        self.dataset = "coco"
        self.dali = False
        self.period = 273
        self.iters = -1
        self.seed = 3
        self.model_size = 'small'
        self.img_sizes = [320, 416]
        self.lr = 0.01
        self.momentum = 0.937
        self.weight_decay = 0.0005
        self.print_freq = 100
        self.world_size = 1
        self.dist_url = "env://"
        self.mosaic = None
        self.distributed = False
        self.classes = None
        self.amp = False

def train(args, resume_from, ckpt_file):
    yolo_args = YoloArgs(args)
    yolo.setup_seed(yolo_args.seed)
    yolo.init_distributed_mode(yolo_args)
    
    device = torch.device("cuda" if torch.cuda.is_available() and yolo_args.use_cuda else "cpu")
    cuda = device.type == "cuda"
    if cuda: yolo.get_gpu_prop(show=True)
    print(f"Device: {device}")
    
    # # Automatic mixed precision
    if cuda and torch.__version__ >= "1.6.0":
        capability = torch.cuda.get_device_capability()[0]
        if capability >= 7: # 7 refers to RTX series GPUs, e.g. 2080Ti, 2080, Titan RTX
            yolo_args.amp = True
            print("Automatic mixed precision (AMP) is enabled!")
        
    # # ---------------------- prepare data loader ------------------------------- #
    
    # The code below is for COCO 2017 dataset
    if not os.path.isdir(args['DATA_DIR']):
        raise Exception("COCO data not download. Please download COCO using './download_coco.sh'")
    splits = ("train2017", "val2017")
    file_roots = [os.path.join(yolo_args.data_dir, 'images', x) for x in splits]
    ann_files = [os.path.join(yolo_args.data_dir, "annotations/instances_{}.json".format(x)) for x in splits]
    if not os.path.isdir(args["EXPT_DIR"]):
        os.makedirs(args["EXPT_DIR"], exist_ok=True)

    transforms = yolo.RandomAffine((0, 0), (0.1, 0.1), (0.9, 1.1), (0, 0, 0, 0))
    dataset_train = yolo.datasets(yolo_args.dataset, file_roots[0], ann_files[0], train=True)
    dataset_test = yolo.datasets(yolo_args.dataset, file_roots[1], ann_files[1], train=True) # set train=True for eval
    if len(dataset_train) < yolo_args.batch_size:
        raise Exception(f"Very low number of samples. Available samples: {len(dataset_train)} | Batch size: {yolo_args.batch_size}")

    if yolo_args.distributed:
        sampler_train = torch.utils.data.distributed.DistributedSampler(dataset_train)
        sampler_test = torch.utils.data.distributed.DistributedSampler(dataset_test)
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)
        sampler_test = torch.utils.data.SequentialSampler(dataset_test)

    batch_sampler_train = yolo.GroupedBatchSampler(
        sampler_train, dataset_train.aspect_ratios, yolo_args.batch_size, drop_last=True)
    batch_sampler_test = yolo.GroupedBatchSampler(
        sampler_test, dataset_test.aspect_ratios, yolo_args.batch_size)

    yolo_args.num_workers = min(os.cpu_count() // 2, 8, yolo_args.batch_size if yolo_args.batch_size > 1 else 0)
    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, batch_sampler=batch_sampler_train, num_workers=yolo_args.num_workers,
        collate_fn=yolo.collate_wrapper, pin_memory=cuda)

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_sampler=batch_sampler_test, num_workers=yolo_args.num_workers,  
        collate_fn=yolo.collate_wrapper, pin_memory=cuda)

    # cuda version of DataLoader, it behaves like DataLoader, but faster
    # DataLoader's pin_memroy should be True
    d_train = yolo.DataPrefetcher(data_loader_train) if cuda else data_loader_train
    d_test = yolo.DataPrefetcher(data_loader_test) if cuda else data_loader_test
        
    yolo_args.warmup_iters = max(1000, 3 * len(d_train))
    
    # -------------------------------------------------------------------------- #

    model_sizes = {"small": (0.33, 0.5), "medium": (0.67, 0.75), "large": (1, 1), "extreme": (1.33, 1.25)}
    num_classes = len(d_train.dataset.classes)
    model = yolo.YOLOv5(num_classes, model_sizes[yolo_args.model_size], img_sizes=yolo_args.img_sizes).to(device)
    model.transformer.mosaic = yolo_args.mosaic
    
    model_without_ddp = model
    if yolo_args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[yolo_args.gpu])
        model_without_ddp = model.module
    
    params = {"conv_weights": [], "biases": [], "others": []}
    for n, p in model_without_ddp.named_parameters():
        if p.requires_grad:
            if p.dim() == 4:
                params["conv_weights"].append(p)
            elif ".bias" in n:
                params["biases"].append(p)
            else:
                params["others"].append(p)

    yolo_args.accumulate = max(1, round(64 / yolo_args.batch_size))
    wd = yolo_args.weight_decay * yolo_args.batch_size * yolo_args.accumulate / 64
    optimizer = torch.optim.SGD(params["biases"], lr=yolo_args.lr, momentum=yolo_args.momentum, nesterov=True)
    optimizer.add_param_group({"params": params["conv_weights"], "weight_decay": wd})
    optimizer.add_param_group({"params": params["others"]})
    lr_lambda = lambda x: math.cos(math.pi * x / ((x // yolo_args.period + 1) * yolo_args.period) / 2) ** 2 * 0.9 + 0.1

    print("Optimizer param groups: ", end="")
    print(", ".join("{} {}".format(len(v), k) for k, v in params.items()))
    del params
    if cuda: torch.cuda.empty_cache()
       
    ema = yolo.ModelEMA(model)
    ema_without_ddp = ema.ema.module if yolo_args.distributed else ema.ema
    
    epochs = args["train_epochs"]

    # ckpt_path = os.path.join(args["EXPT_DIR"], 'ckpt')
    # ckpts = yolo.find_ckpts(ckpt_path)
    ckpt_path = os.path.join(args["EXPT_DIR"], ckpt_file)
    prev_epochs = 0

    if resume_from is not None:
        ckpt_res_path = os.path.join(args["EXPT_DIR"], resume_from)
        checkpoint = torch.load(ckpt_res_path, map_location=device) # load last checkpoint
        model_without_ddp.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        prev_epochs = checkpoint["epochs"]
        ema_without_ddp.load_state_dict(checkpoint["ema"][0])
        ema.updates = checkpoint["ema"][1]
        del checkpoint
        if cuda: torch.cuda.empty_cache()
    else:
        if os.path.isfile(os.path.join(args["EXPT_DIR"], 'yolov5s_official_2cf45318.pth')):
            pre_trained_ckpt_path = os.path.join(args["EXPT_DIR"], 'yolov5s_official_2cf45318.pth')
            checkpoint = torch.load(pre_trained_ckpt_path, map_location=device) # load last checkpoint
            if num_classes != 80:   # Different dataset tuning
                # Remove some node's weights from loading
                remove_head_list = [
                    'head.predictor.mlp.0.weight',
                    'head.predictor.mlp.0.bias',
                    'head.predictor.mlp.1.weight',
                    'head.predictor.mlp.1.bias',
                    'head.predictor.mlp.2.weight',
                    'head.predictor.mlp.2.bias',
                ]
                for item_ in remove_head_list:
                    checkpoint.pop(item_)
            model_without_ddp.load_state_dict(checkpoint, strict=False)

    for epoch in tqdm(range(epochs)):
        
        if not DALI and yolo_args.distributed:
            sampler_train.set_epoch(epoch)
            
        A = time.time()
        yolo_args.lr_epoch = lr_lambda(epoch) * yolo_args.lr
        print("lr_epoch: {:.4f}, factor: {:.4f}".format(yolo_args.lr_epoch, lr_lambda(epoch)))
        iter_train = yolo.train_one_epoch(model, optimizer, d_train, device, epoch, yolo_args, ema)
        A = time.time() - A
        
        B = time.time()
        eval_output, iter_eval = yolo.evaluate(ema.ema, d_test, device, yolo_args)
        B = time.time() - B

        if yolo.get_rank() == 0:
            print("training: {:.2f} s, evaluation: {:.2f} s".format(A, B))
            yolo.collect_gpu_info("yolov5s", [yolo_args.batch_size / iter_train, yolo_args.batch_size / iter_eval])
            print(eval_output.get_AP())
            
    yolo.save_ckpt(model_without_ddp, optimizer, prev_epochs + epochs, ckpt_path, eval_info=str(eval_output), ema=(ema_without_ddp.state_dict(), ema.updates))

def test(args, ckpt_file):
    yolo_args = YoloArgs(args)
    yolo.setup_seed(yolo_args.seed)

    yolo.init_distributed_mode(yolo_args)
    
    device = torch.device("cuda" if torch.cuda.is_available() and yolo_args.use_cuda else "cpu")
    cuda = device.type == "cuda"
    if cuda: yolo.get_gpu_prop(show=True)
    print("\ndevice: {}".format(device))
    model_sizes = {"small": (0.33, 0.5), "medium": (0.67, 0.75), "large": (1, 1), "extreme": (1.33, 1.25)}

    dataset = yolo_args.dataset
    file_root = os.path.join(yolo_args.data_dir, "images", "val2017")
    ann_file = os.path.join(yolo_args.data_dir, "annotations", "instances_val2017.json")
    if not os.path.isdir(args["EXPT_DIR"]):
        os.makedirs(args["EXPT_DIR"], exist_ok=True)
    ds = yolo.datasets(dataset, file_root, ann_file, train=True)
    dl = torch.utils.data.DataLoader(ds, shuffle=True, collate_fn=yolo.collate_wrapper, pin_memory=cuda)
    # DataPrefetcher behaves like PyTorch's DataLoader, but it outputs CUDA tensors
    d = yolo.DataPrefetcher(dl) if cuda else dl

    num_classes = len(d.dataset.classes)
    model = yolo.YOLOv5(num_classes, model_sizes[yolo_args.model_size], img_sizes=yolo_args.img_sizes).to(device)
    model.transformer.mosaic = yolo_args.mosaic
    model.to(device)

    ckpt_path = os.path.join(args["EXPT_DIR"], ckpt_file)
    # ckpts = yolo.find_ckpts(ckpt_path)
    checkpoint = torch.load(ckpt_path, map_location=device) # load last checkpoint
    model.load_state_dict(checkpoint['model'])
    model.eval()
    if cuda: torch.cuda.empty_cache()

    for p in model.parameters():
        p.requires_grad_(False)

    predictions, labels = {}, {}
    for i, data in enumerate(d):
        images = data.images
        targets = data.targets

        with torch.no_grad():
            results, losses = model(images)

        # Batch-size is 1
        target_boxes = targets[0].get('boxes', [])
        target_labels = targets[0].get('labels', [])
        result_boxes = results[0].get('boxes', [])
        result_labels = results[0].get('labels', [])
        result_scores = results[0].get('scores', [])
        result_logits = results[0].get('logits', [])

        target_boxes, target_labels, result_boxes, result_labels, result_scores, result_logits \
            = (item_.cpu().numpy().tolist() \
            if item_ != [] else item_
            for item_ in(target_boxes, target_labels, result_boxes, result_labels, result_scores, result_logits) \
            )

        labels[i] = {
            'boxes':target_boxes,
            'objects':target_labels
        }
        predictions[i] = {
            "boxes": result_boxes,
            "objects": result_labels,
            "scores": result_scores,
            "pre_softmax": result_logits,
        }

    return {"predictions": predictions, "labels": labels}

def infer(args, ckpt_file=None):
    yolo_args = YoloArgs(args)
    yolo.setup_seed(yolo_args.seed)
    yolo.init_distributed_mode(yolo_args)

    device = torch.device("cuda" if torch.cuda.is_available() and yolo_args.use_cuda else "cpu")
    cuda = device.type == "cuda"
    if cuda: yolo.get_gpu_prop(show=True)
    print("\ndevice: {}".format(device))

    dataset = yolo_args.dataset
    file_root = os.path.join(yolo_args.data_dir, "images", "train2017")
    ann_file = os.path.join(yolo_args.data_dir, "annotations", "instances_train2017.json")
    if not os.path.isdir(args["EXPT_DIR"]):
        os.makedirs(args["EXPT_DIR"], exist_ok=True)

    ds = yolo.datasets(dataset, file_root, ann_file, train=True)
    dl = torch.utils.data.DataLoader(ds, shuffle=True, collate_fn=yolo.collate_wrapper, pin_memory=cuda)
    # DataPrefetcher behaves like PyTorch's DataLoader, but it outputs CUDA tensors
    d = yolo.DataPrefetcher(dl) if cuda else dl


    model_sizes = {"small": (0.33, 0.5), "medium": (0.67, 0.75), "large": (1, 1), "extreme": (1.33, 1.25)}
    num_classes = len(d.dataset.classes)
    model = yolo.YOLOv5(num_classes, model_sizes[yolo_args.model_size], img_sizes=yolo_args.img_sizes).to(device)
    model.transformer.mosaic = yolo_args.mosaic

    ckpt_path = os.path.join(args["EXPT_DIR"], ckpt_file)
    checkpoint = torch.load(ckpt_path, map_location=device) # load last checkpoint
    model.load_state_dict(checkpoint['model'])
    model.eval()
    if cuda: torch.cuda.empty_cache()
    model.to(device)

    for p in model.parameters():
        p.requires_grad_(False)

    db_null_flag = True
    predictions, labels = {}, {}
    for i, data in enumerate(d):
        images = data.images
        targets = data.targets
        
        with torch.no_grad():
            results, losses = model(images)
        # with torch.no_grad():
        
        # Batch-size is 1
        target_boxes = targets[0].get('boxes', [])
        target_labels = targets[0].get('labels', [])
        result_boxes = results[0].get('boxes', [])
        result_labels = results[0].get('labels', [])
        result_scores = results[0].get('scores', [])
        result_logits = results[0].get('logits', [])

        target_boxes, target_labels, result_boxes, result_labels, result_scores, result_logits \
            = (item_.cpu().numpy().tolist() \
            if item_ != [] else item_
            for item_ in(target_boxes, target_labels, result_boxes, result_labels, result_scores, result_logits) \
            )

        labels[i] = {
            'boxes':target_boxes,
            'objects':target_labels
        }
        predictions[i] = {
            "boxes": result_boxes,
            "objects": result_labels,
            "scores": result_scores,
            "pre_softmax": result_logits,
        }

    return {'labels': labels, 'predictions': predictions}

if __name__ == '__main__':
    with open("./config.yaml", "r") as stream:
        args = yaml.safe_load(stream)

    train(args=args, ckpt_file='ckpt', resume_from=None)
    test(args=args, ckpt_file='ckpt')
    infer(args=args, ckpt_file='ckpt')
