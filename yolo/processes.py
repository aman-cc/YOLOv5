from glob import glob
import torch
import yolo
from PIL import Image
from yolo.visualize import show
from torchvision import transforms
import argparse
import time
import math
import os

# COCO dataset, 80 classes
classes = (
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
    "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
    "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
    "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
    "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
    "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote",
    "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book",
    "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush")

DALI = False

def getdatasetstate(args, split="train"):
    if split == "train":
        glob('')
        dataset = FolderWithPaths(args["TRAIN_DATA_DIR"])
    else:
        dataset = FolderWithPaths(args["TEST_DATA_DIR"])

    dataset.transform = tv.transforms.Compose(
        [tv.transforms.RandomCrop(32), tv.transforms.ToTensor()]
    )
    trainpath = {}
    batchsize = 1
    loader = DataLoader(dataset, batch_size=batchsize, num_workers=2, shuffle=False)
    for i, (_, _, paths) in enumerate(loader):
        for path in paths:
            if split in path:
                trainpath[i] = path
    return trainpath

def train(args, labeled, resume_from, ckpt_file, yolo_args):
    yolo.init_distributed_mode(yolo_args)
    begin_time = time.time()
    print(time.asctime(time.localtime(begin_time)))
    
    device = torch.device("cuda" if torch.cuda.is_available() and yolo_args.use_cuda else "cpu")
    cuda = device.type == "cuda"
    if cuda: yolo.get_gpu_prop(show=True)
    print("\ndevice: {}".format(device))
    
    # # Automatic mixed precision
    yolo_args.amp = False
    if cuda and torch.__version__ >= "1.6.0":
        capability = torch.cuda.get_device_capability()[0]
        if capability >= 7: # 7 refers to RTX series GPUs, e.g. 2080Ti, 2080, Titan RTX
            yolo_args.amp = True
            print("Automatic mixed precision (AMP) is enabled!")
        
    # # ---------------------- prepare data loader ------------------------------- #
    
    # # NVIDIA DALI, much faster data loader.
    # DALI = cuda & yolo.DALI & yolo_args.dali & (yolo_args.dataset == "coco")
    
    # The code below is for COCO 2017 dataset
    # If you're using VOC dataset or COCO 2012 dataset, remember to revise the code
    splits = ("train2017", "val2017")
    file_roots = [os.path.join(yolo_args.data_dir, 'images', x) for x in splits]
    ann_files = [os.path.join(yolo_args.data_dir, "annotations/instances_{}.json".format(x)) for x in splits]
    if DALI:
        # Currently only support COCO dataset; support distributed training
        
        # DALICOCODataLoader behaves like PyTorch's DataLoader.
        # It consists of Dataset, DataLoader and DataPrefetcher. Thus it outputs CUDA tensor.
        print("Nvidia DALI is utilized!")
        d_train = yolo.DALICOCODataLoader(
            file_roots[0], ann_files[0], yolo_args.batch_size, collate_fn=yolo.collate_wrapper,
            drop_last=True, shuffle=True, device_id=yolo_args.gpu, world_size=yolo_args.world_size)
        
        d_test = yolo.DALICOCODataLoader(
            file_roots[1], ann_files[1], yolo_args.batch_size, collate_fn=yolo.collate_wrapper, 
            device_id=yolo_args.gpu, world_size=yolo_args.world_size)
    else:
        transforms = yolo.RandomAffine((0, 0), (0.1, 0.1), (0.9, 1.1), (0, 0, 0, 0))
        dataset_train = yolo.datasets(yolo_args.dataset, file_roots[0], ann_files[0], train=True, labeled=labeled)
        dataset_test = yolo.datasets(yolo_args.dataset, file_roots[1], ann_files[1], train=True, labeled=labeled) # set train=True for eval
        for x in dataset_train:
            tmp = x

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

    print(yolo_args)
    yolo.setup_seed(yolo_args.seed)
    
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
    
    start_epoch = 0
    ckpts = yolo.find_ckpts(yolo_args.ckpt_path)
    if ckpts:
        checkpoint = torch.load(ckpts[-1], map_location=device) # load last checkpoint
        model_without_ddp.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_epoch = checkpoint["epochs"]
        ema_without_ddp.load_state_dict(checkpoint["ema"][0])
        ema.updates = checkpoint["ema"][1]
        del checkpoint
        if cuda: torch.cuda.empty_cache()

    since = time.time()
    print("\nalready trained: {} epochs; to {} epochs".format(start_epoch, yolo_args.epochs))
    
    # ------------------------------- train ------------------------------------ #
        
    for epoch in range(start_epoch, yolo_args.epochs):
        print("\nepoch: {}".format(epoch + 1))
        
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

        trained_epoch = epoch + 1
        if yolo.get_rank() == 0:
            print("training: {:.2f} s, evaluation: {:.2f} s".format(A, B))
            yolo.collect_gpu_info("yolov5s", [yolo_args.batch_size / iter_train, yolo_args.batch_size / iter_eval])
            print(eval_output.get_AP())
            
            yolo.save_ckpt(
                model_without_ddp, optimizer, trained_epoch, yolo_args.ckpt_path,
                eval_info=str(eval_output), ema=(ema_without_ddp.state_dict(), ema.updates))

            # It will create many checkpoint files during training, so delete some.
            ckpts = yolo.find_ckpts(yolo_args.ckpt_path)
            remaining = 60
            if len(ckpts) > remaining:
                for i in range(len(ckpts) - remaining):
                    os.system("rm {}".format(ckpts[i]))
        
    # -------------------------------------------------------------------------- #

    print("\ntotal time of this training: {:.2f} s".format(time.time() - since))
    if start_epoch < yolo_args.epochs:
        print("already trained: {} epochs\n".format(trained_epoch))

def infer(args, unlabeled, ckpt_file):
    model = yolo.YOLOv5(80, img_sizes=672, score_thresh=0.3)
    model.eval()

    checkpoint = torch.load(ckpt_file)
    model.load_state_dict(checkpoint)

    use_cuda = False
    device = torch.device("cuda" if torch.cuda.is_available() and use_cuda else "cpu")
    dataset = "coco"
    file_root = "../datasets/coco/images/val2017"
    ann_file = "../datasets/coco/annotations/instances_val2017.json"
    cuda = device.type == "cuda"
    if cuda: yolo.get_gpu_prop(show=True)
    ds = yolo.datasets(dataset, file_root, ann_file, train=True)
    dl = torch.utils.data.DataLoader(ds, shuffle=True, collate_fn=yolo.collate_wrapper, pin_memory=cuda)
    # DataPrefetcher behaves like PyTorch's DataLoader, but it outputs CUDA tensors
    d = yolo.DataPrefetcher(dl) if cuda else dl
    model.to(device)

    if ckpt_file:
        checkpoint = torch.load(ckpt_file, map_location=device)
        if "ema" in checkpoint:
            model.load_state_dict(checkpoint["ema"][0])
            print(checkpoint["eval_info"])
        else:
            model.load_state_dict(checkpoint)
        del checkpoint
        if cuda: torch.cuda.empty_cache()
        
    for p in model.parameters():
        p.requires_grad_(False)


    iters = 10
    outputs_fin = {}
    for i, data in enumerate(d):
        if i >= iters - 1:
            break

        images = data.images
        targets = data.targets
        
        with torch.no_grad():
            results, losses = model(images)

        for j in range(len(results)):
            outputs_fin[unlabeled[i]] = {}
            outputs_fin[unlabeled[i]]["boxes"] = results[j]['boxes'].cpu().numpy().tolist()
            outputs_fin[unlabeled[i]]["prediction"] = results[j]['labels'].cpu().numpy().tolist()
            outputs_fin[unlabeled[i]]["pre_softmax"] = results[j]['logits'].cpu().numpy().tolist()

    return {"outputs": outputs_fin}

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--use-cuda", action="store_true") # whether use the GPU
    
    parser.add_argument("--dataset", default="coco") # style of dataset, choice: ["coco", "voc"]
    parser.add_argument("--data-dir", default="/data/coco2017") # root directory of the dataset
    parser.add_argument("--dali", action="store_true") # NVIDIA's DataLoader, faster but without random affine
    parser.add_argument("--ckpt-path") # basic checkpoint path
    parser.add_argument("--results") # path where to save the evaluation results
    
    # you may not train the model for 273 epochs once, and want to split it into several tasks.
    # set epochs={the target epoch of each training task}
    parser.add_argument("--epochs", type=int, default=1)
    
    # total epochs. iterations=500000, true batch size=64, so total epochs=272.93
    parser.add_argument("--period", type=int, default=273) 
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--iters", type=int, default=-1) # max iterations per epoch, -1 denotes an entire epoch
    
    parser.add_argument("--seed", type=int, default=3) # random seed
    parser.add_argument("--model-size", default="small") # choice: ["small", "medium", "large", "extreme"]
    parser.add_argument('--img-sizes', nargs="+", type=int, default=[320, 416]) # range of input images' max_size during training
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--momentum", type=float, default=0.937)
    parser.add_argument("--weight-decay", type=float, default=0.0005)
    
    parser.add_argument("--mosaic", action="store_true") # mosaic data augmentaion, increasing ~2% AP, a little slow
    parser.add_argument("--print-freq", type=int, default=100) # frequency of printing losses during training
    parser.add_argument("--world-size", type=int, default=1) # total number of processes
    parser.add_argument("--dist-url", default="env://") # distributed initial method
    
    parser.add_argument("--root") # gpu cloud platform special
    yolo_args = parser.parse_args()
    
    if yolo_args.ckpt_path is None:
        yolo_args.ckpt_path = "./checkpoint.pth"
    if yolo_args.results is None:
        yolo_args.results = os.path.join(os.path.dirname(yolo_args.ckpt_path), "results.json")

    train(args=None, labeled=[156,509,603,1774], resume_from=None, ckpt_file='ckpt', yolo_args=yolo_args)
    # infer(args=None, unlabeled=[0,2,3,10], ckpt_file="ckpts/yolov5s_official_2cf45318.pth")