import os
import sys

import torch
import yaml

path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(path, '..'))
import yolo
from yolov5 import YOLOv5


def test_yolov5_eval():
    with open("tests/test_config.yaml", "r") as stream:
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
    dataset_test = yolo.datasets(
        yolo_obj.dataset, file_roots[1], ann_files[1], train=True
    )  # set train=True for eval
    if len(dataset_test) < yolo_obj.batch_size:
        raise Exception(
            f"Very low number of samples. Available samples: {len(dataset_test)} | Batch size: {yolo_obj.batch_size}"
        )

    sampler_test = torch.utils.data.SequentialSampler(dataset_test)

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

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test,
        batch_sampler=batch_sampler_test,
        num_workers=num_workers,
        collate_fn=yolo.collate_wrapper,
        pin_memory=cuda,
    )
    d_test = yolo.DataPrefetcher(data_loader_test) if cuda else data_loader_test

    num_classes = len(dataset_test.classes)
    warmup_iters = max(1000, 3 * len(dataset_test))
    yolo_obj.load_model(num_classes, warmup_iters, device)
    yolo_obj.load_weights('ckpt', device, pretrained=False)
    mAP = yolo_obj.evaluate(d_test, device)
    assert mAP >= 20.0, f"Low mAP value on test set: {mAP}"