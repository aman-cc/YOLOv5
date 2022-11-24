# Object Detection using YOLOv5
Source: https://github.com/Okery/YOLOv5-PyTorch

## 1. Download COCO Dataset
```
./download_coco.sh
```

## 2. Activate virtualenv and install requirements
(Tested in python-3.8)
```
python3.8 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
pip install pycocotools
```
If having isses with pycocotools, install these libraries:
```
sudo apt install sox ffmpeg libcairo2 libcairo2-dev libpython3.8-dev
```

## 3. [Optional] Download pretrained weights
Use pretrained weights to speed up training.
```
mkdir log
cd log
wget 'https://github.com/Okery/YOLOv5-PyTorch/releases/download/v0.3/yolov5s_official_2cf45318.pth'
```

## 4. [Optional] Fine-tune on custom dataset
- Update labels.json with your custom labels.
- Split train, val images and place images inside `data/images/train2017` and `data/images/val2017`.
- Place train labels inside `data/annotations` in COCO format JSON with filename: `data/annotations/instances_train2017.json`
- Place val labels inside `data/annotations` in COCO format JSON with filename: `data/annotations/instances_val2017.json`
- Make sure number of samples are more than you batch size (in config).

## Misc
- Number of COCO training samples = 118287
- Number of COCO test samples = 40670