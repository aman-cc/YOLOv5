# Object Detection using YOLOv5
Object detection library YOLOv5 based on just pytorch. pycocotools pip package used for MSCOCO evaluation. [[Source]](https://github.com/Okery/YOLOv5-PyTorch)

[![Python application](https://github.com/aman-cc/YOLOv5/actions/workflows/python-app.yml/badge.svg)](https://github.com/aman-cc/YOLOv5/actions/workflows/python-app.yml)

[![codecov](https://codecov.io/github/aman-cc/YOLOv5/branch/main/graph/badge.svg?token=SBVAWWY7WZ)](https://codecov.io/github/aman-cc/YOLOv5)

<a target="_blank" href="https://colab.research.google.com/github/aman-cc/YOLOv5/blob/main/notebook_demo.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>

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
