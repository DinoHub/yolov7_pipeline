# YOLOv7 package

## Adapted/Forked from [WongKinYiu's repo](https://github.com/WongKinYiu/yolov7)

Last "merge" date: 7th Sept 2022

## Changes from original repo

- YOLOv7 can be used as a package, with minimal requirements for inference only

## Using YOLOv7 as a Package for Inference

To use YOLOv7 as a package for inference, follow these steps:

### Setting up Using Docker Compose (Recommended)

Using Docker Compose is recommended for setting up the environment. Follow these steps:

1. **Clone the Repository**:
    Clone the YOLOv7 repository. It doesn't need to be in the same folder as your main project. Checkout the `inference` branch.

    ```bash
    git clone https://github.com/DinoHub/yolov7_pipeline.git
    git checkout inference
    ```

1. **Edit Configurations**:

    Edit configurations in `build/docker-compose.yaml` and `build/.env` accordingly. Best practice would be to have a `/data` folder for data (images/videos/etc.) and a `/models` folder for model related items, e.g., weights or cfgs.

1. **Build and Run Docker Container**:

    Build the Docker container and start the Docker Compose:

    ```bash
    cd build
    docker-compose up
    ```

1. **Execute Scripts**:

    Once the container is built, open another terminal and enter the container to execute your scripts. Replace `yolov7_inference` with your image name if you've changed it.

    ```bash
    cd build
    docker-compose exec yolov7_inference bash
    ```

    You can now run inference scripts inside the container. Refer to the [Running YOLOv7 for Inference](#running-yolov7-for-inference) section for details.

### Manual Set Up

If you prefer manual setup, follow these steps:

1. **Clone the Repository**:
    Clone the YOLOv7 repository. It doesn't need to be in the same folder as your main project. Checkout the `inference` branch.

    ```bash
    git clone https://github.com/DinoHub/yolov7_pipeline.git
    git checkout inference
    ```

1. **Install the requirements for YOLOv7**:

    ```bash
    pip install -r build/requirements.txt
    ```

1. **Install YOLOv7 Package**:

    Navigate to your project folder and install YOLOv7 as a package.

    ```
    python3 -m pip install --no-cache-dir /path/to/yolov7
    ```

    Alternatively, use an editable package for faster builds:

    ```
    python3 -m pip install -e /path/to/yolov7
    ```

    **Note**: `/path/to/yolov7` should be the `/src` folder of this repo

## Running YOLOv7 for Inference
1. **Download Weights**:

    Use the provided script to download desired weights:

    ```bash
    cd yolov7/weights
    ./get_weights.sh yolov7 yolov7-e6
    ```

1. **Import YOLOv7 Wrapper Class**:

    In your code, import the YOLOv7 wrapper class for inference. Alternatively, refer to `scripts/inference.py` for example usage:

    ```bash
    from yolov7.yolov7 import YOLOv7
    ```

# Official YOLOv7

Implementation of paper - [YOLOv7: Trainable bag-of-freebies sets new state-of-the-art for real-time object detectors](https://arxiv.org/abs/2207.02696)

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/yolov7-trainable-bag-of-freebies-sets-new/real-time-object-detection-on-coco)](https://paperswithcode.com/sota/real-time-object-detection-on-coco?p=yolov7-trainable-bag-of-freebies-sets-new)
[![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/akhaliq/yolov7)
<a href="https://colab.research.google.com/gist/AlexeyAB/b769f5795e65fdab80086f6cb7940dae/yolov7detection.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"></a>
[![arxiv.org](http://img.shields.io/badge/cs.CV-arXiv%3A2207.02696-B31B1B.svg)](https://arxiv.org/abs/2207.02696)

<div align="center">
    <a href="./">
        <img src="./figure/performance.png" width="79%"/>
    </a>
</div>

## Performance

MS COCO

| Model | Test Size | AP<sup>test</sup> | AP<sub>50</sub><sup>test</sup> | AP<sub>75</sub><sup>test</sup> | batch 1 fps | batch 32 average time |
| :-- | :-: | :-: | :-: | :-: | :-: | :-: |
| [**YOLOv7**](https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7.pt) | 640 | **51.4%** | **69.7%** | **55.9%** | 161 *fps* | 2.8 *ms* |
| [**YOLOv7-X**](https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7x.pt) | 640 | **53.1%** | **71.2%** | **57.8%** | 114 *fps* | 4.3 *ms* |
|  |  |  |  |  |  |  |
| [**YOLOv7-W6**](https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7-w6.pt) | 1280 | **54.9%** | **72.6%** | **60.1%** | 84 *fps* | 7.6 *ms* |
| [**YOLOv7-E6**](https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7-e6.pt) | 1280 | **56.0%** | **73.5%** | **61.2%** | 56 *fps* | 12.3 *ms* |
| [**YOLOv7-D6**](https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7-d6.pt) | 1280 | **56.6%** | **74.0%** | **61.8%** | 44 *fps* | 15.0 *ms* |
| [**YOLOv7-E6E**](https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7-e6e.pt) | 1280 | **56.8%** | **74.4%** | **62.1%** | 36 *fps* | 18.7 *ms* |

## Citation

```
@article{wang2022yolov7,
  title={{YOLOv7}: Trainable bag-of-freebies sets new state-of-the-art for real-time object detectors},
  author={Wang, Chien-Yao and Bochkovskiy, Alexey and Liao, Hong-Yuan Mark},
  journal={arXiv preprint arXiv:2207.02696},
  year={2022}
}
```

## Acknowledgements

<details><summary> <b>Expand</b> </summary>

* [https://github.com/AlexeyAB/darknet](https://github.com/AlexeyAB/darknet)
* [https://github.com/WongKinYiu/yolor](https://github.com/WongKinYiu/yolor)
* [https://github.com/WongKinYiu/PyTorch_YOLOv4](https://github.com/WongKinYiu/PyTorch_YOLOv4)
* [https://github.com/WongKinYiu/ScaledYOLOv4](https://github.com/WongKinYiu/ScaledYOLOv4)
* [https://github.com/Megvii-BaseDetection/YOLOX](https://github.com/Megvii-BaseDetection/YOLOX)
* [https://github.com/ultralytics/yolov3](https://github.com/ultralytics/yolov3)
* [https://github.com/ultralytics/yolov5](https://github.com/ultralytics/yolov5)
* [https://github.com/DingXiaoH/RepVGG](https://github.com/DingXiaoH/RepVGG)
* [https://github.com/JUGGHM/OREPA_CVPR2022](https://github.com/JUGGHM/OREPA_CVPR2022)
* [https://github.com/TexasInstruments/edgeai-yolov5/tree/yolo-pose](https://github.com/TexasInstruments/edgeai-yolov5/tree/yolo-pose)

</details>
