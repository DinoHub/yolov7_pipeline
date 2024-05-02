import argparse
import cv2
import logging
from pathlib import Path
import torch
from time import perf_counter

from yolov7.yolov7 import YOLOv7
from script.sahi_general import SahiGeneral

"""
Perform YOLOv7 inference on an image/folder of images.

This script takes in an image or a folder with images and performs object detection using YOLOv7. 
Optionally, it can utilize SAHI for improved accuracy.
The detected objects are annotated in the output images.

Usage:
    python inference_image.py [-i INPUT_FOLDER/FILE] [-o OUTPUT_FOLDER] [-w WEIGHTS_PATH] [-c CONFIG_PATH] [-cl CLASSES [CLASSES ...]] [--sahi]
"""

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()

def parse_args():
    parser = argparse.ArgumentParser(description="Object detection script")
    parser.add_argument("-i", "--input_folder", type=str, default="/data/input", help="Input folder or file path")
    parser.add_argument("-o", "--output_folder", type=str, default="/data/output", help="Output folder path")
    # Please ensure weights have been reparameterised and state dict has been saved (refer to main branch)
    parser.add_argument("-w", "--weights_path", type=str, default="/models/best.pt", help="YOLOv7 weights file path")
    parser.add_argument("-c", "--config_path", type=str, default="/models/yolov7.yaml", help="YOLOv7 config file path")
    parser.add_argument("-cl", "--classes", nargs='+', default=None, help="Target classes")
    parser.add_argument("-sahi", "--use_sahi", action='store_true', help="Use SAHI for inference")
    return parser.parse_args()

def initialize_yolov7_model(weights_path, config_path):
    """
    Initialize the YOLOv7 object detection model.

    Args:
        weights_path (str): File path to the YOLOv7 weights.
        config_path (str): File path to the YOLOv7 config.

    Returns:
        yolov7.YOLOv7: Initialized YOLOv7 model object.
    """
    yolov7 = YOLOv7(
        weights=weights_path,
        cfg=config_path,
        bgr=True,
        device='cuda',
        model_image_size=640,
        max_batch_size=16,
        half=True,
        same_size=False,
        conf_thresh=0.6,
        trace=False,
        cudnn_benchmark=False,
    )
    return yolov7

def initialize_sahi_model(yolov7):
    """
    Initialize the SAHI (Scalable Accuracy Heatmap Implementation) model.

    Args:
        yolov7 (yolov7.YOLOv7): Initialized YOLOv7 model object.

    Returns:
        SahiGeneral: Initialized SAHI model object.
    """
    sahi = SahiGeneral(
        model=yolov7,
        sahi_image_height_threshold=900,
        sahi_image_width_threshold=900,
        sahi_slice_height=256,
        sahi_slice_width=256,
        sahi_overlap_height_ratio=0.2,
        sahi_overlap_width_ratio=0.2,
        sahi_postprocess_type="GREEDYNMM",
        sahi_postprocess_match_metric="IOS",
        sahi_postprocess_match_threshold=0.5,
        sahi_postprocess_class_agnostic=True,
        full_frame_detection=True
    )
    return sahi

def draw_bbox(frame, text, bbox, color=(0, 0, 255), bbox_thickness=2, font_scale=1.0, font_thickness=2):
    left, top, right, bottom = bbox
    cv2.rectangle(frame, (left, top), (right, bottom), color, bbox_thickness)
    cv2.putText(frame, text, (left, top + 25), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness=font_thickness)

def detect_yolov7(yolov7, images):
    torch.cuda.synchronize()
    start_time = perf_counter()

    all_detections = yolov7.detect_get_box_in(images, box_format='ltrb', classes=None, buffer_ratio=0.0)

    torch.cuda.synchronize()
    duration = perf_counter() - start_time
    logger.info(f'Time taken: {(duration * 1000):0.2f}ms for {len(images)} images.')

    for idx, detections in enumerate(all_detections):
        frame = images[idx].copy()
        logger.info(f'Image {image_paths[idx].name}: {len(detections)} detections')

        for det in detections:
            bbox, score, class_ = det
            text = f"{class_}, {score:.2f}"
            draw_bbox(frame, text, bbox)

        output_path = Path(output_folder) / f'{image_paths[idx].stem}_det.jpg'
        cv2.imwrite(str(output_path), frame)

def detect_sahi(yolov7, images, classes):
    sahi = initialize_sahi_model(yolov7)

    torch.cuda.synchronize()
    start_time = perf_counter()
    detections = [sahi.detect([img], classes) for img in images]

    torch.cuda.synchronize()
    duration = perf_counter() - start_time
    logger.info(f'Time taken: {(duration * 1000):0.2f}ms for {len(images)} images.')

    for i, img in enumerate(images):
        frame = img.copy()
        img_detections = detections[i][0]
        for det in img_detections:
            classname = det['label']
            confidence = det['confidence']
            text = f"{classname}, {confidence:.2f}"
            draw_bbox(frame, text, (det['l'], det['t'], det['r'], det['b']))

        output_path = output_folder / f'{image_paths[i].stem}_sahi.jpg'
        cv2.imwrite(str(output_path), frame)

if __name__ == "__main__":
    args = parse_args()

    input_folder = Path(args.input_folder)
    output_folder = Path(args.output_folder)
    weights_path = args.weights_path
    config_path = args.config_path
    target_classes = args.classes
    use_sahi = args.use_sahi

    # Model initialization
    yolov7 = initialize_yolov7_model(weights_path, config_path)

    # Process images
    image_suffix = ['.jpg', '.jpeg', '.png']
    if input_folder.is_file():
        if input_folder.suffix.lower() in image_suffix:
            images = [cv2.imread(str(input_folder))]
        else:
            raise ValueError(f"Unsupported file type: {input_folder}. "
                         f"Supported image suffixes are: {', '.join(image_suffix)}")
    elif input_folder.is_dir():
        image_paths = [path for path in input_folder.iterdir() if path.suffix in image_suffix]
        images = [cv2.imread(str(img_path)) for img_path in image_paths]
    else:
        raise FileNotFoundError(f"Input path does not exist: {input_folder}")

    output_folder.mkdir(parents=True, exist_ok=True)

    # Perform detections
    if use_sahi:
        detect_sahi(yolov7, images, target_classes)
    else:
        detect_yolov7(yolov7, images)

    logger.info(f"Completed. Output images saved to {str(output_folder)}.")
