from importlib_resources import files
from pathlib import Path
from time import perf_counter

import cv2
import torch

from script.sahi_general import SahiGeneral
from yolov7.yolov7 import YOLOv7

folder_path = Path('/data')
weights="/models/lre03-best.pt"
cfg="/models/lre03_yolov7-e6e.yaml"
images_suffixes = ['.jpg', '.jpeg', '.png']

image_paths = []
for path in folder_path.iterdir():
    if path.suffix in images_suffixes:
        image_paths.append(path)

output_folder = Path('/data/output')
output_folder.mkdir(parents=True, exist_ok=True)


yolov7 = YOLOv7(
    weights=weights,
    cfg=cfg,
    bgr=True,
    device='cuda',
    model_image_size=640,
    max_batch_size=16,
    half=True,
    same_size=True,
    conf_thresh=0.6,
    trace=False,
    cudnn_benchmark=False,
)

classes = ['airplane', 'ship', 'vehicle']

'''
    SAHI library needs to be installed
    Model needs to have classname_to_idx function and get_detections_dict function
    classname_to_idx : int
        class index of the classname given
    get_detections_dict : List[dict]
        list of detections for each frame with keys: label, confidence, t, l, b, r, w, h
'''
sahi_general = SahiGeneral(
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

imgs = [cv2.imread(str(imgpath)) for imgpath in image_paths]

torch.cuda.synchronize()
tic = perf_counter()
detections = []
for img in imgs:
    detections.append(sahi_general.detect([img], classes))

torch.cuda.synchronize()
dur = perf_counter() - tic

print(f'Time taken: {(dur*1000):0.2f}ms')

for i, img in enumerate(imgs):
    draw_frame = img.copy()
    img_detections = detections[i][0]
    for det in img_detections:
        l = det['l']
        t = det['t']
        r = det['r']
        b = det['b']
        classname = det['label']
        conf = det['confidence']
        text = f"{classname}, {conf:.2f}"
        color = (0, 0, 255)
        # cv2.rectangle(draw_frame, (l, t), (r, b), (255, 255, 0), 1)
        # cv2.putText(draw_frame, classname, (l, t-8), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 0))
        cv2.rectangle(draw_frame, (l,t), (r,b), color, 2)
        cv2.putText(draw_frame, text, (l, t-8), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, thickness=2)

    output_path = output_folder / f'{image_paths[i].stem}_out.jpg'
    cv2.imwrite(str(output_path), draw_frame)
