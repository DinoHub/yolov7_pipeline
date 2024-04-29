from importlib_resources import files
from pathlib import Path

import cv2

from yolov7.yolov7 import YOLOv7


src_folder = '/data/'
output_folder = '/data/output_no_sahi'
weights="/models/lre03-best.pt"
cfg="/models/lre03_yolov7-e6e.yaml"
Path(output_folder).mkdir(parents=True, exist_ok=True)

yolov7 = YOLOv7(
    weights=weights,
    cfg=cfg,
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

all_imgpaths = []
images_suffixes = ['.jpg', '.jpeg', '.png']

for suffix in images_suffixes:
    all_imgpaths.extend([imgpath for imgpath in Path(src_folder).glob(f"*{suffix}")])
all_imgs = [cv2.imread(str(imgpath)) for imgpath in all_imgpaths]

all_dets = yolov7.detect_get_box_in(all_imgs, box_format='ltrb', classes=None, buffer_ratio=0.0)
# print('detections: {}'.format(dets))

for idx, dets in enumerate(all_dets):
    draw_frame = all_imgs[idx].copy()
    print(f'img {all_imgpaths[idx].name}: {len(dets)} detections')
    for det in dets:
        # print(det)
        bb, score, class_ = det
        text = f"{class_}, {score:.2f}"
        l,t,r,b = bb
        color = (0, 0, 255)
        cv2.rectangle(draw_frame, (l,t), (r,b), color, 2)
        cv2.putText(draw_frame, text, (l, t+25), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, thickness=2)

    output_path = Path(output_folder) / f'{all_imgpaths[idx].stem}_det.jpg'
    cv2.imwrite(str(output_path), draw_frame)
