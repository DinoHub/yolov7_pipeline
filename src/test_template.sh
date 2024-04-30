NAME='yolov7-test'
DATA_CONFIG='/src/data/custom.yaml' # Remember to edit this file's "test" path and number of classes (nc)
WEIGHTS='/models/weights/best.pt'

IMG_SIZE=1280
BATCH_SIZE=32
CONF_THRES=0.001
IOU_THRES=0.65
DEVICE_NUM=0

# Only needed for test_coco.py
CFG='/src/cfg/deploy/yolov7.yaml' # Remember to edit this file's "test" path and number of classes (nc)

python test.py \
  --weights $WEIGHTS \
  --data $DATA_CONFIG \
  --img-size $IMG_SIZE \
  --batch-size $BATCH_SIZE \
  --conf-thres $CONF_THRES \
  --iou-thres $IOU_THRES \
  --save-json \
  --task 'test' \
  --device $DEVICE_NUM \
  --name $NAME

# python3 test_coco.py \
#     --weights $WEIGHTS \
#     --data $DATA_CONFIG \
#     --img-size $IMG_SIZE \
#     --batch-size $BATCH_SIZE \
#     --conf-thres $CONF_THRES \
#     --iou-thres $IOU_THRES \
#     --save-json \
#     --task "test" \
#     --device $DEVICE_NUM \
#     --name $NAME \
#     --cfg $CFG \
#     --evaluate-fbeta