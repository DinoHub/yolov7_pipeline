WEIGHTS='/mnt/rootfs/realsense/yolov7/weights/reparam_state.pt'
CFG='/mnt/rootfs/realsense/yolov7/cfg/deploy/yolov7x.yaml'
INFERENCE_FOLDER='/mnt/rootfs/realsense/scripts/output/inference_videos'
RAW_VIDEO_FOLDER='/mnt/rootfs/realsense/scripts/output/raw_videos'

WIDTH=640
HEIGHT=480
FPS=6

python3 inference_pyrs.py \
  -w $WEIGHTS \
  -c $CFG \
  --bgr \
  --gpu_device 0 \
  --model_image_size 640 \
  --max_batch_size 64 \
  --conf_thresh 0.25 \
  --nms_thresh 0.45 \
  --width $WIDTH \
  --height $HEIGHT \
  --fps $FPS \
  --same_size \
  --publish-bbox \
  # --inference-folder $INFERENCE_FOLDER \
  # --raw-video-folder $RAW_VIDEO_FOLDER \
  # --display \
  # --publish-img-chip \
  # --publish-depth-params
