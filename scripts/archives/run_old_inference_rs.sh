WEIGHTS='/home/hopermfjetson/Downloads/realsense/yolov7/weights/reparam_state.pt'
CFG='/home/hopermfjetson/Downloads/realsense/yolov7/cfg/deploy/yolov7x.yaml'
INFERENCE_FOLDER='/home/hopermfjetson/Downloads/realsense/xavier/output/inference_videos'
RAW_VIDEO_FOLDER='/home/hopermfjetson/Downloads/realsense/xavier/output/raw_videos'

WIDTH=640
HEIGHT=480
FPS=6

python3 old_inference_rs.py \
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
  --same_size
  #--inference-folder $INFERENCE_FOLDER \
  #--raw-video-folder $RAW_VIDEO_FOLDER \
