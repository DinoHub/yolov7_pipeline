# IMG_FOLDER='/mnt/c/Users/wenyi/Documents/Blender/synth_dataset/oct-bg-medium/img'
# JSON_PATH='/mnt/c/Users/wenyi/Documents/Blender/synth_dataset/oct-bg-medium/coco_annotations.json'
# OUTPUT_PATH='/mnt/c/Users/wenyi/Documents/XFS/dog/datasets/oct_bg_medium'
# NS_PATH='/mnt/c/Users/wenyi/Documents/XFS/dog/datasets/oct_bg_medium_NS'
# OUTPUT_TILED_PATH='/mnt/c/Users/wenyi/Documents/XFS/dog/datasets/oct_bg_medium'
IMG_FOLDER='/mnt/c/Users/wenyi/Documents/XFS/dog/datasets/2-dogs_moving_medium-to-far_outdoors/images'
JSON_PATH='/mnt/c/Users/wenyi/Documents/XFS/dog/datasets/2-dogs_moving_medium-to-far_outdoors/2-dogs_moving_medium-to-far_outdoors_annotations.json'
OUTPUT_PATH='/mnt/c/Users/wenyi/Documents/XFS/dog/datasets/2-dogs_moving_medium-to-far_outdoors/yolo'

# For resizing images
WIDTH=1280
HEIGHT=720
OUTPUT_RESIZE_PATH='/mnt/c/Users/wenyi/Documents/XFS/dog/datasets/oct_bg_medium_1280x720'

python coco_to_yolo.py \
  $IMG_FOLDER \
  $JSON_PATH \
  $OUTPUT_PATH

# Optional: Further split into train, val & test folders
# Remember to delete "images" and "labels" from $OUTPUT_PATH
# python split_train_val_test.py \
#   $OUTPUT_PATH \
#   $OUTPUT_PATH
#   --test True

# # Optional: Tiling
# python tiling.py \
#   $OUTPUT_PATH \
#   $OUTPUT_TILED_PATH \
#   --negative-samples-path $NS_PATH \
#   --size 640

# # Optional: Resize images
# python resize_images.py \
#   --input-dir $OUTPUT_PATH \
#   --output-dir $OUTPUT_RESIZE_PATH \
#   --width $WIDTH \
#   --height $HEIGHT
