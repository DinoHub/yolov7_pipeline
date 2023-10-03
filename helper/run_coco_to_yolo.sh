IMG_FOLDER='/path/to/img/folder'
JSON_PATH='/path/to/json/annotations.json'
OUTPUT_PATH='/path/to/output/path'
NS_PATH='/path/to/ns/path'
OUTPUT_TILED_PATH='/path/to/tiled/output/path'

# For resizing images
WIDTH=1280
HEIGHT=720
OUTPUT_RESIZE_PATH='/path/to/resize/output/path'

python coco_to_yolo.py \
  $IMG_FOLDER \
  $JSON_PATH \
  $OUTPUT_PATH

# Optional: Further split into train, val & test folders
# Remember to delete "images" and "labels" from $OUTPUT_PATH
python split_train_val_test.py \
  $OUTPUT_PATH \
  $OUTPUT_PATH
  --test True

# Optional: Tiling
python tiling.py \
  $OUTPUT_PATH \
  $OUTPUT_TILED_PATH \
  --negative-samples-path $NS_PATH \
  --size 640

# Optional: Resize images
python resize_images.py \
  --input-dir $OUTPUT_PATH \
  --output-dir $OUTPUT_RESIZE_PATH \
  --width $WIDTH \
  --height $HEIGHT
