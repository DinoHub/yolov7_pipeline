DATA_SOURCE='/data/test.mp4' # Can be image or video, but tracking is not available for images

# Path to reparameterized + save_state_dict weights (refer to main branch)
WEIGHTS_FOLDER='/models/weights/'
WEIGHTS='best.pt'

# Inference parameters
CONF=0.6
IMG_SIZE=1280

NAME=$WEIGHTS'_conf'$CONF
DEVICE_NUM=0

python detect.py \
    --weights $WEIGHTS_FOLDER$WEIGHTS'.pt' \
    --conf $CONF \
    --img-size $IMG_SIZE \
    --source $DATA_SOURCE \
    --name $NAME \
    --device $DEVICE_NUM

# python detect_track.py \
#     --weights $WEIGHTS_FOLDER$WEIGHTS'.pt' \
#     --conf $CONF \
#     --img-size $IMG_SIZE \
#     --source $DATA_SOURCE \
#     --name $NAME \
#     --device $DEVICE_NUM