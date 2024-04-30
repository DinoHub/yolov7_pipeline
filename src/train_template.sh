# Note: Results will be saved in runs/train
EXPR_PROJECT='yolov7-test'

EXPR_BATCHSIZE=32
EXPR_EPOCHS=300
EXPR_IMGSIZE=1280

# IMPORTANT: 
# 1) Edit your 'DATA' file so that it points to your dataset
# 2) Ensure your 'CFG' file is correct, including 'nc' (number of classes)
DATA='/src/data/custom.yaml'
CFG='/src/cfg/training/yolov7.yaml'
HYP='/src/data/hyp.scratch.custom.yaml'
WEIGHTS='weights/yolov7_training.pt'

SAVE_PERIOD=50
NUM_WORKERS=8
DEVICE=0

# IMPORTANT: Note that if --rect is used, set "loss_ota=0.0" in data/hyp.scratch.custom.yaml. Set to loss_ota=1.0 if doing square training
python train.py \
  --workers $NUM_WORKERS \
  --device $DEVICE \
  --batch-size $EXPR_BATCHSIZE \
  --epochs $EXPR_EPOCHS \
  --data $DATA \
  --img-size $EXPR_IMGSIZE \
  --cfg $CFG \
  --weights $WEIGHTS \
  --hyp $HYP \
  --name $EXPR_PROJECT'_'$EXPR_IMGSIZE'_bs'$EXPR_BATCHSIZE'_e'$EXPR_EPOCHS \
  --save_period $SAVE_PERIOD \
  # --rect