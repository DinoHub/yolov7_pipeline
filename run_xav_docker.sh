WORKSPACE=/home/hopermfjetson/realsense

xhost +local:docker
docker run -it --rm --net=host \
	--runtime nvidia \
	--platform=linux/arm64 \
	--privileged \
	-w $WORKSPACE \
	-v $WORKSPACE:$WORKSPACE \
	--ipc=host \
	-v $HOME/.Xauthority:/root/.Xauthority:rw \
	-v /tmp/.X11-unix:/tmp/.X11-unix \
	-e DISPLAY=unix$DISPLAY \
	-e QT_X11_NO_MITSHM=1 \
	yolov7_realsense:ver0.2_ros1
