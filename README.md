# YOLOv7 Package with RealSense

## Table of Contents

1. [Overview](#overview)
2. [Setting Up on NVIDIA Jetson (Xavier/Orin)](#setting-up-on-nvidia-jetson-xavierorin)
    1. [Prerequisites](#prerequisites)
    2. [Setup Instructions](#setup-instructions)
3. [Quick Start (Running a YOLOv7 Inference with RealSense Camera)](#quick-start-running-a-yolov7-inference-with-realsense-camera)
    1. [Option 1: Using `rospy`](#option-1-using-rospy)
    2. [Option 2: Using `pyrealsense2` Wrapper](#option-2-using-pyrealsense2-wrapper)
4. [Scripts](#scripts)
4. [Others (For XFS)](#others-for-xfs)

## Overview

### Adapted/Forked from WongKinYiu's Repository

- Repository Link: https://github.com/WongKinYiu/yolov7
- Last "merge" date: 7th Sept 2022

### Updates to the Repository

The following changes have been made to the original repository:

- YOLOv7 Modification:
    - YOLOv7 has been modified to function as a **package** solely for inference purposes.
    - Modifications were made to this `edge` branch to enable compatibility with **RealSense cameras** (through `pyrealsense2` and/or `rospy`).

- ARM Processor Support:
    - This repository is **compatible with ARM processors**, allowing users to run the inference process on these devices.

## Setting Up on NVIDIA Jetson (Xavier/Orin)

This program should work on ARM processors in general but has only been tested on the NVIDIA Jetson Xavier.

### Prerequisites
- Ubuntu OS
- RealSense Camera
- For Jetson Xavier: External hard disk with Linux File System (eg. ext4) for extra storage

### Setup Instructions

Please execute the following instructions in order when setting up a new NVIDIA Jetson Xavier.

1. **Flashing Ubuntu**

    Please note that this repository is compatible only with Jetpack versions **5.0.2** and later (but only tested on 5.0.2).

    If you are using **Jetson Xavier**, follow the provided guide: [Jetson Xavier AGX Driver Installation](https://wiki.seeedstudio.com/Jetson_Xavier_AGX_H01_Driver_Installation/) for step-by-step instructions on updating your Jetson device to the most recent Jetpack version.

    For **Jetson Orin** users, refer to this guide [Jetson AGX Orin Driver Installation](https://developer.nvidia.com/embedded/learn/get-started-jetson-agx-orin-devkit) for step-by-step instructions on updating your Jetson device to the most recent Jetpack version.

1. **Date & Time Adustment (if applicable)**

    In the event that the date and time on the NVIDIA Jetson Xavier device are inaccurate, please make sure to adjust them each time the device is rebooted. Otherwise, there may be complications during program installation and execution.

1. **External Disk Automount Configuration**

    Automount the SSD/HDD/SD card onto `/mnt/rootfs`. Ensure that the disk is of Linux File System type.

    1. Identify the **UUID** and **file system type** of your drive by executing the following command:
        ```bash
        sudo blkid
        ```

    1. Create a mount point for your drive under the `/mnt` directory. In this example, we will use `/mnt/rootfs`.
        ```bash
        sudo mkdir /mnt/rootfs
        ```

    1. Append the following line to the `/etc/fstab` file using your preferred text editor:
        ```
        UUID=<uuid-of-your-drive>  <mount-point>  <file-system-type>  <mount-option>  <dump>  <pass>
        ```
        For example,
        ```bash
        UUID=eb67c479-962f-4bcc-b3fe-cefaf908f01e  /mnt/rootfs  ext4  defaults  0  2
        ```

    1. Verify the automount configuration by executing the following command:
        ```bash
        sudo mount -a
        ```

    For additional information and details, please refer to the following [link](https://www.linuxbabe.com/desktop-linux/how-to-automount-file-systems-on-linux).

1. **Docker Installation**

    Before proceeding with the installation of Docker, ensure that no other versions of Docker are currently installed. If any are found, please **uninstall** them prior to continuing. 
    1. Download and install Docker for ARM using the following commands:
        ```bash
        sudo apt-get update
        sudo apt-get upgrade
        curl -fsSL test.docker.com -o get-docker.sh && sh get-docker.sh
        sudo usermod -aG docker $USER 
        ```

    1. Log out of the system and then log back in to apply the group membership changes. Verify the successful installation of Docker by running the following command:
        ```bash
        docker run hello-world 
        ```
    For more detailed information and instructions, please refer to the following [link](https://www.docker.com/blog/getting-started-with-docker-for-arm-on-linux/)

1. **NVIDIA Container Toolkit Installation (for GPU Usage)**

    1. To install the NVIDIA Container Toolkit on the NVIDIA Jetson Xavier, execute the following commands:
        ```bash
        distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
        curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
        curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list

        sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
        sudo apt install -y nvidia-docker2
        sudo systemctl daemon-reload
        sudo systemctl restart docker
        ```
    
    Note: When prompted to allow changes to `/etc/docker/daemon.json` during installation, accept the changes.
    For more detailed information and instructions, please refer to the following [link](https://dev.to/caelinsutch/running-docker-containers-for-the-nvidia-jetson-nano-5a06)

1. **Relocating Docker Storage (if required) - Insufficient Storage Capacity**

    Due to the limited storage capacity of the NVIDIA Jetson Xavier (16GB), it may become necessary to relocate Docker's storage location to an external hard disk. Follow the steps below:

    1. Stop the Docker daemon
        ```bash
        sudo service docker stop
        ```
    1. Open the `/etc/docker/daemon.json` file using a text editor and add the following JSON configuration:
        ```json
        {
            "data-root": "/path/to/your/docker"
        }
        ```
        Replace `/path/to/your/docker` with the desired path on the external hard disk. 
        
        Note that if the file already contains other information, make sure to include a comma at the end of the "data-root" statement.

    1. Create a new directory on the external hard disk to store Docker's data and transfer the data over. For instance, we will use the directory `/mnt/rootfs/docker`.
        ```bash
        sudo mkdir /mnt/rootfs/docker
        sudo rsync -aP /var/lib/docker/ /mnt/rootfs/docker
        sudo mv /var/lib/docker /var/lib/docker.old
        sudo service docker start
        ```

    For more detailed information and instructions, please refer to the following [link](https://www.guguweb.com/2019/02/07/how-to-move-docker-data-directory-to-another-location-on-ubuntu/).

1. **Installing `tmux` (if required)**

    Tmux is a terminal multiplexer that allows you to create and manage multiple terminal sessions within a single window. It's particularly useful for projects where you need to detach your terminal session and reattach it later, so your session will continue running while your  SSH connection is intermittent or disconnected. With tmux, you can keep processes running in the background and re-access them whenever needed.

    To install tmux, execute the following command:
    ```
    sudo apt install tmux
    ```

    To run `tmux`, execute the following command:
    ```
    tmux
    ```
    You can then run your code in `tmux` and detach and reattach wherever necessary.

    For more information on the `tmux` installation and its commands, please refer to this [link](https://linuxize.com/post/getting-started-with-tmux/#google_vignette).

## Quick Start (Running a YOLOv7 Inference with RealSense Camera)

This guide provides instructions on running YOLOv7 inference using a RealSense camera via either `rospy` or `pyrealsense2`. Keep in mind that when using the `pyrealsense2` wrapper, other scripts will not be able to access the same camera (either via `rospy`/`pyrealsense2`).

### Option 1: Using `rospy`

Follow the steps below to set up and run the program.

##### Running the Program

1. Clone this repository and switch to the `edge` branch
1. **Option 1: Build the Docker image:** Execute the following command to build the Docker image:
    ```
    docker build -f Dockerfile.ros -t yolov7-realsense-arm .
    ```
    **Option 2: Load the saved Docker image:** Run the following command to load the pre-built Docker image:
    ```
    docker load << DOCKER_IMG_NAME.tar.gz
    ```
    Replace `DOCKER_IMG_NAME.tar.gz` with the saved Docker image.

1. Create and enter the Docker container using the provided script:
    ```
    bash run_xav_docker.sh
    ```

1. Import the wrapper classes for YOLOv7 and pyrealsense2 for inference, or use the provided scripts for inference with the RealSense camera:
    - Import the classes manually in your script:
        ```
        import rospy
        from yolov7.yolov7 import YOLOv7
        ```
    - Alternatively, you can use the [provided scripts below](#scripts)

### Option 2: Using `pyrealsense2` Wrapper

Keep in mind that when using the `pyrealsense2` wrapper, other scripts will not be able to access the same camera (either via `rospy`/`pyrealsense2`). Follow the steps below to set up and run the program.

#### Running the Program
Follow the steps below to run a live YOLOv7 inference script using a RealSense camera:

1. Clone this repository and switch to the `edge` branch
1. **Option 1: Build the Docker image:** Execute the following command to build the Docker image:
    ```
    docker build -f Dockerfile.pyrs -t yolov7-realsense-arm .
    ```
    **Option 2: Load the saved Docker image:** Run the following command to load the pre-built Docker image:
    ```
    docker load << DOCKER_IMG_NAME.tar.gz
    ```
    Replace `DOCKER_IMG_NAME.tar.gz` with the saved Docker image.

1. Create and enter the Docker container using the provided script:
    ```
    bash run_xav_docker.sh
    ```
1. Import the wrapper classes for YOLOv7 and pyrealsense2 for inference, or use the provided scripts for inference with the RealSense camera:
    - Import the classes manually in your script:
        ```
        import pyrealsense2 as rs
        from yolov7.yolov7 import YOLOv7
        ```
    - Alternatively, you can use the [provided scripts below](#scripts)

Notes: 
- If the RealSense camera connected via USB is not functioning, ensure that it is connected to a USB 3.0 port rather than a USB 2.0 port.
- For debugging purposes, you can download and use the [realsense-viewer application](https://dev.intelrealsense.com/docs/nvidia-jetson-tx2-installation).

### Camera Configuration

The camera configuration, including fps, width, and height, may vary for each RealSense camera. In our tests, we used the **RealSense camera D455** on the NVIDIA Jetson Xavier, which runs at a maximum of:

- 1280x720: ~6fps (use 5fps)
- 640x480: ~8fps (use 5fps)

To determine the appropriate configuration for your camera, you can run a test using the respective bash scripts with your desired width and height parameters. Take note of the **fps** value printed out during the test. It is important to use this fps value as the input for both your desired width and height parameters when running the inference script. Failure to provide the correct fps will result in the saved videos (inference, raw) being fast-forwarded or slowed down.

## Scripts

The scripts are located within the `scripts` folder. Each script has an associated shell script that can be executed with bash (e.g., `bash run_inference_ros.sh`). Feel free to customize both the parameters and scripts to suit your specific use case.

Here is a list of the default scripts available:

### 1. Inferencing using pyrealsense2 wrapper

**Script**: `inference_pyrs.py`

**Corresponding bash script**: `run_inference_pyrs.sh`

- This script utilizes the `pyrealsense2` library to access images from the video stream.
- The script contains several parameters categorised into three groups: General, YOLOv7 and Video settings. For comprehensive details about these parameters, consult the script's code.

### 2. Inferencing using rospy

**Script**: `inference_ros.py`

**Corresponding bash script**: `run_inference_ros.sh`

- This script utilizes the `rospy` library to subscribe to the rostopic publishing the images.
- Similar to the previous script, it includes several parameters categorized into General, YOLOv7, and Video Settings. For comprehensive details about these parameters, refer to the script's code.

### 3. Publisher code (for rospy)

**Script**: `publisher.py`

Used by both scripts above to publish the following messages to the following topic(s):
1. 'bbox': ('/cv/bounding_box', BoundingBox2DArray),
1. 'chip': ('/cv/target_chip', Image),
1. 'depth_frame': ('/cv/depth_frame', Image) (incomplete)
1. 'cam_info': ('/cv/camera_info', CameraInfo)

To change any of the topics, edit Line 30 in `publisher.py` directly. Note that messages are only published when there is a detection (i.e. no message will be published for a frame with no detections).

### 4. Subscriber code (for rospy)

**Script**: `subscriber.py`

Used by `inference_ros.py`. The default topic is `/camera/color/image_raw`, edit Line 77 of `inference_ros.py` to change the subscription topic.

## Others (For XFS)

### For testing purposes

**Note that you should ONLY do this if the NUC has not initialised the following**

- Creating the `roscore`
    ```
    export ROS_MASTER_URI=http://192.168.168.105:11311
    ROS_IP=192.168.168.102
    export ROS_DOMAIN_ID=123
    export CYCLONEDDS_URI="<CycloneDDS><Domain><General><NetworkInterfaceAddress>$(ifconfig | awk '/192.168.168.102/ {print a}{a = $0}' | awk '{print $1}' | sed 's/.$//')</></></></>"

    roscore &
    ```
    You may proceed to kill this terminal.

- Starting up the FRAMOS RealSense Cameras:
    ```
    export ROS_MASTER_URI=http://192.168.168.105:11311
    ROS_IP=192.168.168.102
    export ROS_DOMAIN_ID=123
    export CYCLONEDDS_URI="<CycloneDDS><Domain><General><NetworkInterfaceAddress>$(ifconfig | awk '/192.168.168.102/ {print a}{a = $0}' | awk '{print $1}' | sed 's/.$//')</></></></>"

    cd /path/to/catkin_ws
    source devel/setup.bash
    roslaunch realsense2_camera framos_rs_camera.launch
    ```
    Replace `/path/to/catkin_ws` with the path to your `catkin_ws` folder. You may use `find ~/ -name "*catkin_ws*"` to find the folder location.