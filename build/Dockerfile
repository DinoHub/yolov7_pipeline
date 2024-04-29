FROM nvcr.io/nvidia/pytorch:23.06-py3

ENV DEBIAN_FRONTEND=noninteractive

ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8
ENV TZ=Asia/Singapore
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

RUN apt-get -y update
RUN apt-get -y upgrade
RUN apt -y update

RUN apt-get install --no-install-recommends -y \
    software-properties-common \
    build-essential \
    libgl1-mesa-glx libgl1-mesa-dev \
    vim nano ffmpeg

RUN apt-get clean && rm -rf /tmp/* /var/tmp/* /var/lib/apt/lists/* && apt-get -y autoremove

RUN rm -rf /var/cache/apt/archives/

### APT END ###
RUN apt-get update && apt-get install -y python3-pip
RUN python3 -m pip install --upgrade pip setuptools

ADD build/ /build
WORKDIR /build

# Fix issue wrt cv2's VideoCapture() function not working. Reference issue & solution: https://github.com/NVIDIA/DALI/issues/5052
RUN pip uninstall -y cv2 && \
    python_path="/usr/local/lib/python3.10/dist-packages/cv2" && \
    rm -rf $python_path

RUN pip install -r requirements.txt

# yolov7-inference
ADD src/ /src
WORKDIR /src
RUN python3 -m pip install --no-cache-dir .

# sahi
WORKDIR /
RUN git clone https://github.com/tehwenyi/sahi_general && cd sahi_general && python3 -m pip install --no-cache-dir .

WORKDIR /src