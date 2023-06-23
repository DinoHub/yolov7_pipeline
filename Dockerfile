# docker build -t yolov7 .

FROM nvidia/cuda:10.2-cudnn8-devel-ubuntu18.04

ENV DEBIAN_FRONTEND=noninteractive

ENV cwd="/home/"
WORKDIR $cwd

ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8
ENV TZ=Asia/Singapore
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# ENV TORCH_CUDA_ARCH_LIST="7.5 8.6"

RUN apt-get -y update
RUN apt-get -y upgrade
RUN apt -y update

RUN apt-get install --no-install-recommends -y \
    software-properties-common \
    build-essential \
    libgl1-mesa-glx

# upgrade python to version 3.9 (IMPT: remove python3-dev and python3-pip if already installed)
RUN apt-get remove -y python3.8-dev
RUN add-apt-repository ppa:deadsnakes/ppa && apt-get install --no-install-recommends -y python3.9-dev python3.9-venv python3-pip
RUN apt -y update
# Set python3.9 as the default python
RUN python3.9 -m venv /venv
ENV PATH=/venv/bin:$PATH

RUN apt-get clean && rm -rf /tmp/* /var/tmp/* /var/lib/apt/lists/* && apt-get -y autoremove

RUN rm -rf /var/cache/apt/archives/

### APT END ###

RUN python3 -m pip install --upgrade pip setuptools

COPY requirements.txt requirements.txt
RUN pip3 install --no-cache-dir -r requirements.txt

# fdet-api
RUN pip3 install git+https://github.com/yhsmiley/fdet-api.git#subdirectory=PythonAPI
