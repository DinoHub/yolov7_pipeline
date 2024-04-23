FROM nvcr.io/nvidia/pytorch:23.06-py3

ENV DEBIAN_FRONTEND=noninteractive

ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8
ENV TZ=Asia/Singapore
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

RUN apt-get -y update \
    && apt-get -y upgrade

RUN apt-get install --no-install-recommends -y --fix-missing \
    software-properties-common \
    build-essential \
    libgl1-mesa-glx \
    git ffmpeg vim nano

RUN apt-get clean && rm -rf /tmp/* /var/tmp/* /var/lib/apt/lists/* && apt-get -y autoremove

RUN rm -rf /var/cache/apt/archives/

### APT END ###
RUN apt-get update && apt-get install -y python3-pip
RUN python3 -m pip install --upgrade pip setuptools

ADD build/ /build
WORKDIR /build

RUN pip install -r requirements.txt

# fdet-api
RUN pip3 install git+https://github.com/yhsmiley/fdet-api.git#subdirectory=PythonAPI

WORKDIR /src
