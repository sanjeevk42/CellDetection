FROM tensorflow/tensorflow:0.12.1-gpu

RUN apt-get update -y &&  apt-get install -y \
    libgeos-dev \
    python-pip \
    python-tk

RUN pip install \
    opencv-python \
    scikit-learn \
    click keras==1.2.2 \
    shapely \
    munkres \
    h5py \
    fire

