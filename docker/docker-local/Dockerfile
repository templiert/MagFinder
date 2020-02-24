ARG CUDA="10.0"
ARG CUDNN="7"

FROM nvidia/cuda:${CUDA}-cudnn${CUDNN}-devel-ubuntu16.04

ARG USERNAME=myuser
ARG PASSWORD=1234

RUN echo 'debconf debconf/frontend select Noninteractive' | debconf-set-selections

# install basics
RUN apt-get update -y \
 && apt-get install -y apt-utils git curl ca-certificates bzip2 cmake tree htop bmon iotop g++ \
 && apt-get install -y libglib2.0-0 libsm6 libxext6 libxrender-dev

# Install Miniconda
RUN curl -so /miniconda.sh https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh \
 && chmod +x /miniconda.sh \
 && /miniconda.sh -b -p /miniconda \
 && rm /miniconda.sh

ENV PATH=/miniconda/bin:$PATH

# Create a Python 3.7 environment
RUN /miniconda/bin/conda install -y conda-build \
 && /miniconda/bin/conda create -y --name py37 python=3.7 \
 && /miniconda/bin/conda clean -ya

ENV CONDA_DEFAULT_ENV=py37
ENV CONDA_PREFIX=/miniconda/envs/$CONDA_DEFAULT_ENV
ENV PATH=$CONDA_PREFIX/bin:$PATH
ENV CONDA_AUTO_UPDATE_CONDA=false

RUN conda install -y ipython
RUN pip install ninja yacs cython matplotlib opencv-python tqdm pandas flask pyyaml pillow scikit-learn

# Install PyTorch 1.0 Nightly
RUN conda install -y pytorch=1.1.0 cuda100 -c pytorch \
 && conda clean -ya

# Install TorchVision master
RUN conda install -y torchvision==0.2.2 -c pytorch

# install pycocotools
RUN git clone https://github.com/cocodataset/cocoapi.git \
 && cd cocoapi/PythonAPI \
 && python setup.py build_ext install

# install apex
RUN git clone https://github.com/NVIDIA/apex.git \
 && cd apex \
 && python setup.py install --cuda_ext --cpp_ext

# install PyTorch Detection
ARG FORCE_CUDA="1"
ENV FORCE_CUDA=${FORCE_CUDA}
RUN git clone https://${USERNAME}:${PASSWORD}@bitbucket.org/tomgratte/sectionsegmentationml.git \
 && cd sectionsegmentationml \
 && python setup.py build develop

# isnatl Cloud SDK
RUN curl https://dl.google.com/dl/cloudsdk/channels/rapid/downloads/google-cloud-sdk-251.0.0-linux-x86_64.tar.gz | tar -xz
RUN ./google-cloud-sdk/install.sh --path-update true -q
RUN ./google-cloud-sdk/bin/gsutil cp gs://sectionsegmentationdata/PIPELINE_SETUP/model_final.pkl /sectionsegmentationml/PIPELINE_SETUP/.

# install app engine for bucket access
RUN apt-get update && apt-get install -y curl lsb-release sudo rsync
RUN export GCSFUSE_REPO=gcsfuse-`lsb_release -c -s` \
    && echo "deb http://packages.cloud.google.com/apt $GCSFUSE_REPO main" | tee /etc/apt/sources.list.d/gcsfuse.list \
    && curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | apt-key add -
RUN apt-get update  && apt-get install -y gcsfuse

# set running environments
RUN mkdir -p /sectionsegmentationml/bucket
RUN echo "gcsfuse sectionsegmentationdata /sectionsegmentationml/bucket" >> /root/.bashrc

WORKDIR /sectionsegmentationml
