FROM meadml/cuda10.1-cudnn7-devel-ubuntu18.04-python3.6
RUN apt update -y

ENV PATH="/root/miniconda3/bin:${PATH}"
ARG PATH="/root/miniconda3/bin:${PATH}"

RUN apt-get install -y wget

RUN wget \
    https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
    && mkdir /root/.conda \
    && bash Miniconda3-latest-Linux-x86_64.sh -b \
    && rm -f Miniconda3-latest-Linux-x86_64.sh 
RUN conda init
RUN conda --version
RUN conda update --all

RUN conda create -y -n editline python=3.6

RUN conda install --name editline --yes -c pytorch\
    pytorch=1.7.1 \
    torchvision \
    cudatoolkit=10.2

RUN conda install --name editline --yes -c conda-forge -c fvcore fvcore

RUN conda install --name editline --yes jupyter
RUN conda install --name editline --yes flask
RUN conda install --name editline --yes requests
RUN conda install --name editline --yes pillow

#RUN pip install --upgrade pip && \
RUN /bin/bash -c ". activate editline && \
    pip install --upgrade pip && \
    pip install \
    tensorly \
    chumpy \
    scikit-image \
    matplotlib \
    imageio \
    black \
    isort \
    dill \
    flake8 \
    flake8-bugbear \
    flake8-comprehensions"

RUN /bin/bash -c ". activate editline && \
    conda install \
    opencv \
    matplotlib \
    scipy"

RUN conda install --name editline -c fvcore -c iopath -c conda-forge fvcore iopath
RUN conda install --name editline pytorch3d -c pytorch3d
RUN /bin/bash -c ". activate editline"
