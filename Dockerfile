FROM nvidia/cuda:10.0-cudnn7-devel-ubuntu18.04

# Core Linux Deps
RUN DEBIAN_FRONTEND=noninteractive apt-get update && apt-get install -y --fix-missing --no-install-recommends \
        build-essential \
    	cmake \
        curl \
        git \
	gfortran \
        pkg-config \
        pbzip2 \
        rsync \
        software-properties-common \
        libboost-all-dev \
        libopenblas-dev \ 
        libtbb2 \
        libtbb-dev \
        libjpeg-dev \
        libpng-dev \
        libtiff-dev \
	libgraphicsmagick1-dev \
        libavformat-dev \
        libhdf5-dev \
        libpq-dev \
	libgraphicsmagick1-dev \
	libavcodec-dev \
	libgtk2.0-dev \
	liblapack-dev \
        liblapacke-dev \
	libswscale-dev \
	libcanberra-gtk-module \
        libboost-dev \
        libeigen3-dev \
	wget \
        vim \
        qt5-default \
        unzip \
	zip \ 
        && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*  && \
    apt-get clean && rm -rf /tmp/* /var/tmp/*


# Install TensorRT (TPU Access)
RUN apt-get update && \
        apt-get install nvinfer-runtime-trt-repo-ubuntu1804-5.0.2-ga-cuda10.0 && \
        apt-get update && \
        apt-get install libnvinfer5=5.0.2-1+cuda10.0

RUN file="$(ls -1 /usr/local/)" && echo $file


# Install Anaconda


RUN wget --quiet https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh && \
/bin/bash Miniconda3-latest-Linux-x86_64.sh -f -b -p /opt/conda && \
rm Miniconda3-latest-Linux-x86_64.sh
ENV PATH /opt/conda/bin:$PATH


# For CUDA profiling, TensorFlow requires CUPTI.
ENV LD_LIBRARY_PATH /usr/local/cuda/extras/CUPTI/lib64:$LD_LIBRARY_PATH


ARG PYTHON=python3
ARG PIP=pip3

# See http://bugs.python.org/issue19846
ENV LANG C.UTF-8

RUN apt-get update && apt-get install -y \
    ${PYTHON} \
    ${PYTHON}-pip

RUN ${PIP} --no-cache-dir install --upgrade \
    pip \
    setuptools

# Install H2O4GPU for CUDA 10
RUN conda create -n h2o4gpuenv -c h2oai -c conda-forge h2o4gpu-cuda92
RUN conda init bash
RUN /bin/bash -c conda activate h2o4gpuenv

# Install molecular descriptor calculator
RUN conda install -c rdkit -c mordred-descriptor mordred

# Install OpenChem
RUN git clone https://github.com/Mariewelt/OpenChem.git
RUN conda install --yes --file /OpenChem/requirements.txt
RUN conda install -c rdkit rdkit nox cairo
RUN conda install pytorch torchvision -c pytorch

# Some TF tools expect a "python" binary
RUN ln -s $(which ${PYTHON}) /usr/local/bin/python 

# Install TF for GPU
RUN pip install tensorflow-gpu

#COPY bashrc /etc/bash.bashrc
#RUN chmod a+rwx /etc/bash.bashrc

RUN ${PIP} --no-cache-dir install jupyter matplotlib pyinstrument

# Misc deps
RUN ${PIP} --no-cache-dir install \
    hdf5storage \
    h5py \
    py3nvml \
    scikit-learn \
    future \
    cupy-cuda100 \
    pycuda \
    "dask[complete]" \
    featuretools \
    tqdm \
    xgboost \
    seaborn \
    pytest \
    pytest-cov \
    ipython


RUN conda install -c anaconda ipykernel 

WORKDIR /

RUN mkdir -p /tf/tensorflow-tutorials && chmod -R a+rwx /tf/
RUN mkdir /.local && chmod a+rwx /.local
RUN chmod -R 777 /.local
RUN apt-get install -y --no-install-recommends wget
WORKDIR /tf/tensorflow-tutorials
RUN wget https://raw.githubusercontent.com/tensorflow/docs/master/site/en/tutorials/keras/basic_classification.ipynb
RUN wget https://raw.githubusercontent.com/tensorflow/docs/master/site/en/tutorials/keras/basic_text_classification.ipynb
RUN wget https://raw.githubusercontent.com/tensorflow/docs/master/site/en/tutorials/keras/overfit_and_underfit.ipynb
RUN wget https://raw.githubusercontent.com/tensorflow/docs/master/site/en/tutorials/keras/save_and_restore_models.ipynb

RUN apt-get autoremove -y && apt-get remove -y wget

WORKDIR /tf
EXPOSE 8888 6006

RUN useradd -ms /bin/bash container_user
RUN ${PYTHON} -m ipykernel.kernelspec

CMD ["bash", "-c", "source /etc/bash.bashrc && jupyter notebook --notebook-dir=/tf --ip 0.0.0.0 --no-browser --allow-root --NotebookApp.custom_display_url='http://localhost:8888'"]
