FROM nvidia/cuda:10.0-cudnn7-devel-ubuntu18.04

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        curl \
        pkg-config \
        rsync \
        software-properties-common \
        unzip \
        && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN apt-get update && \
        apt-get install nvinfer-runtime-trt-repo-ubuntu1804-5.0.2-ga-cuda10.0 && \
        apt-get update && \
        apt-get install libnvinfer5=5.0.2-1+cuda10.0

RUN file="$(ls -1 /usr/local/)" && echo $file

RUN apt-get -y update
RUN apt-get install -y --fix-missing \
    cmake \
    gfortran \
    git \
    wget \
    curl \
    graphicsmagick \
    libgraphicsmagick1-dev \
    libavcodec-dev \
    libavformat-dev \
    libboost-all-dev \
    libopenblas-dev \ 
    pbzip2 \
    libgtk2.0-dev \
    libjpeg-dev \
    liblapack-dev \
    libswscale-dev \
    pkg-config \
    python3-numpy \
    zip \    
    && apt-get clean && rm -rf /tmp/* /var/tmp/*

RUN add-apt-repository ppa:jonathonf/python-3.6
RUN apt-get update
RUN apt-get install python3.6 libpython3.6

# Install Anaconda

ARG PYTHON_VERSION=3.6
ARG CONDA_VERSION=3
ARG CONDA_PY_VERSION=4.5.11

ENV PATH /opt/conda/bin:$PATH
RUN wget — quiet https://repo.anaconda.com/miniconda/Miniconda$ CONDA_VERSION-$ CONDA_PY_VERSION-Linux-x86_64.sh -O ~/miniconda.sh && \
    /bin/bash ~/miniconda.sh -b -p /opt/conda && \
    rm ~/miniconda.sh && \
    /opt/conda/bin/conda clean -tipsy && \
    ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh && \
    echo “. /opt/conda/etc/profile.d/conda.sh” >> ~/.bashrc && \
    echo “conda activate base” >> ~/.bashrc

# Create a conda environment to use the h2o4gpu
RUN conda update -n base -c defaults conda && \
    conda create -y -n gpuenvs -c h2oai -c conda-forge h2o4gpu-cuda9

RUN conda install -c conda-forge rdkit

# You can add the new created environment to the path
#ENV PATH /opt/conda/envs/gpuenvs/bin:$PATH

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

RUN pip install future
RUN pip install cupy-cuda100
RUN pip install pycuda
RUN pip install tensorflow-serving-api==1.9.0
RUN pip install "dask[complete]" 
RUN pip install featuretools
RUN pip install h2o4gpu-0.3.2-cp36-cp36m-linux_x86_64.whl
RUN pip install modred
RUN pip install tqdm
RUN pip install xgboost
RUN pip install seaborn
RUN pip install pytest
RUN pip install pytest-cov
RUN pip install ipython
RUN pip install ipykernel
RUN pip install scipy
RUN pip install scikit-learn
RUN pip install pytorch
RUN pip install torchvision

RUN git clone https://github.com/Mariewelt/OpenChem.git
RUN cd Openchem
RUN conda install --yes --file requirements.txt
RUN conda install -c rdkit rdkit nox cairo
RUN conda install pytorch torchvision -c pytorch

# Some TF tools expect a "python" binary
RUN ln -s $(which ${PYTHON}) /usr/local/bin/python 

RUN pip install tensorflow-gpu

#COPY bashrc /etc/bash.bashrc
#RUN chmod a+rwx /etc/bash.bashrc

RUN ${PIP} --no-cache-dir install jupyter matplotlib pyinstrument
# RUN ${PIP} install jupyter matplotlib o pencv-python opencv-contrib-python pyinstrument

# Core linux dependencies. 
RUN apt-get install -y --fix-missing \
        build-essential \
        cmake \
    	curl \
	gfortran \
    	graphicsmagick \
        git \
        wget \
        unzip \
        yasm \
        pkg-config \
        libswscale-dev \
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
	libboost-all-dev \
	libgtk2.0-dev \
	liblapack-dev \
        liblapacke-dev \
	libswscale-dev \
	libcanberra-gtk-module \
        libboost-dev \
        libboost-all-dev \
        libeigen3-dev \
	python3-dev \
	python3-numpy \
	python3-scipy \
	software-properties-common \
	zip \
        vim \
        qt5-default \
	&& apt-get clean && rm -rf /tmp/* /var/tmp/*

RUN ${PIP} --no-cache-dir install \
    hdf5storage \
    h5py \
    py3nvml \
    scikit-image \
    scikit-learn

WORKDIR /

# dlib
RUN cd ~ && \
    mkdir -p dlib && \
    git clone -b 'v19.16' --single-branch https://github.com/davisking/dlib.git dlib/ && \
    cd  dlib/ && \
    python3 setup.py install --yes USE_AVX_INSTRUCTIONS --yes DLIB_USE_CUDA --clean


RUN mkdir -p /tf/tensorflow-tutorials && chmod -R a+rwx /tf/
RUN mkdir /.local && chmod a+rwx /.local
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
