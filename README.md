### Drug Discovery using H2O4GPU 
ReLeaSE solution solved with: H2O4GPU, Anaconda, Docker, CUDA 10, RDKit, TensorFlow-GPU, OpenChem and more!

# Reinforcement Learning for Drug Discovery using H2O4GPU
Drug Discovery using H2O4GPU based off of the following paper: Deep ReLeaSE (Reinforcement Learning for de-novo Drug Design) by: 
Mariya Popova, Olexandr Isayev, Alexander Tropsha. *Deep Reinforcement Learning for de-novo Drug Design*. Science Advances, 2018, Vol. 4, no. 7, eaap7885. DOI: [10.1126/sciadv.aap7885](http://dx.doi.org/10.1126/sciadv.aap7885). Please note that this implementation of Deep Reinforcement Learning for de-novo Drug Design aka ReLeaSE method only works on Linux.


### Chemical and Molecule Analysis for Drug Discovery

![github-small](https://avatars0.githubusercontent.com/u/1402695?s=200&v=4)
![github-small](https://raw.githubusercontent.com/github/explore/80688e429a7d4ef2fca1e82350fe8e3517d3494d/topics/docker/docker.png)
![github-small](https://avatars2.githubusercontent.com/u/1158637?s=200&v=4)
![github-small](https://avatars0.githubusercontent.com/u/15658638?s=200&v=4)


* [H2O4GPU](https://github.com/h2oai/h2o4gpu): H2O4GPU is a collection of GPU solvers by H2Oai with APIs in Python and R. The Python API builds upon the easy-to-use scikit-learn API and its well-tested CPU-based algorithms. It can be used as a drop-in replacement for scikit-learn.

* [OpenChem](https://mariewelt.github.io/OpenChem/html/index.html): OpenChem is a deep learning toolkit for Computational Chemistry with PyTorch backend

* [RDKit](https://www.rdkit.org/docs/Install.html)
    - 2D and 3D molecular operations
    - Descriptor generation for machine learning
    - Molecular database cartridge for PostgreSQL
    - Cheminformatics nodes for KNIME (distributed from the KNIME community site: https://www.knime.com/rdkit)
    - TUTORIALS: ` https://github.com/rdkit/rdkit-tutorials `

* [Mordred](https://github.com/mordred-descriptor/mordred)
    - Compute modal decompositions and reduced-order models, easily, efficiently, and in parallel

* [TensorFlow for GPU v1.13.1](https://www.tensorflow.org/install/gpu): Machine Learning

* [TensorBoard](https://www.datacamp.com/community/tutorials/tensorboard-tutorial): Understand, debug, and optimize

* [PyTorch](https://pytorch.org/tutorials/): Neural Networks from research to production


### Distributed Feature Engineering 

* [Dask Distributed](https://dask.org/): Distributed ingestion of data

* [Feature Tools](https://docs.featuretools.com/): Automated feature engineering


### CUDA for GPU/TPU Enablement

* [NVIDIA TensorRT inference accelerator and CUDA 10](https://developer.nvidia.com/tensorrt): CUDA + TPUs makes you awesome

* [PyCUDA 2019](https://mathema.tician.de/software/pycuda/): Python interface for direct access to GPU or TPU

* [CuPy:latest](https://cupy.chainer.org/): GPU accelerated drop in replacement for numpy

* [cuDNN7.4.1.5 for deeep learning in CNN's](https://developer.nvidia.com/cudnn): GPU-accelerated library of primitives for deep neural networks

### Misc 

* [tqdm](https://github.com/tqdm/tqdm): Progess bars
 
### Operating System inside the container
* Ubuntu 18.04 so you can 'nix your way through the cmd line!

### Good to know
* Hot Reloading: code updates will automatically update in container from /apps folder.
* TensorBoard is on localhost:6006 and GPU enabled Jupyter is on localhost:8888.
* Python 3.6 (Stable & Secure)
* Only Tesla Pascal and Turing GPU Architecture are supported 
* Test with synthetic data that compares GPU to CPU benchmark, and Tensorboard example:
   
   1. [CPU/GPU Benchmark](https://github.com/joehoeller/ /blob/master/apps/gpu_benchmarks/benchmark.py)
   
   2. [Tensorboard to understand & debug neural networks](https://github.com/joehoeller/ /blob/master/apps/gpu_benchmarks/tensorboard.py)


-------------------------------------------------------------


## Demos

* JAK2_min_max_demo.ipynb -- [JAK2](https://www.ebi.ac.uk/chembl/target/inspect/CHEMBL2363062) pIC50 minimization and maximization
* LogP_optimization_demo.ipynb -- optimization of logP to be in a drug-like region 
from 0 to 5 according to [Lipinski's rule of five](https://en.wikipedia.org/wiki/Lipinski%27s_rule_of_five).
* RecurrentQSAR-example-logp.ipynb -- training a Recurrent Neural Network to predict logP from SMILES
using [OpenChem](https://github.com/Mariewelt/OpenChem) toolkit.

**Disclaimer**: JAK2 demo uses Random Forest predictor instead of Recurrent Neural Network,
since the availability of the dataset with JAK2 activity data used in the
"Deep Reinforcement Learning for de novo Drug Design" paper is restricted under
the license agreement. So instead we use the JAK2 activity data downladed from
ChEMBL (CHEMBL2971) and curated. The size of this dataset is ~2000 data points,
which is not enough to build a reliable deep neural network. If you want to see
a demo with RNN, please checkout logP optimization

-------------------------------------------------------------

### Before you begin (This might be optional) ###

Link to nvidia-docker2 install: [Tutorial](https://medium.com/@sh.tsang/docker-tutorial-5-nvidia-docker-2-0-installation-in-ubuntu-18-04-cb80f17cac65)

You must install nvidia-docker2 and all it's deps first, assuming that is done, run:


 ` sudo apt-get install nvidia-docker2 `
 
 ` sudo pkill -SIGHUP dockerd `
 
 ` sudo systemctl daemon-reload `
 
 ` sudo systemctl restart docker `
 

How to run this container:


### Step 1 ###

` docker build -t <container name> . `  < note the . after <container name>


### Step 2 ###

Run the image, mount the volumes for Jupyter and app folder for your fav IDE, and finally the expose ports `8888` for TF1, and `6006` for TensorBoard.


` docker run --rm -it --runtime=nvidia --user $(id -u):$(id -g) --group-add container_user --group-add sudo -v "${PWD}:/apps" -v $(pwd):/tf/notebooks  -p 8888:8888 -p 0.0.0.0:6006:6006  <container name> `


### Step 3: Check to make sure GPU drivers and CUDA is running ###

- Exec into the container and check if your GPU is registering in the container and CUDA is working:

- Get the container id:

` docker ps `

- Exec into container:

` docker exec -u root -t -i <container id> /bin/bash `

- Check if NVIDIA GPU DRIVERS have container access:

` nvidia-smi `

- Check if CUDA is working:

` nvcc -V `


### Step 4: How to launch TensorBoard ###

(It helps to use multiple tabs in cmd line, as you have to leave at least 1 tab open for TensorBoard@:6006)

- Demonstrates the functionality of TensorBoard dashboard


- Exec into container if you haven't, as shown above:


- Get the `<container id>`:
 

` docker ps `


` docker exec -u root -t -i <container id> /bin/bash `


- Then run in cmd line:


` tensorboard --logdir=//tmp/tensorflow/mnist/logs `


- Type in: ` cd / ` to get root.

Then cd into the folder that hot reloads code from your local folder/fav IDE at: `/apps/apps/gpu_benchmarks` and run:


` python tensorboard.py `


- Go to the browser and navigate to: ` localhost:6006 `



### Step 5: Run tests to prove container based GPU perf ###

- Demonstrate GPU vs CPU performance:

- Exec into the container if you haven't, and cd over to /tf/notebooks/apps/gpu_benchmarks and run:

- CPU Perf:

` python benchmark.py cpu 10000 `

- CPU perf should return something like this:

`Shape: (10000, 10000) Device: /cpu:0
Time taken: 0:00:03.934996`

- GPU perf:

` python benchmark.py gpu 10000 `

- GPU perf should return something like this:

`Shape: (10000, 10000) Device: /gpu:0
Time taken: 0:00:01.032577`


--------------------------------------------------


### Known conflicts with nvidia-docker and Ubuntu ###

AppArmor on Ubuntu has sec issues, so remove docker from it on your local box, (it does not hurt security on your computer):

` sudo aa-remove-unknown `

--------------------------------------------------

If building impactful data science tools for pharma is important to you or your business, please get in touch.

#### Contact
Email: joehoeller@gmail.com




