# Installation

Please follow these instructions to setup the computation environment. I use docker for the pytorch environment to avoid dependency issues and portability. 

* Prepare a compute environment with GPU (optional but recommended). 

* Install `docker` by following the instructions in this [link](https://www.docker.com/community-edition). 

* [Optinal] [Only for Linux] Install `nvidia-docker` by following the instructions in this [link](https://github.com/NVIDIA/nvidia-docker/wiki).

* Setup the pytorch docker image by following these instructions:
  ```
  $ nvidia-docker run -ti --rm --ipc=host -p 8888:8888 pytorch/pytorch:latest
  ```

* From within the docker image:
  ```
  # pip install jupyter jupyter-lab --upgrade
  # git clone https://github.com/buntyke/pytorch-hed
  ```

* Run jupyter notebook with the following command:
  ```
  # jupyter lab --no-browser --allow-root --ip=0.0.0.0
  ```
  Access lab environment from a browser.
