# hed_pytorch

Pytorch implementation of [Holistically-nested Edge Detection (HED)][1]. This repo can be used to train a HED model and perform test inference. The implementation was evaluated on the BSDS dataset.

* [Prerequisites](#prerequisites)
* [Usage](#usage)
* [Performance](#performance)
* [Files](#files)
* [Acknoledgement](#acknowledgement)
* [References](#references)

## Prerequisites

* Compute environment with GPU (optional).
* Python environment: [Anaconda](https://conda.io/docs/user-guide/install/index.html) is recommended.
* SciPy stack: [Install instructions](https://www.scipy.org/install.html).
* [Optional] CUDA and NVidia Drivers: [Install instructions](https://developer.nvidia.com/cuda-downloads).
* Pytorch: [Install instructions](http://pytorch.org/).

I use docker to avoid dependency problems. Installation instructions for my setup are available [here](Install.md).

## Usage

* Download repository:
  ```
  $ git clone https://github.com/buntyke/pytorch-hed.git
  ```
* Create `data` folder, download and extract BSDS dataset into folder:
  ```
  $ cd pytorch-hed/
  $ mkdir data; cd data
  $ wget http://vcl.ucsd.edu/hed/HED-BSDS.tar
  $ tar -xvf HED-BSDS.tar
  $ rm HED-BSDS.tar
  $ cd HED-BSDS/
  $ head -n 10 train_pair.lst > val_pair.lst
  $ cd ../../
  ```
* Download the VGG pretrained model to initialize training
  ```
  $ mkdir model; cd model/
  $ wget https://download.pytorch.org/models/vgg16-397923af.pth
  $ mv vgg16-397923af.pth vgg16.pth
  $ cd ..
  ```
* Train HED model by running `train.py` or `train.ipynb` notebook following the instructions:
  ```
  $ python train.py 
  ```
  The trained model along with validation results are stored in the train folder.

## Performance

* 

## Files

* [train.ipynb](train.ipynb): Notebook to train HED model.
* [trainer.py](trainer.py): Helper class to train model and perform validation.
* [model.py](model.py): HED model definition given through several class implementations.
* [dataproc.py](dataproc.py): Dataset class implementation used in Trainer class.

## Acknowledgement

The source code is derived from three different implementations available online. Thanks to [@s9xie][2] for original caffe implementation. Thanks to [@EliasVansteenkiste][3], [@xlliu][4], [@BinWang-shu][5] for the pytorch implementations.

[1]: https://arxiv.org/abs/1504.06375 "HED"

[2]: https://github.com/s9xie/hed "Caffe"

[3]: https://github.com/EliasVansteenkiste/edge_detection_framework "Pytorch 1"

[4]: https://github.com/xlliu7/hed.pytorch "Pytorch 2"

[5]: https://github.com/BinWang-shu/pytorch_hed "Pytorch 3"