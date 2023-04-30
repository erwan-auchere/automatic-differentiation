# Automatic differentiation of optimization solutions

A common criticism made against deep learning models is their lack of interpretability. For this reason, the past few years have seen emerge different attempts to produce interpretable layers to be used in a deep learning framework.<br>
In this project, we experiment with two of these methods: the neural sort [1] (a continuous relaxation of the sorting operator) and the optnet [2] (a neural layer that solves a (potentially constrained) quadratic program).

---
## Neural Sort
This project proposes a Pytorch implementation of the [NeuralSort](https://arxiv.org/abs/1903.08850) presentend by Aditya Grover et al. in 2019.

## Project tree
The code is in the folder Neuralsort of this repositorie and it is organized as follows :

- dataset.py: Make the data processing
- dknn_layer.py: define "Differentiable k-nearest neighbors" layer.
- neuralsort.py: define the sort operateor
- run_knn.py: run the standard knn
- run_baseline.py: run the esnet meodel
- run_dknn.py: run the neuralsort model for the Differentiable knn
- Our_expérience.ipynb: show the results.
-requirements.txt

## Prerequisite

To replicate the results you will need the following modules :
- pytorch
- numpy
- scikit-learn
-tensorflow
- scipy
-matplotlib
-torch
-torchvision

To install them we recommend you to use the file requirements.txt which is in this folder by doing from a Terminal or a Command Prompt:

- clone or download this repositorie names Bayesian-GANS

- move to this folder : cd Neuralsort

```
pip install -r requirements.txt
```

## Run
To replicate the results run the script as follows from a Terminal or a Command Prompt:

- For execute Differentiable knn : python run_dknn.py -- nearest neighbors number --temperature(tau) --learning rate --method=deterministic --dataset name 

- For execute classical knn : python run_knn.py --dataset name
- For execute resnet model : python run_baseline.py --dataset name --learning rate.

## Nota:Options for Differentiable knn


```
  --k INT                 number of nearest neighbors
  --tau FLOAT             temperature of sorting operator
  --nloglr FLOAT          negative log10 of learning rate
  --dataset STRING        one of 'mnist', 'fashion-mnist', 'cifar10', 'cifar100','emnist_minst', 'emnist_digit'
  --num_train_queries INT number of queries to evaluate during training.
  --num_train_neighbors INT number of neighbors to consider during training.
  --num_samples INT       number of samples for stochastic methods
  --num_epochs INT        number of epochs to train
  -resume                 start a new model, instead of loading an older one
```

## Examples

_Training dKNN model to classify CIFAR100 digits_

```
cd Neuralsort
python run_dknn.py --k=9 --tau=85 --nloglr=3 --method=deterministic --dataset=cifar100
```

_Training standard knn to classify mnist_

```
cd cd Neuralsort

python run_knn.py  --dataset=mnist
```

_Training resnet model to calssify mnist digit_

```
cd Neuralsort
python run_baseline.py --dataset=mnist --nloglr=3
```

## Opt Net

The folder `optnet` contains:
- a python file with several functions and classes to define the models used;
- a python notebook with an application of the OptNet to solve the time series denoising problem.

---
## Bibliography
[1] Aditya Grover, Eric Wang, Aaron Zweig, and Stefano Ermon. Stochastic optimization of sorting networks via continuous relaxations. In International Conference on Learning Representations. 2019.<br>
[2] Brandon Amos and J. Zico Kolter. OptNet: Differentiable Optimization as a Layer in Neural Networks. 2021.

