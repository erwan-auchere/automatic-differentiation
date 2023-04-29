# Automatic differentiation of optimization solutions

A common criticism made against deep learning models is their lack of interpretability. For this reason, the past few years have seen emerge different attempts to produce interpretable layers to be used in a deep learning framework.<br>
In this project, we experiment with two of these methods: the neural sort [1] (a continuous relaxation of the sorting operator) and the optnet [2] (a neural layer that solves a (potentially constrained) quadratic program).

---
## Neural Sort


---
## Opt Net

The folder `optnet` contains:
- a python file with several functions and classes to define the models used;
- a python notebook with an application of the OptNet to solve the time series denoising problem.

---
## Bibliography
[1] Aditya Grover, Eric Wang, Aaron Zweig, and Stefano Ermon. Stochastic optimization of sorting networks via continuous relaxations. In International Conference on Learning Representations. 2019.<br>
[2] Brandon Amos and J. Zico Kolter. OptNet: Differentiable Optimization as a Layer in Neural Networks. 2021.

