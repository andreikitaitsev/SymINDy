---
title: 'SynINDy: Symbolic Identification of Nonlinear Dynamics'
tags:
  - Python
  - Symbolic Regression
  - Genetic Programming
  - Sparse Regression
  - Data-driven Dynamical Systems
authors:
  - name: Andrei Kitaitsev
    affiliation: 1 # (Multiple affiliations must be quoted)
  - name: Matteo Manzi
    orcid: 0000-0002-5229-0746
    affiliation: 2

affiliations:
 - name: CURE finance, Berlin, Germany
   index: 1
 - name: DATACRUNCH, Paris, France
   index: 2
date: 04 August 2022
bibliography: paper.bib
---

# Summary

The context of data-driven dynamical systems is characterized by the goal of 
automating the discovery of governing equations from data, for example with Machine Learning 
techniques, by means of constructing interpretable, 
white-box models with both generalizing and extrapolating capabilities.

Leveraging the powerful modelling capacity of symbolic regression and the 
sparse representability of dynamical systems, the python library **SymINDy** (Symbolic Identification of Nonlinear Dynamics) 
allows one to generate a symbolic representation of orbital anomalies from state observations only.  

# Statement of need

@Lipson used Genetic Programming to automate the inference of symbolic relations from data. Building on the work, a number of open source libraries have been developed
[@deap; cranmer2020discovering; cranmer2020pysr].

@SINDy101 later proposed a different framework for data-driven dynamical systems, more recently
translated in Python [@desilva2020pysindy]. The limitation of SINDy, which goes together 
with its efficiency and interpretability, is the linearity assumption between the features 
and the state derivative: while different optimization frameworks, like SR3, 
have been introduced [@Champion], these are sensitive to the initial 
guess on nonlinear parameters, which are also assumed to be constant.

A less limiting framework, which trades generality with lack of interpretability,
consists in the use of Neural Networks to reconstruct differential equations [@sciml; @kidger2021on]: in order to
keep interpretability but increase the generality of SINDy, @ManziIAC:2020 introduced the combination 
of Genetic Programming-based Symbolic Regression and Sparse Regression for the reconstruction of ordinary differential equations, which has lead to the development of SymINDy.

# Mathematics

The starting assumption is access to a number of measurements, in general noisy, that can be arranged into a snapshot matrix:

$$
    \mathbf{Y} = [\mathbf{y}_1(t_1), \ \ \mathbf{y}_2(t_2), \ \ \ldots, \ \ \mathbf{y}_n(t_n)]
$$

which can be used to numerically estimate a state-derivative matrix

$$
    \dot{\mathbf{Y}} = [\dot{\mathbf{y}}_1(t_1), \ \ \dot{\mathbf{y}}_2(t_2), \ \ \ldots, \ \ \dot{\mathbf{y}}_n(t_n)]
$$

The two quantities are assumed to be related by

$$     
\dot{\mathbf{Y}} \approx \Theta(\mathbf{Y}) \boldsymbol\Xi
$$

in which the state-dependent matrix is associated with a genetic programming individual, defined as a set of symbolic expressions, which in general are nonlinear and are characterized by nonlinear parameters

$$
{\Theta(\mathbf{Y})} = 
\begin{bmatrix}
\Theta_1(\mathbf{Y}) & \Theta_2(\mathbf{Y}) & \Theta_3(\mathbf{Y})
\end{bmatrix}
$$

 $\boldsymbol\Xi$ is a sparse matrix of constant coefficients, which is computed using the Sequential Thresholded Least Squares (STLSQ) algorithm.
 
# Acknowledgements

We acknowledge Miles Cranmer, Nathan Kutz, Christopher Rackauckas and Joel Rosenfeld for useful discussions on the topic of
data-driven dynamical systems.

# References