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

@Lipson used Genetic Programming to automate the inference of symbolic relations from data.
A number of open source libraries have been developed in this context
[@deap; cranmer2020discovering; cranmer2020pysr].

@SINDy101 proposed a framework for data-driven dynamical systems, more recently
translated in Python [@desilva2020pysindy]. The limitation of SINDy, which goes together 
with its efficiency and interpretability, is the linearity assumption between the features 
and the state derivative: while different optimization frameworks, like SR3, 
have been introduced [@Champion], these are sensitive to the initial 
guess on nonlinear parameters, which are also assumed to be constant.

A less limiting framework, which trades generality with lack of interpretability,
consists in the use of Neural Networks to reconstruct differential equations [@sciml; @kidger2021on]: in order to
keep interpretability but increase the generality of SINDy, @ManziIAC:2020 introduced the combination 
of Genetic Programming-based Symbolic Regression and Sparse Regression for the reconstruction of ordinary differential equations,
which lead to the development of SymINDy.

# Mathematics

$$     
\dot{\mathbf{Y}} \approx \Theta(\mathbf{Y}) \boldsymbol\Xi
$$

$$
{\Theta(\mathbf{Y})} = 
\begin{bmatrix}
\Theta_1(\mathbf{Y}) & \Theta_2(\mathbf{Y}) & \Theta_3(\mathbf{Y})
\end{bmatrix}
$$

<img src="figures/tree0.png" alt="drawing" width="200"/>
<img src="figures/tree1.png" alt="drawing" width="200"/>
<img src="figures/tree2.png" alt="drawing" width="200"/>

# Figures

Figures can be included like this:
![Caption for example figure.\label{fig:example}](figure.png)
and referenced from text using \autoref{fig:example}.

Figure sizes can be customized by adding an optional second parameter:
![Caption for example figure.](figure.png){ width=20% }

# Acknowledgements

We acknowledge Miles Cranmer, Nathan Kutz, Christopher Rackauckas and Joel Rosenfeld for useful discussions on the topic of
data-driven dynamical systems.

# References