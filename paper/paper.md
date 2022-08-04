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

The forces on stars, galaxies, and dark matter under external gravitational
fields lead to the dynamical evolution of structures in the universe. The orbits
of these bodies are therefore key to understanding the formation, history, and
future state of galaxies. The field of "galactic dynamics," which aims to model
the gravitating components of galaxies to study their structure and evolution,
is now well-established, commonly taught, and frequently used in astronomy.
Aside from toy problems and demonstrations, the majority of problems require
efficient numerical tools, many of which require the same base code (e.g., for
performing numerical orbit integration).

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

Single dollars ($) are required for inline mathematics e.g. $f(x) = e^{\pi/x}$

Double dollars make self-standing equations:

$$\Theta(x) = \left\{\begin{array}{l}
0\textrm{ if } x < 0\cr
1\textrm{ else}
\end{array}\right.$$

You can also use plain \LaTeX for equations
\begin{equation}\label{eq:fourier}
\hat f(\omega) = \int_{-\infty}^{\infty} f(x) e^{i\omega x} dx
\end{equation}
and refer to \autoref{eq:fourier} from text.
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