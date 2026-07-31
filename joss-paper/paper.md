---
title: 'StableEMRIFisher: Stable and rapid Fisher information matrices for extreme-mass-ratio inspirals'
tags:
  - Python
  - gravitational waves
  - Laser Interferometer Space Antenna
  - extreme-mass-ratio inspirals
  - Fisher information matrix
  - parameter estimation
  - data analysis
authors:
  - name: Shubham Kejriwal
    orcid: 0009-0004-5838-1886
    affiliation: "1"
    corresponding: true
  - name: Ollie Burke
    orcid: 0000-0003-2393-209X
    affiliation: "2"
  - name: Christian E. A. Chapman-Bird
    orcid: 0000-0002-2728-9612
    affiliation: "3"
  - name: Alvin J. K. Chua
    orcid: 0000-0001-5242-8269
    affiliation: "1, 4"
affiliations:
  - index: 1
    name: Department of Physics, National University of Singapore, 2 Science Drive 3, Singapore 117551
  - index: 2
    name: Institute for Gravitational Wave Astronomy & School of Physics and Astronomy, University of Birmingham, Edgbaston, Birmingham B15 2TT, UK
  - index: 3
    name: School of Physics and Astronomy, University of Glasgow, Glasgow G12 8QQ, UK
  - index: 4
    name: Department of Mathematics, National University of Singapore, 2 Science Drive 2, Singapore 117543
date: 31 July 2026
bibliography: paper.bib
---

# Summary

<!-- This is where we talk about the background and motivation. What are EMRIs? What are Fisher information matrices and why are they useful? Cite any other EMRI Fisher matrix tools here. What does StableEMRIFisher do? An executive summary in words of the tool, and structure of the package. Where is it hosted? Reference to readthedocs. -->

# Statement of need

<!-- Why Fisher information matrices are difficult to compute reliably: highly oscillatory waveforms making finite differencing difficult: different tuning parameters in the finite difference calculations, strong correlations across the (intrinsic) parameter space, impact of the response, etc. How StableEMRIFisher circumvents these challenges by initializing a grid of finite difference deltas, finite differencing with respect to amplitude and mode-phases that vary on the radiation-reaction timescale which is much slower than the orbital timescale allowing stable finite differencing, etc... Emphasize the cost of Fisher matrix computation using SEF v/s MCMC citing previous studies -->

# Research impact statement

<!-- TODO: cite the specific papers that used SEF -->

So far, `StableEMRIFisher` has been used in the following publications:

1. 

# AI usage disclosure

No generative AI tools were used in the development of the software, its
documentation, or the authoring of this paper.


# Acknowledgements

We thank the developers of the `FastEMRIWaveforms`, `fastlisaresponse`, and
`lisatools` packages. S.K. acknowledges support from the NUS Research Scholarship and the computational resources accessed from NUS IT Research and Computing Group.

# References