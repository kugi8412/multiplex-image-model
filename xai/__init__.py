#!/usr/bin/env python
# -*- coding: utf-8 -*-
# __init__.py

"""
XAI package for ImmuVis multiplex image models.

Provides explainability tools for understanding marker relationships and
model behaviour:

    counterfactual
        Gradient-based virtual perturbation — remove target markers from
        input, then optimise remaining markers to maximise / minimise
        a target prediction. Uncertainty-weighted to focus on confident
        regions.

    attribution
        Integrated Gradients and gradient x input saliency for
        per-marker, per-pixel attribution maps. Shows which input
        channels drive a specific reconstruction target.

    latent_likelihood
        PixelCNN-based explainability on frozen encoder latent space.
        Surprise maps, conditional likelihood attribution, marker
        dependency graphs, uncertainty decomposition, and counterfactual
        latent sampling.
"""

from .counterfactual import CounterfactualPerturbation
from .attribution import MarkerAttribution
from .latent_likelihood import LatentLikelihoodXAI

__all__ = [
    "CounterfactualPerturbation",
    "MarkerAttribution",
    "LatentLikelihoodXAI",
]
