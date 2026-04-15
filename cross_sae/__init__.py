"""Cross-Sparse Autoencoder package for comparing latent representations.

Provides tools for training sparse autoencoders on frozen model latents
and comparing representations between ImmuVis and VirTues (or any pair
of multiplex imaging models).

Modules
-------
sparse_autoencoder
    SAE model definitions: vanilla L1-SAE, TopK-SAE, and Gated-SAE.
train_sae
    Extract latents from a frozen model and train an SAE dictionary.
train_cross_sae
    Train paired SAEs and compute cross-model activation analysis.
interpret
    Marker-level variance analysis, feature–marker attribution,
    and predictive feature identification.
visualize
    Spatial activation maps, feature dashboards, distance-vs-activation
    plots, and cross-model comparison panels.
"""
