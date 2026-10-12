# Dog Breed Classification

**Personal Project | Python, PyTorch, NumPy, scikit-learn**

- Implemented and trained CNN and Vision Transformer classifiers in PyTorch with train-only normalization, early stopping, and validation-loss checkpointing; built an interactive dog breed prediction demo.
- Transferred features from an eight-class source task and evaluated four layer-freezing strategies, improving held-out test AUROC from 0.7304 to 0.8228 (+0.0924) by freezing all three convolutional layers.

Optional third bullet, when space allows:

- Built an automated experiment pipeline covering freezing, dropout, and weight decay, with training-only augmentation, per-run checkpoints, and validation-based model selection.

For a software engineering resume, the first bullet can emphasize inference:

- Built a PyTorch classification pipeline with CNN and Vision Transformer models, portable inference bundles containing preprocessing metadata, and an interactive image upload and prediction interface.

The default bullets retain the paired historical AUROC values. The separate
reproduction achieved 0.7312 to 0.8260 (+0.0948), with one random seed. Use one
comparison consistently; do not combine the old baseline with the new transfer
score or present a single-seed change as a statistically established improvement.
Use "extended a coursework project" when discussing the project's origin in interviews.
