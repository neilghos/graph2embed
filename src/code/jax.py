raise ModuleNotFoundError(
    "jax is intentionally disabled for this experiment repo because TensorFlow "
    "only uses it as an optional TFLite dependency during DeepChem import."
)
