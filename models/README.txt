# Place your GGUF model here.
# Recommended: napi-2b-q4_k_m.gguf (quantized ~1.5 GB)
# Download from: https://huggingface.co/ or your preferred model source.

# Napi can also use its own local neural brain:
#   napi_neural_brain.npz
#
# This file is trained by:
#   python train_neural_brain.py
#
# It contains NumPy skip-gram neural embeddings learned from knowledge/,
# memory notes, and recent messages. It is smaller than a GGUF LLM, but it is
# a real local neural weight file used by model="napi-neural-brain".
