import numpy as np
import json

# Load the model files
with open("harpertoken-cartpole/metadata.json") as f:
    metadata = json.load(f)

weights = np.load("harpertoken-cartpole/weights.npy")

# Print information
print("Metadata:", metadata)
print("Weights shape:", weights.shape)
print("Weights dtype:", weights.dtype)
print("Weights min:", weights.min())
print("Weights max:", weights.max())
print("Weights contents:")
print(weights)
