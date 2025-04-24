from transformers.utils import move_cache
import torch

if torch.backends.mps.is_available():
    print("MPS is available!")

move_cache()
print("Cache migration done.")
