import torch

if torch.cuda.is_available():
    print("SUCCESS !")
else:
    print("FAIL !")

print(f"CUDA version : {torch.version.cuda}")
