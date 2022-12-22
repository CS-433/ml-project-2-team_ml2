#!/usr/bin/python


# Import libraries
from get_prediction import *

useCUNet = False     # set to True to run the CUNet model
if useCUNet:
    from src.training_CUNet import *
else:
    from src.training import *


# Check for gpu availability:
if torch.cuda.is_available():
    print("CUDA IS AVAILABLE!")
else:
    print("WARNING: CUDA NOT AVAILABLE!")


# ===== TRAINING NEURAL NETWORK =====
torch.cuda.empty_cache()
image_size = 400
model = 'unet'  # Choose between 'unet' and 'resunet'
num_epochs = 100
frac_data = 1.0
device = 'cuda' if torch.cuda.is_available() else 'cpu'

optimizer_kwargs = dict(
    lr=1e-3,
    weight_decay=1e-2,
)

train_acc, val_acc, model = run_training(
    model_factory=UNet,
    num_epochs=num_epochs,
    optimizer_kwargs=optimizer_kwargs,
    device=device,
    frac_data=frac_data,
    model=model
)


# ===== MAKE-SAVE PREDICTION =====
get_prediction(model)


# ===== SAVE MODEL =====
filename_model = f"Predictions/model.pth"
torch.save(model, filename_model)

