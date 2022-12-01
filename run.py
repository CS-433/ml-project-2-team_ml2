#!/usr/bin/python


# Import libraries
from src.Modeles.UNet import *
from src.training import *
from src.Modeles.ResUNet import *


print("Data Loaded!")

# Check for gpu availability:
if torch.cuda.is_available():
    print("CUDA IS AVAILABLE!")
else:
    print("WARNING: CUDA NOT AVAILABLE!")

# Train neural network
torch.cuda.empty_cache()
image_size = 400
model_factory = resunet
num_epochs = 5
frac_data = 1.0
device = 'cuda' if torch.cuda.is_available() else 'cpu'

optimizer_kwargs = dict(
    lr=1e-3,
    weight_decay=1e-2,
)

train_acc, val_acc, model = run_training(
    model_factory=resunet,
    num_epochs=num_epochs,
    optimizer_kwargs=optimizer_kwargs,
    device=device,
    frac_data=frac_data
)
# Make-save prediction
