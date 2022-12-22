from get_prediction import *
from src.training import *
filename_model = "Predictions/model.pth"

model = torch.load(filename_model)
model.eval()

get_prediction(model)
