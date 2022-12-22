import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from src.Save_Load.load_data import *
from src.Submission.mask_to_submission import *
from src.Modeles.ResUNet import *
from src.Modeles.UNet import *


# Load data
def get_dataloaders(split, frac_data, shuffle=True):
    data_dataset = Roads(split=split, frac_data=frac_data)
    # data loader
    data_loader = DataLoader(data_dataset,
                             batch_size=3,
                             shuffle=shuffle,
                             num_workers=0,
                             pin_memory=False)
    return data_dataset, data_loader

def get_prediction(model):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    test_dataset, test_loader = get_dataloaders("test", frac_data=1, shuffle=False)

    # Run forward pass
    print("Evaluating predictions...")
    with torch.no_grad():
        model.eval()  # Important: eval mode (affects dropout, batch norm etc)
        outputs = torch.zeros((1, 1, 400, 400))
        for data, _ in test_loader:  # run the
            data = data.float()
            data = data.to(device)
            output = model(data)  # tensor[batch,1,400,400]

            # apply threshold to binaries the output
            threshold = 0.5
            output[output < threshold] = 0
            output[output > threshold] = 1

            outputs = torch.cat((outputs, output.cpu()), dim=0)

    outputs = outputs[1:]
    # Convert from 400x400 four corners labels to 606x608 whole lab
    labels = fuse_four_corners_labels(outputs)
    n_labels = labels.size(dim=0)

    # Create images for submission
    print("Saving predictions")
    image_files = []
    for ind in range(0, n_labels):
        image_file = 'Predictions/images/satImage_' + '%.3d' % (ind + 1) + '.png'
        plt.imsave(image_file, labels[ind].squeeze(), cmap="gray")
        image_files.append(image_file)

    submission_file = 'Submissions/final_submission.csv'
    masks_to_submission(submission_file, *image_files)


if __name__ == '__main__':
    #model = UNet(3, 1)
    #model = ResUnet(channel=3)
    model = torch.load('Predictions/model.pth')
    #model.load_state_dict(torch.load('../Predictions/model.pth'))
    model.eval()
    print("Getting predictions...")
    get_prediction(model)
    print("Done!")
