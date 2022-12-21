import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from src.Modeles.UNet import *
from src.Modeles.ResUNet import *
from src.Save_Load.load_data import *
from src.Save_Load.save_data import *
from src.Submission.mask_to_submission import *


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


# Initialize neural network

# Training of the model by iterating on the epochs
def train_epoch(model, optimizer, scheduler, criterion, train_loader, epoch, device):
    model.train()
    loss_history = []
    accuracy_history = []
    lr_history = []
    for batch_idx, (data, target) in enumerate(train_loader):
        target = target.type(torch.LongTensor)  # avoid an error idk why?
        data, target = data.float().to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        output = output.flatten().float()  # [batch*400*400]
        target = target.flatten().float()  # [batch*400*400]
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        scheduler.step()

        predictions = output.cpu().detach().numpy()
        ground_truth = target.cpu().detach().numpy()
        predictions = np.where(predictions > 0.5, 1, 0)
        accuracy_float = np.mean((predictions == ground_truth))
        loss_float = loss.item()

        loss_history.append(loss_float)
        accuracy_history.append(accuracy_float)
        lr_history.append(scheduler.get_last_lr()[0])
        if batch_idx % (len(train_loader.dataset) // len(data) // 10) == 0:
            print(
                f"Train Epoch: {epoch}-{batch_idx:03d} "
                f"batch_loss={loss_float:0.2e} "
                f"batch_acc={accuracy_float:0.3f} "
                f"lr={scheduler.get_last_lr()[0]:0.3e} "
            )
    return loss_history, accuracy_history, lr_history


# Evaluation of the dataset loaded on "val_loader" with the model in "eval" mode:
def validate(model, device, val_loader, criterion, SaveResults=True):
    with torch.no_grad():
        model.eval()  # Important: eval mode (affects dropout, batch norm etc)
        test_loss = 0
        accuracy_float = 0
        targets = torch.zeros((1, 1, 400, 400))
        outputs = torch.zeros((1, 1, 400, 400))
        for data, target in val_loader:  # run the
            target = target.type(torch.LongTensor)  # avoid an error idk why?
            data = data.float()
            data, target = data.to(device), target.to(device)
            output = model(data)  # tensor[batch,1,400,400]

            # apply threshold to binaries the output
            threshold = 0.5
            output[output < threshold] = 0
            output[output > threshold] = 1

            if SaveResults:
                targets = torch.cat((targets, target.cpu()), dim=0)
                outputs = torch.cat((outputs, output.cpu()), dim=0)
            output = output.flatten().float()  # [batch*200*200]
            target = target.flatten().float()  # [batch*200*200]
            '''print(output.shape)
            print(target.shape)'''
            test_loss += criterion(output, target)

            # predictions = output.argmax(1).cpu().detach().numpy()
            predictions = output.cpu().detach().numpy()
            ground_truth = target.cpu().detach().numpy()

            accuracy_float += np.mean((predictions == ground_truth))

        test_loss /= len(val_loader)

        # Loading of output into folder "Results/Prediction_imgs" :
        if SaveResults:
            save_data(outputs, "Results/Prediction_imgs/", target=False)
            save_data(targets, "Results/Prediction_imgs/", target=True)

        print(
            "Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)".format(
                test_loss,
                accuracy_float,
                len(val_loader),
                100.0 * accuracy_float / len(val_loader),
            )
        )
        model.train()  # so that the model can be trained again after evaluation
        return test_loss, accuracy_float / len(val_loader)


def run_training(model_factory, num_epochs, optimizer_kwargs, device="cuda", frac_data='1.0', model='unet'):
    # ===== Data Loading =====
    train_dataset, train_loader = get_dataloaders("train", frac_data)
    val_dataset, val_loader = get_dataloaders("val", frac_data)
    print(f"Training data_set size : {len(train_dataset)}")
    print(f"Validation data_set size : {len(val_dataset)}")
    print("DATASETS LOADED! ")

    # ===== Model, Optimizer and Criterion =====
    if model == 'unet':
        model = UNet(3, 1)
    elif model == 'resunet':
        model = ResUnet(channel=3)
    else:
        raise ValueError('INVALID MODEL CHOSEN !')
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), **optimizer_kwargs)
    criterion = torch.nn.functional.cross_entropy
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=(len(train_dataset) * num_epochs) // train_loader.batch_size,
    )

    # ===== Train Model =====
    lr_history = []
    train_loss_history = []
    train_acc_history = []
    val_loss_history = []
    val_acc_history = []
    for epoch in range(1, num_epochs + 1):
        # Training one epoch
        train_loss, train_acc, lrs = train_epoch(
            model, optimizer, scheduler, criterion, train_loader, epoch, device
        )
        train_loss_history.extend(train_loss)
        train_acc_history.extend(train_acc)
        lr_history.extend(lrs)

        # Run the validation data through the model
        val_loss, val_acc = validate(model, device, val_loader, criterion)
        val_loss_history.append(val_loss)
        val_acc_history.append(val_acc)

    return train_acc_history, val_acc_history, model


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

    submission_file = 'final_submission.csv'
    masks_to_submission(submission_file, *image_files)
