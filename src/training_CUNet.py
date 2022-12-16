#import matplotlib.pyplot as plt

import os
import glob
from torch.utils.data import DataLoader

from src.Modeles.UNet import *
from src.Modeles.MDUNet import *
from src.Save_Load.load_data import *
from src.Save_Load.save_data import *
from src.Submission.mask_to_submission import *

# Load data
def get_dataloaders(split, frac_data=1.0, shuffle=True):
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
def train_epoch(model, optimizer, scheduler, criterion, train_loader, epoch, device, threshold, SaveResults=False):
    model.train()
    loss_history = []
    accuracy_history = []
    lr_history = []
    outputs = torch.zeros((1, 1, 400, 400))
    #targets = torch.zeros((1, 1, 400, 400))
    for batch_idx, (data, target) in enumerate(train_loader):
        target = target.type(torch.LongTensor)  # avoid an error idk why?
        data, target = data.float().to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        if SaveResults:
            outputs = torch.cat((outputs, output.cpu()), dim=0)
            #targets = torch.cat((targets, target.cpu()), dim=0)
        output = output.flatten().float()  # [batch*400*400]
        target = target.flatten().float()  # [batch*400*400]
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        scheduler.step()
        predictions = output.cpu().detach().numpy()
        ground_truth = target.cpu().detach().numpy()
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
    if SaveResults:
        outputs[outputs > threshold] = 0    # apply threshold to keep unsure results
        outputs = outputs[1::, :, :, :]  # remove first empty value
        save_data(outputs, "Results/temp/", target=False)   # save data to train MDUNet
        #targets = targets[1:, :, :, :]  # remove first empty value
        #save_data(targets, "Results/temp/", target=True)   # save data to train MDUNet
    return loss_history, accuracy_history, lr_history


# Evaluation of the dataset loaded on "val_loader" with the model in "eval" mode:
def validate(model, device, val_loader, criterion, SaveResults=False):
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
            #threshold = 0.5
            output[output < 0.5] = 0
            output[output > 0.5] = 1

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


def validate_CUNet(modelUNet, modelMDUnet, device, val_loader, criterion, frac_data, SaveResults=True):
    with torch.no_grad():
        modelUNet.eval()  # Important: eval mode (affects dropout, batch norm etc)
        modelMDUnet.eval()  # Important: eval mode (affects dropout, batch norm etc)
        test_loss = 0
        accuracy_float = 0
        targets = torch.zeros((1, 1, 400, 400))
        outputsUNet = torch.zeros((1, 1, 400, 400))
        outputsMDUNet = torch.zeros((1, 1, 400, 400))

        # remove files from temp folder
        files = glob.glob('Results/temp/*')
        for f in files:
            os.remove(f)

        for data, target in val_loader:  # run the
            target = target.type(torch.LongTensor)  # avoid an error idk why?
            data = data.float()
            data, target = data.to(device), target.to(device)
            outputUNet = modelUNet(data)  # tensor[batch,1,400,400]
            # apply threshold to binaries the output
            #threshold = 0.5
            outputsUNet = torch.cat((outputsUNet, outputUNet.cpu()), dim=0)
            targets = torch.cat((targets, target.cpu()), dim=0)

        outputsUNet = outputsUNet[1::, :, :, :]
        outputsUNet[outputsUNet > 0.9] = 0
        targets = targets[1::, :, :, :]
        print(f"outputsUNet shape : {outputsUNet.shape}")
        save_data(outputsUNet, "Results/temp/", target=False)   # save data to train MDUNet
        train_inter_dataset, train_inter_loader = get_dataloaders("inter_val", frac_data=frac_data)

        for inter, target in train_inter_loader:  # run the
            target = target.type(torch.LongTensor)  # avoid an error idk why?
            inter = inter.float()
            inter, target = inter.to(device), target.to(device)
            outputMDUNet = modelMDUnet(inter)  # tensor[batch,1,400,400]
            outputsMDUNet = torch.cat((outputsMDUNet, outputMDUNet.cpu()), dim=0)

        outputsMDUNet = outputsMDUNet[1:, :, :, :]
        # apply threshold to binaries the output
        outputsUNet[outputsUNet < 0.5] = 0
        outputsUNet[outputsUNet > 0.5] = 1
        outputsMDUNet[outputsMDUNet < 0.5] = 0
        outputsMDUNet[outputsMDUNet > 0.5] = 1

        # combination of outputs
        print(f"outputsUNet shape : {outputsUNet.shape}")
        print(f"outputsMDUNet shape : {outputsMDUNet.shape}")
        save_data(outputsUNet, "Results/Prediction_imgs/pred_UNet/", target=False)   # save data to train MDUNet
        save_data(outputsMDUNet, "Results/Prediction_imgs/pred_MDUNet/", target=False)   # save data to train MDUNet
        outputs = outputsUNet
        outputs[outputsMDUNet == 1] = 1

        outputs_flat = outputs.flatten().float()  # [batch*200*200]
        targets_flat = targets.flatten().float()  # [batch*200*200]
        '''print(output.shape)
        print(target.shape)'''
        test_loss += criterion(outputs_flat, targets_flat)

        predictions = outputs_flat.cpu().detach().numpy()
        ground_truth = targets_flat.cpu().detach().numpy()

        accuracy_float = np.mean((predictions == ground_truth))

        test_loss /= len(val_loader)

        # Loading of output into folder "Results/Prediction_imgs" :
        if SaveResults:
            save_data(outputs, "Results/Prediction_imgs/", target=False)
            save_data(targets, "Results/Prediction_imgs/", target=True)

        print(
            "===== OVER ALL: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)".format(
                test_loss,
                accuracy_float,
                len(val_loader),
                100.0 * accuracy_float / len(val_loader),
            )
        )
        return test_loss, accuracy_float / len(val_loader)


def run_training(model_factory, num_epochs, optimizer_kwargs, device="cuda", frac_data=1.0):
    # ===== Data Loading =====
    train_dataset, train_loader = get_dataloaders("train", frac_data=frac_data)
    val_dataset, val_loader = get_dataloaders("val", frac_data=frac_data)
    print(f"Training data_set size : {len(train_dataset)}")
    print(f"Validation data_set size : {len(val_dataset)}")
    print("DATASETS LOADED! ")

    # ===== Model, Optimizer and Criterion =====
    modelUNet = UNet(3, 1)
    modelUNet = modelUNet.to(device)
    optimizer = torch.optim.AdamW(modelUNet.parameters(), **optimizer_kwargs)
    criterion = torch.nn.functional.cross_entropy
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=(len(train_dataset) * num_epochs) // train_loader.batch_size,
    )

    # ===== Train Model =====
    threshold = 0.9
    # Train UNet
    lr_UNet_history = []
    train_loss_UNet_history = []
    train_acc_UNet_history = []
    val_loss_UNet_history = []
    val_acc_UNet_history = []

    for epoch in range(1, num_epochs + 1):
        # Training one epoch
        SaveResults = False if epoch < num_epochs else True
        train_loss, train_acc, lrs = train_epoch(
            modelUNet, optimizer, scheduler, criterion, train_loader, epoch, device, threshold, SaveResults
        )
        train_loss_UNet_history.extend(train_loss)
        train_acc_UNet_history.extend(train_acc)
        lr_UNet_history.extend(lrs)

        # Run the validation data through the model
        val_loss, val_acc = validate(modelUNet, device, val_loader, criterion)
        val_loss_UNet_history.append(val_loss)
        val_acc_UNet_history.append(val_acc)


    # Train MDUNet
    train_inter_dataset, train_inter_loader = get_dataloaders("inter_train", frac_data=frac_data)
    #save_data(train_inter_dataset[0], 'Results/Prediction_imgs/test/', target=False)                !!!!!!!!!!!!!!!!!!!!!!
    print(len(train_inter_dataset))
    modelMDUNet = MDUNet(1, 1)
    modelMDUNet = modelMDUNet.to(device)
    optimizerMDUNet = torch.optim.AdamW(modelMDUNet.parameters(), **optimizer_kwargs)
    criterionMDUNet = torch.nn.functional.cross_entropy
    schedulerMDUNet = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizerMDUNet,
        T_max=(len(train_inter_dataset) * num_epochs) // train_inter_loader.batch_size,
    )
    lr_MDUNet_history = []
    train_loss_MDUNet_history = []
    train_acc_MDUNet_history = []
    val_loss_MDUNet_history = []
    val_acc_MDUNet_history = []
    for epoch in range(1, num_epochs + 1):
        # Training one epoch
        SaveResults = False
        train_loss, train_acc, lrs = train_epoch(
            modelMDUNet, optimizerMDUNet, schedulerMDUNet, criterionMDUNet, train_inter_loader, epoch, device, SaveResults
        )
        train_loss_MDUNet_history.extend(train_loss)
        train_acc_MDUNet_history.extend(train_acc)
        lr_MDUNet_history.extend(lrs)

        # Run the validation data through the model
        #val_loss, val_acc = validate(modelMDUNet, device, val_loader, criterion)
        val_loss_MDUNet_history.append(val_loss)
        val_acc_MDUNet_history.append(val_acc)

    val_loss_ult, val_acc_ult = validate_CUNet(modelUNet, modelMDUNet, device, val_loader, criterion, frac_data)
    return train_acc_UNet_history, val_acc_UNet_history, modelUNet


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
            #threshold = 0.5
            output[output < 0.5] = 0
            output[output > 0.5] = 1

            outputs = torch.cat((outputs, output.cpu()), dim=0)

    outputs = outputs[1:]
    # Convert from 400x400 four corners labels to 606x608 whole lab
    labels = fuse_four_corners_labels(outputs)
    n_labels = labels.size(dim=0)

    # Create images for submission
    """print("Saving predictions")
    image_files = []
    for ind in range(0, n_labels):
        image_file = 'Predictions/satImage_' + '%.3d' % (ind + 1) + '.png'
        plt.imsave(image_file, labels[ind].squeeze(), cmap="gray")
        image_files.append(image_file)

    submission_file = 'final_submission.csv'
    masks_to_submission(submission_file, *image_files)"""
