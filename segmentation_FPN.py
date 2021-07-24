import torch
import torchvision
from torch import nn
import os
import math
from camus_dataset import read_data
import time
import tqdm
import matplotlib.pylab as plt
import numpy as np
import segmentation_models_pytorch as smp

def run(num_epochs=100,
        modelname="FPN",
        pretrained=False,
        batch_size=4,
        validation_split = 0.05,
        run_train=False,
        run_test=True):


    # Set default output directory
    output = os.path.join("output", "segmentation", "{}".format(modelname))
    os.makedirs(output, exist_ok=True)

    # Set device for computations
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Set up model
    #model = torchvision.models.detection.__dict__[modelname](pretrained=pretrained)
    model=smp.FPN('resnet18', classes=4, encoder_weights='imagenet',activation='softmax') 
    model.encoder.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False) # change channel from 3 to 1
    # print(model)
    # n_class = 4
    # model.classifier[-1] = torch.nn.Conv2d(model.classifier[-1].in_channels, n_class, kernel_size=model.classifier[-1].kernel_size)  # change number of outputs
    model.to(device)

    # Set up optimizer
    #optim = torch.optim.SGD(model.parameters(), lr=1e-3, momentum=0.9)
    optim = torch.optim.Adam(model.parameters(), lr=1e-5)
    lr_step_period = math.inf
    scheduler = torch.optim.lr_scheduler.StepLR(optim, lr_step_period)

    # load data
    train_dataloader, val_dataloader, test_dataloader = read_data(batch_size=batch_size, validation_split=validation_split)
    dataloaders = {'train': train_dataloader, 'val': val_dataloader}

   # Run training and testing loops
    with open(os.path.join(output, "log.csv"), "a") as f:
        epoch_resume = 0 if run_train is True else num_epochs
        bestLoss = float("inf")
        try:
            # Attempt to load checkpoint
            checkpoint = torch.load(os.path.join(output, "checkpoint.pt"))
            model.load_state_dict(checkpoint['state_dict'])
            optim.load_state_dict(checkpoint['opt_dict'])
            scheduler.load_state_dict(checkpoint['scheduler_dict'])
            epoch_resume = checkpoint["epoch"] + 1
            bestLoss = checkpoint["best_loss"]
            f.write("Resuming from epoch {}\n".format(epoch_resume))
        except FileNotFoundError:
            f.write("Starting run from scratch\n")

        for epoch in range(epoch_resume, num_epochs):
            print("Epoch #{}".format(epoch), flush=True)
            for phase in ['train', 'val']:
                start_time = time.time()
                for i in range(torch.cuda.device_count()):
                    torch.cuda.reset_max_memory_allocated(i)
                    torch.cuda.reset_max_memory_cached(i)

                loss = run_epoch(model, dataloaders[phase], phase == "train", optim, device)
                f.write("{},{},{},{},{},{},{}\n".format(epoch,
                                                                    phase,
                                                                    loss,
                                                                    time.time() - start_time,
                                                                    sum(torch.cuda.max_memory_allocated() for i in range(torch.cuda.device_count())),
                                                                    sum(torch.cuda.max_memory_cached() for i in range(torch.cuda.device_count())),
                                                                    batch_size))
                f.flush()
            scheduler.step()

            # Save checkpoint
            save = {
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'best_loss': bestLoss,
                'loss': loss,
                'opt_dict': optim.state_dict(),
                'scheduler_dict': scheduler.state_dict(),
            }
            torch.save(save, os.path.join(output, "checkpoint.pt"))
            if loss < bestLoss:
                torch.save(save, os.path.join(output, "best.pt"))
                bestLoss = loss

        # Load best weights
        checkpoint = torch.load(os.path.join(output, "best.pt"))
        model.load_state_dict(checkpoint['state_dict'])
        f.write("Best validation loss {} from epoch {}\n".format(checkpoint["loss"], checkpoint["epoch"]))

        if run_test:
            # Run on test
            # loss = run_epoch(model, test_dataloader, False, None, device)
            loss = run_epoch(model, val_dataloader, False, None, device)


def run_epoch(model, dataloader, train, optim, device):
    """Run one epoch of training/evaluation for segmentation.

    Args:
        model (torch.nn.Module): Model to train/evaulate.
        dataloder (torch.utils.data.DataLoader): Dataloader for dataset.
        train (bool): Whether or not to train model.
        optim (torch.optim.Optimizer): Optimizer
        device (torch.device): Device to run on
    """

    total = 0.
    n = 0+1e-16
    model.train(train)

    with torch.set_grad_enabled(train):
        with tqdm.tqdm(total=len(dataloader)) as pbar:
            for idx, data in enumerate(dataloader):
                image = data['2CH_ED']
                label = data['2CH_ED_gt']

                # Run prediction for images and compute loss
                image = image.to(device)
                label = label.long()
                label = label.to(device)
                # print(np.shape(image)
                # image=int(np.array(image))
                # y_predict = model(image)["out"]
                y_predict = model(image)
                loss = torch.nn.functional.cross_entropy(y_predict, label[:,0,:,:], reduction="mean")

                # Take gradient step if training
                if train:
                    optim.zero_grad()
                    loss.backward()
                    optim.step()
                else:
                    softmax = torch.exp(y_predict).cpu()
                    prob = list(softmax.numpy())
                    predictions = np.argmax(prob, axis=1)
                    gt_image = label[:,0,:,:].cpu()
                    display_image(gt_image, predictions, idx)

                # Accumulate losses and compute baselines
                total += loss.item()
                n += label.size(0)

                # Show info on process bar
                pbar.set_postfix_str("{:.4f} , {:.4f}".format(total / n , loss.item()))
                pbar.update()

    return (total / n )

def display_image(source_images, predict_images, idx):
    plt.figure(figsize=(20, 16))
    plt.gray()
    plt.subplots_adjust(0, 0, 1, 1, 0.01, 0.01)
    batch_size = source_images.shape[0]
    for i in range(batch_size):
        plt.subplot(2, batch_size, i + 1), plt.imshow(source_images[i]), plt.axis('off')
        if i == 0:
            plt.text(0, -20, "Origin:", fontsize=60)
        plt.subplot(2, batch_size, i + batch_size + 1), plt.imshow(predict_images[i]), plt.axis('off')
        if i == 0:
            plt.text(0, -20, "Predict:",fontsize=60)
    plt.savefig('result_{}.pdf'.format(idx+1))
    plt.show()


if __name__ == '__main__':
    run(modelname="FPN",
    pretrained = False,
    batch_size = 4,
    num_epochs=50,
    validation_split = 0.05,
    run_train=True,
    run_test=True)
