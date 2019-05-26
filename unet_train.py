from models.unet import UNet
from dataloaders.dataset.sem import SEMData
import torch
import numpy as np
import torch.nn as nn


def accuracy_check(mask, prediction):
    ims = [mask, prediction]
    np_ims = []
    for item in ims:
        if 'str' in str(type(item)):
            item = np.array(Image.open(item))
        elif 'PIL' in str(type(item)):
            item = np.array(item)
        elif 'torch' in str(type(item)):
            item = item.numpy()
        np_ims.append(item)

    compare = np.equal(np_ims[0], np_ims[1])
    accuracy = np.sum(compare)

    return accuracy / len(np_ims[0].flatten())


def accuracy_check_for_batch(masks, predictions, batch_size):
    total_acc = 0
    for index in range(batch_size):
        total_acc += accuracy_check(masks[index], predictions[index])
    return total_acc / batch_size


def train_model(model, data_train, criterion, optimizer):

    model.train()

    for batch, sample in enumerate(data_train):
        images = sample['image'].cuda()
        masks = sample['label'].cuda()
        outputs = model(images)
        loss = criterion(outputs, masks.long())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


def get_loss_and_accuracy(model, data_train, criterion):
    model.eval()
    total_acc = 0
    total_loss = 0
    for batch, sample in enumerate(data_train):
        with torch.no_grad():
            images = sample['image'].cuda()
            masks = sample['label'].cuda()
            outputs = model(images)
            loss = criterion(outputs, masks.long())
            preds = torch.argmax(outputs, dim=1).float()
            acc = accuracy_check_for_batch(masks.cpu(), preds.cpu(), images.size()[0])
            total_acc = total_acc + acc
            total_loss = total_loss + loss.cpu().item()
    return total_acc / (batch + 1), total_loss / (batch + 1)


if __name__ == "__main__":

    SEM_train = SEMData(512, 'train')

    SEM_val = SEMData(512, 'val')

    SEM_train_load = torch.utils.data.DataLoader(dataset=SEM_train,
                                                 num_workers=16, batch_size=2, shuffle=True)

    SEM_val_load = torch.utils.data.DataLoader(dataset=SEM_val,
                                               num_workers=3, batch_size=1, shuffle=True)

    model = UNet(1, 2)
    model = torch.nn.DataParallel(model, device_ids=list(range(torch.cuda.device_count()))).cuda()

    # Loss function
    criterion = nn.CrossEntropyLoss(reduction='mean')

    # Optimizerd
    optimizer = torch.optim.RMSprop(model.module.parameters(), lr=0.001)

    # Parameters
    epoch_start = 0
    epoch_end = 2000

    # Train
    print("Initializing Training!")
    for i in range(epoch_start, epoch_end):
        # train the model
        train_model(model, SEM_train_load, criterion, optimizer)
        train_acc, train_loss = get_loss_and_accuracy(model, SEM_train_load, criterion)

        #train_loss = train_loss / len(SEM_train)
        print('Epoch', str(i + 1), 'Train loss:', train_loss, "Train acc", train_acc)

        # Validation every 5 epoch
        if (i + 1) % 5 == 0:
            val_acc, val_loss = get_loss_and_accuracy(model, SEM_val_load, criterion)
            print('Val loss:', val_loss, "val acc:", val_acc)
