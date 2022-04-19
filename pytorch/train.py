# -*- coding: utf-8 -*-
import torch
from Create_Dataset import *
from Model import *
from torch import optim
from torch.autograd import Variable
import logging
from torch.utils.data import random_split
import Model
import loss as L
import FocalLoss as FL
import io
import matplotlib.pyplot as plt
import time, h5py

if __name__ == '__main__':

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # inputs = inputs.to(device)


    # img1_path = './data/change_detection_train/train/im1/'
    # img2_path = './data/change_detection_train/train/im2/'
    # label1_path = './data/change_detection_train/train/label1/'
    # label2_path = './data/change_detection_train/train/label2/'

    img1_path = './data/change_detection_train/train_aug/im1/'
    img2_path = './data/change_detection_train/train_aug/im2/'
    label1_path = './data/change_detection_train/train_aug/label1/'
    label2_path = './data/change_detection_train/train_aug/label2/'
    dataset = CDDataset(img1_path, img2_path, label1_path, label2_path, aug=1)
    # h5_path = './data/training_data.hdf5'
    # h5_file = h5py.File(h5_path, 'r')
    # dataset = CDDataset(h5_file)

    # hyper-parameters
    size = 4
    epochs = 100
    lr = .007

    # # split validation
    # val_split = .2
    #
    # n_val = int(len(dataset) * val_split)
    # n_train = len(dataset) - n_val
    # train, val = random_split(dataset, [n_train, n_val])
    #
    # train_loader = DataLoader(dataset, batch_size=size, shuffle=True, num_workers=2)
    # val_loader = DataLoader(val, batch_size=size, shuffle=True, num_workers=2)

    train_loader = DataLoader(dataset, batch_size=size, shuffle=True, num_workers=2)

    # net = Model.CDNet().to(device)
    net = Model.CDNet_Resnet().to(device)
    # net.load_state_dict(torch.load('./log_files/batch4_epoch100_CE_L1_Loss_params_ResnetBackbone.pkl'))

    weight = torch.tensor([0.5, 1.5, 1, 0.5, 1, 0.5, 2])

    # criterion_mDice = L.DiceLoss().to(device)
    criterion_CE = FL.FocalLoss2d(weight=weight).to(device)
    # criterion_CE = nn.CrossEntropyLoss(weight=weight).to(device)
    # criterion_L1 = nn.L1Loss().to(device)

    SGD = False
    if SGD == True:
        optimizer = optim.Adam(net.parameters(),
                               lr=lr, weight_decay=0.0001)
    else:
        optimizer = optim.SGD(net.parameters(),
                              lr=lr,
                              momentum=0.9, weight_decay=0.0001)

    # scheduler = optim.lr_scheduler.MultiStepLR(optimizer,
    #                                            [10, 20, 30, 40, 50])

    # scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5)

    weight1 = torch.tensor(0.5)
    weight2 = torch.tensor(0.5)
    weight3 = torch.tensor(0.)

    # record loss and acc
    log_file = io.open("./log_files/log_lr_0.007_2.txt", "w")

    print('Start training...')
    for epoch in range(0, epochs):

        print('learning rate:', optimizer.param_groups[0]['lr'])
        train_loss = 0
        val_loss = 0
        train_acc = 0
        val_acc = 0

        time_start = time.time()

        for i_batch, train_batched in enumerate(train_loader):
            optimizer.zero_grad()
            img1 = train_batched['image1']
            img2 = train_batched['image2']

            label1 = train_batched['label_binary1']
            label2 = train_batched['label_binary2']

            label1_onehot = train_batched['onehot1']
            label2_onehot = train_batched['onehot2']

            tensor1 = Variable(img1.float()).to(device)
            tensor2 = Variable(img2.float()).to(device)

            label1_onehot = label1_onehot.to(device).float()
            label2_onehot = label2_onehot.to(device).float()

            # net input
            ## 3 channels input
            # output1, output2 = net(tensor1, tensor2)
            # 4 channels input
            output1, output2 = net(tensor1, tensor2)

            loss1 = criterion_CE(output1, label1.long().to(device))
            loss2 = criterion_CE(output2, label2.long().to(device))

            # # backgroud loss should be equal
            # loss3 = criterion_L1(output1[:, 0, :, :], output2[:, 0, :, :])
            # loss = weight1 * loss1 + weight2 * loss2 + weight3*loss3
            loss = weight1 * loss1 + weight2 * loss2

            acc1 = L.dice_coeff(output1, label1_onehot)
            acc2 = L.dice_coeff(output2, label2_onehot)
            acc = 0.5 * acc1.item() + 0.5 * acc2.item()
            #acc = 0.

            loss.backward()
            optimizer.step()

            train_loss += loss1.item()
            train_acc += acc

        scheduler.step(train_loss)

        # for i_batch, val_batched in enumerate(val_loader):
        #     img1 = train_batched['image1']
        #     img2 = train_batched['image2']
        #     # img = train_batched['image']
        #
        #     label1 = train_batched['label_binary1']
        #     label2 = train_batched['label_binary2']
        #
        #     label1_onehot = train_batched['onehot1'].float().to(device)
        #     label2_onehot = train_batched['onehot2'].float().to(device)
        #
        #     VI1 = train_batched['VI1'].float().to(device)
        #     VI2 = train_batched['VI2'].float().to(device)
        #
        #     tensor1 = Variable(img1.float()).to(device)
        #     tensor2 = Variable(img2.float()).to(device)
        #     # tensor = Variable(img.float()).to(device)
        #     # target1 = Variable(label1.long()).to(device)
        #     # target2 = Variable(label2.long()).to(device)
        #
        #     # net input
        #     # semantic_img1, semantic_img2 = net(tensor1, tensor2)
        #     # 6 channels input
        #     semantic_img1, semantic_img2 = net(tensor1, tensor2, VI1, VI2)
        #
        #     loss1 = criterion_CE(semantic_img1, label1.long().to(device))
        #     loss2 = criterion_CE(semantic_img2, label2.long().to(device))
        #     # loss3 = criterion_BCE(semantic_img1, binaryCD)
        #     acc1 = L.dice_coeff(semantic_img1, label1_onehot)
        #     acc2 = L.dice_coeff(semantic_img2, label2_onehot)
        #
        #     # loss = 0.4*loss1+0.4*loss2+0.2*loss3
        #     loss = weight1 * loss1 + weight2 * loss2
        #     acc = 0.5 * acc1.item() + 0.5 * acc2.item()
        #
        #     val_loss += loss.item()
        #     val_acc += acc
        #
        #     del tensor1, tensor2, label1, label2, \
        #         loss1, loss2, acc1, acc2, loss, acc, \
        #         semantic_img1, semantic_img2

        print('Epoch:[{}/{}]\t train_loss={:.5f}\t acc={:.3f}\t val_loss={:.5f}\t val_acc={:.3f}'
              .format(epoch, epochs, train_loss / len(train_loader), train_acc / len(train_loader),
                      val_loss / 1, val_acc / 1))
        # record loss and acc
        log_file.writelines('Epoch:[{}/{}]\t train_loss={:.5f}\t acc={:.3f}\t val_loss={:.5f}\t val_acc={:.3f}'
                            .format(epoch, epochs, train_loss / len(train_loader), train_acc / len(train_loader),
                                    val_loss / 1, val_acc / 1) + '\n')
        time_end = time.time()
        print('time cost', time_end - time_start, 's')

    print('finish training')
    log_file.close()  # 关闭文件
    torch.save(net.state_dict(), './log_files/batch4_epoch100_Focal_Loss_params_ResnetBackbone_Dilate.pkl')
    # model_object.load_state_dict(torch.load('params.pkl'))
    #
    #     # colormap
    #     # colormap = {
    #     #     [255, 255, 255],
    #     #     [0, 0, 255],
    #     #     [128, 128, 128],
    #     #     [0, 128, 0],
    #     #     [0, 255, 0],
    #     #     [128, 0, 0],
    #     #     [255, 0, 0]
    #     # }
    #     # class_name = {
    #     #     'unchanged',
    #     #     'water',
    #     #     'groud',
    #     #     'vegtation',
    #     #     'tree',
    #     #     'building',
    #     #     'court'
    #     # }
