import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.utils as utils
import time  # tcw20182159tcw
# import pytorch_fft.fft as fft
from torch.autograd import Variable
from torch.utils.data import DataLoader
# from tensorboardX import SummaryWriter
from torch.nn.modules.loss import _Loss
from models3 import ATDNet
from dataset import prepare_data, Dataset
from utils import *
from thop import profile
from torchstat import stat
import torch

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

parser = argparse.ArgumentParser(description="ATDNet")
parser.add_argument("--preprocess", type=bool, default=False, help='run prepare_data or not') # True  False
parser.add_argument("--batchSize", type=int, default=128, help="Training batch size")
parser.add_argument("--num_of_layers", type=int, default=17, help="Number of total layers")  #This parameter is not required
parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
parser.add_argument("--milestone", type=int, default=20, help="When to decay learning rate; should be less than epochs") #20
parser.add_argument("--lr", type=float, default=1e-2, help="Initial learning rate")
parser.add_argument("--outf", type=str, default="logs", help='path of log files')
parser.add_argument("--mode", type=str, default="S", help='S')  # DEFAULT = S
parser.add_argument("--noiseL", type=float, default=25, help='noise level; ignored when mode=B')
parser.add_argument("--val_noiseL", type=float, default=25, help='noise level used on validation set')
'''
parser.add_argument("--clip",type=float,default=0.005,help='Clipping Gradients. Default=0.4') #tcw201809131446tcw
parser.add_argument("--momentum",default=0.9,type='float',help = 'Momentum, Default:0.9') #tcw201809131447tcw
parser.add_argument("--weight-decay","-wd",default=1e-3,type=float,help='Weight decay, Default:1e-4') #tcw20180913347tcw
'''
opt = parser.parse_args()


def tv_norm(x, beta=4.0):
    img = x[0]
    dy = img - img
    dx = img - img
    dx[:, 1:, :] = -img[:, :-1, :] + img[:, 1:, :]
    dy[:, :, 1:] = -img[:, :, :-1] + img[:, :, 1:]
    return ((dx.pow(2) + dy.pow(2)).pow(beta / 2.)).sum()


def main():
    # Load dataset
    save_dir = opt.outf + 'sigma' + str(opt.noiseL) + '_' + time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime()) + '/'
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    # Load dataset
    print('Loading dataset ...\n')
    dataset_train = Dataset(train=True)
    dataset_val = Dataset(train=False)
    loader_train = DataLoader(dataset=dataset_train, num_workers=4, batch_size=opt.batchSize, shuffle=True)
    print("# of training samples: %d\n" % int(len(dataset_train)))
    # Build model
    net = ATDNet(channels=1, num_of_layers=opt.num_of_layers)


    criterion = nn.MSELoss(size_average=False)
    # criterion = sum_squared_error()
    # Move to GPU
    device_ids = [0]

    #model = models.__dict__[net]()
    model = nn.DataParallel(net, device_ids=device_ids).cuda()


    '''Load pretraining parameters'''
    #model.load_state_dict(torch.load('./logs/model_24.pth'))


    criterion.cuda()
    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=opt.lr)
    # training
    # writer = SummaryWriter(opt.outf)
    # step = 0
    noiseL_B = [0, 55]
    psnr_list = []

    # for epoch in range(opt.epochs):
    #     if epoch <= opt.milestone:
    #         current_lr = opt.lr
    #     if epoch > 15 and epoch <= 20:   #大于30  小于60
    #         current_lr = opt.lr / 2.  #0.005
    #     if epoch > 20 and epoch <= 40:  # 大于60 小于90
    #         current_lr = opt.lr / 5.  #0.001
    #     if epoch > 40 and epoch <= 180:
    #         current_lr = opt.lr / 1000.

    for epoch in range(opt.epochs):
        if epoch <= opt.milestone:
            current_lr = opt.lr  #0.01
        if epoch > 20 and epoch <= 40:  # 大于30  小于60
            current_lr = opt.lr / 10.  # 0.001
        if epoch > 40 and epoch <= 60:  # 大于60 小于90
            current_lr = opt.lr / 10.  # 0.0001
        if epoch > 60 and epoch <= 180:
            current_lr = opt.lr / 10.

        # set learning rate
        for param_group in optimizer.param_groups:
            param_group["lr"] = current_lr
        print('learning rate %f' % current_lr)
        # train
        for i, data in enumerate(loader_train, 0):
            # training step
            model.train()
            img_train = data
            if opt.mode == 'S':
                noise = torch.FloatTensor(img_train.size()).normal_(mean=0, std=opt.noiseL / 255.)
                # noise = torch.FloatTensor(img_train.size()).normal_(mean=0, std=1)*opt.noiseL/255.
            if opt.mode == 'B':
                noise = torch.zeros(img_train.size())
                stdN = np.random.uniform(noiseL_B[0], noiseL_B[1], size=noise.size()[0])
                for n in range(noise.size()[0]):
                    sizeN = noise[0, :, :, :].size()
                    noise[n, :, :, :] = torch.FloatTensor(sizeN).normal_(mean=0, std=stdN[n] / 255.)
                    # noise[n,:,:,:] = torch.FloatTensor(sizeN).normal_(mean=0,std=1)*stdN[n]/255.

            imgn_train = img_train + noise


            img_train, imgn_train = Variable(img_train.cuda()), Variable(imgn_train.cuda())
            noise = Variable(noise.cuda())


            out_train = model(imgn_train)

            loss = criterion(out_train, img_train) / (imgn_train.size()[0] * 2)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            model.eval()

            # 计算参数
            # flops, params = profile(model, (imgn_train,))
            # print('flops: ', flops, 'params: ', params)


            out_train = torch.clamp(model(imgn_train), 0., 1.)
            psnr_train = batch_PSNR(out_train, img_train, 1.)
            print("[epoch %d][%d/%d] loss: %.4f PSNR_train: %.4f" %
                  (epoch + 1, i + 1, len(loader_train), loss.item(), psnr_train))
            # if you are using older version of PyTorch, you may need to change loss.item() to loss.data[0]
            '''            
            if step % 10 == 0:
                # Log the scalar values
                writer.add_scalar('loss', loss.item(), step)
                writer.add_scalar('PSNR on training data', psnr_train, step)
            step += 1
            '''
        ## the end of each epoch
        model.eval()  # tcw20180915tcw
        '''
        model_name = 'model'+ '_' + str(epoch+1) + '.pth' #tcw201809071117tcw
        torch.save(model.state_dict(), os.path.join(save_dir, model_name)) #tcw201809062210tcw
        '''
        '''
        for param in model.parameters():
            param.requires_grad = False
        '''
        # validate
        with torch.no_grad():
            psnr_val = 0
            for k in range(len(dataset_val)):
                img_val = torch.unsqueeze(dataset_val[k], 0)
                torch.manual_seed(0)  # set the seed,tcw201809030915
                noise = torch.FloatTensor(img_val.size()).normal_(mean=0, std=opt.val_noiseL / 255.)
                # noise = torch.FloatTensor(img_val.size()).normal_(mean=0, std=1)*opt.val_noiseL/255.
                imgn_val = img_val + noise
                img_val, imgn_val = Variable(img_val.cuda()), Variable(imgn_val.cuda(), requires_grad=False)

                # print 'a'
                # out_val = torch.clamp(imgn_val-model(imgn_val), 0., 1.)
                out_val = torch.clamp(model(imgn_val), 0., 1.)
                # print 'b'
                psnr_val += batch_PSNR(out_val, img_val, 1.)
            psnr_val /= len(dataset_val)
            psnr_val1 = str(psnr_val)
            psnr_list.append(psnr_val1)
            print("\n[epoch %d] PSNR_val: %.4f" % (epoch + 1, psnr_val))

            # filename = save_dir + 'psnr.txt'
            # f = open(filename, 'w')
            # f.write(psnr_val1+'\n')
        #f.close()
        model_name = 'model' + '_' + str(epoch + 1) + '.pth'
        torch.save(model.state_dict(), os.path.join(save_dir, model_name))

    filename = save_dir + 'psnr.txt'
    f = open(filename, 'w')
    for line in psnr_list:
        f.write(line + '\n')
    f.close()


if __name__ == "__main__":
    if opt.preprocess:
        if opt.mode == 'S':
            prepare_data(data_path='data', patch_size=50, stride=20, aug_times=1)
            # prepare_data(data_path='data', patch_size=50, stride=40, aug_times=1)

    main()
