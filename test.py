import cv2
import os
import argparse
import glob
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from models3 import ATDNet
from utils import *
from thop import profile
from  torchvision import utils as vutils
import pathlib

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

parser = argparse.ArgumentParser(description="ATDNet_Test")
parser.add_argument("--num_of_layers", type=int, default=17, help="Number of total layers")
parser.add_argument("--logdir", type=str, default="logs", help='path of log files')
parser.add_argument("--test_data", type=str, default='Set12', help='test on Set12 or Set68 kodak24')
parser.add_argument("--test_noiseL", type=float, default=50, help='noise level used on test set')
opt = parser.parse_args()

def normalize(data):
    return data/255.

def main():
    # Build model
    print('Loading model ...\n')
    net = ATDNet(channels=1, num_of_layers=opt.num_of_layers)

    device_ids = [0]
    model = nn.DataParallel(net, device_ids=device_ids).cuda()
    model.load_state_dict(torch.load(os.path.join(opt.logdir, '50.pth')))
    model.eval()
    # load data info
    print('Loading data info ...\n')
    files_source = glob.glob(os.path.join('data', opt.test_data, '*'))
    files_source.sort()
    # process data
    psnr_test = 0
    ssim_test = 0
    for f in files_source:
        # image
        Img = cv2.imread(f)
        Img = normalize(np.float32(Img[:,:,0]))
        Img = np.expand_dims(Img, 0)
        Img = np.expand_dims(Img, 1)
        ISource = torch.Tensor(Img)
        # noise
        torch.manual_seed(0)
        noise = torch.FloatTensor(ISource.size()).normal_(mean=0, std=opt.test_noiseL/255.)
        # noisy image
        INoisy = ISource + noise
        ISource = Variable(ISource)
        INoisy = Variable(INoisy)
        ISource= ISource.cuda()
        INoisy = INoisy.cuda()
        with torch.no_grad():
            Out = torch.clamp(model(INoisy), 0., 1.)


        psnr = batch_PSNR(Out, ISource, 1.)   # PSNR
        SSIM = ssim(Out, ISource)     # SSIM
        print(SSIM)
        #print(psnr)
        ssim_test += SSIM

        psnr_test += psnr
       # print("%s PSNR %f" % (f, psnr))
    psnr_test /= len(files_source)
    ssim_test /= len(files_source)
    print("\nPSNR on test data %f" % psnr_test)
    #print("\nSSIM on test data %f" % ssim_test)

if __name__ == "__main__":
    main()
