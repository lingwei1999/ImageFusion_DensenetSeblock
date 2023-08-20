import logging
from pathlib import Path

import cv2
import numpy as np

import torch.nn.functional as F
import torch
import os

import kornia
from kornia.losses import SSIMLoss
from kornia.metrics import AverageMeter
from kornia.filters import Sobel
from torch import nn, Tensor
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from models.DenseNet_Seblock import DenseNet as model
from models.Perceptual_net import Perceptual_net
from utils.environment_probe import EnvironmentProbe
from utils.fusion_data_msf import FusionData

class MeanConv(nn.Module):
    def __init__(self, size=3):
        super(MeanConv, self).__init__()

        self.sum_conv = nn.Conv2d(
            1, 1, size, 1, padding=size//2, dilation=1, padding_mode='reflect')
        self.sum_conv.weight.data[True] = 1/(size**2)
        self.sum_conv.bias.data.zero_()

    def forward(self, x):
        return torch.clamp(self.sum_conv(x), 1e-10)


class Sobelxy(nn.Module):
    def __init__(self):
        super(Sobelxy, self).__init__()
        kernelx = [[-1, 0, 1],
                   [-2, 0, 2],
                   [-1, 0, 1]]
        kernely = [[1, 2, 1],
                   [0, 0, 0],
                   [-1, -2, -1]]
        kernelx = torch.FloatTensor(kernelx).unsqueeze(0).unsqueeze(0)
        kernely = torch.FloatTensor(kernely).unsqueeze(0).unsqueeze(0)

        self.sobelx = nn.Conv2d(1, 1, 3, 1, padding=1,
                                dilation=1, padding_mode='reflect').cuda()
        self.sobelx.weight.data = kernelx
        self.sobelx.bias.data.zero_()

        self.sobely = nn.Conv2d(1, 1, 3, 1, padding=1,
                                dilation=1, padding_mode='reflect').cuda()
        self.sobely.weight.data = kernely
        self.sobely.bias.data.zero_()

    def forward(self, x):
        sobelx = self.sobelx(x)
        sobely = self.sobely(x)
        return torch.abs(sobelx)+torch.abs(sobely)


class Train:
    """
    The train process for TarDAL.
    """

    def __init__(self, environment_probe: EnvironmentProbe, config: dict):
        logging.info(f'Training')
        logging.info(f'ID: {config.id}')
        self.config = config
        self.environment_probe = environment_probe

        # modules
        self.net = model()
        if config.load != None:
            params = torch.load(config.load, map_location='cpu')
            self.net.load_state_dict(params['net'])
        para = sum([np.prod(list(p.size())) for p in self.net.parameters()])
        logging.info('Model params: {:}'.format(para))
        self.per_model = Perceptual_net()

        # WGAN adam optim
        logging.info(f'Adam | learning rate: {config.learning_rate}')
        self.opt_net = Adam(self.net.parameters(), lr=config.learning_rate)

        # move to device
        logging.info(f'module device: {environment_probe.device}')

        self.net.to(environment_probe.device)

        # loss

        self.MSE_Loss = torch.nn.MSELoss(reduction='none')
        self.MSE_Loss.to(environment_probe.device)
        self.SSIM = SSIMLoss(window_size=11, reduction='none')
        self.SSIM.to(environment_probe.device)

        self.sobel = Sobelxy()
        self.sobel.to(environment_probe.device)
        self.sum = MeanConv(31)
        self.sum.to(environment_probe.device)
        self.blur = transforms.GaussianBlur(13, 5)
        self.blur.to(environment_probe.device)

        # datasets
        folder = Path(config.folder)
        resize = transforms.Resize((config.size, config.size))
        train_dataset = TrainData(folder, mode='train', transforms=resize)
        self.train_dataloader = DataLoader(
            train_dataset, config.batch_size, True, num_workers=config.num_workers, pin_memory=True)

        eval_dataset = FusionData(folder, mode='val', transforms=resize)
        self.eval_dataloader = DataLoader(
            eval_dataset, config.batch_size, False, num_workers=config.num_workers, pin_memory=True)

        logging.info(
            f'dataset | folder: {str(folder)} | train size: {len(self.train_dataloader) * config.batch_size}')
        logging.info(
            f'dataset | folder: {str(folder)} |   val size: {len(self.eval_dataloader) * config.batch_size}')

    def loss_cal(self, gray_i1, gray_i2, fusion, fusion_inverse, epoch):

        grad_i1 = self.sobel(gray_i1)
        grad_i2 = self.sobel(gray_i2)
        grad_fusion = self.sobel(fusion)
        grad_fusion_inv = self.sobel(fusion_inverse)

        grad_blur_i1 = self.sobel(self.blur(gray_i1))
        grad_blur_i2 = self.sobel(self.blur(gray_i2))
        grad_blur_fusion = self.sobel(self.blur(fusion))
        grad_blur_fusion_inv = self.sobel(self.blur(fusion_inverse))
            
        detail_grad_i1 =  self.sum(grad_blur_i1)
        detail_grad_i1 = detail_grad_i1.view(detail_grad_i1.size(0), -1)
        detail_grad_i1 -= detail_grad_i1.min(1, keepdim=True)[0]
        detail_grad_i1 /= torch.clamp(detail_grad_i1.max(1, keepdim=True)[0], 1e-10)
        detail_grad_i1 = detail_grad_i1.view(grad_i1.shape)

        detail_grad_i2 = self.sum(grad_blur_i2)
        detail_grad_i2 = detail_grad_i2.view(detail_grad_i2.size(0), -1)
        detail_grad_i2 -= detail_grad_i2.min(1, keepdim=True)[0]
        detail_grad_i2 /= torch.clamp(detail_grad_i2.max(1, keepdim=True)[0], 1e-10)
        detail_grad_i2 = detail_grad_i2.view(grad_i1.shape)

        detail_i1 = detail_grad_i1
        detail_i2 = detail_grad_i2
        w1 = torch.clamp(detail_i1, 5e-11) / torch.clamp((detail_i1+detail_i2), 1e-10)
        w2 = 1-w1

        i_gt = gray_i1*w1 + gray_i2*w2
        i_gt_inv = (1-gray_i1)*w1 + (1-gray_i2)*w2

        l_int = ((self.MSE_Loss(fusion, i_gt) + self.SSIM(fusion, i_gt)).mean() + (self.MSE_Loss(fusion_inverse, i_gt_inv) + self.SSIM(fusion_inverse, i_gt_inv)).mean())*5


        #### Gradient
        if epoch<5:
            grad_blur_en = torch.max(grad_blur_i1*(w1*2), grad_blur_i2*(w2*2))
            mse_grad = self.MSE_Loss(grad_blur_fusion, grad_blur_en).mean()
            mse_grad_inv = self.MSE_Loss(grad_blur_fusion_inv, grad_blur_en).mean() 
            l_mse_grad = 10*(mse_grad + mse_grad_inv)
        else:
            grad_en = torch.max(grad_i1*(w1*2), grad_i2*(w2*2))
            mse_grad = self.MSE_Loss(grad_fusion, grad_en).mean()
            mse_grad_inv = self.MSE_Loss(grad_fusion_inv, grad_en).mean()
            l_mse_grad = (mse_grad + mse_grad_inv)

        l_grad = l_mse_grad


        # Common
        feat_f = self.per_model(fusion)
        feat_f_inv = self.per_model(1-fusion_inverse)
        l_common = torch.tensor([self.MSE_Loss(i, j).mean() for i, j in zip(feat_f, feat_f_inv)]).mean()

        loss = l_int + l_grad + l_common

        return [loss, l_int, l_common, l_grad, l_mse_grad]

    def train_generator(self, i1, i2, epoch) -> dict:
        """
        Train generator 'ir + vi -> fus'
        """

        logging.debug('train generator')

        self.net.train()

        fusion = self.net(i1, i2)
        fusion_inverse = self.net(1-i1, 1-i2)

        gray_i1 = transforms.functional.rgb_to_grayscale(i1, 1)
        gray_i2 = transforms.functional.rgb_to_grayscale(i2, 1)

        # calculate loss towards criterion
        loss, l_int, l_common, l_grad, l_mse_grad = self.loss_cal(gray_i1, gray_i2, fusion, fusion_inverse, epoch)
        
        # backwardG
        self.opt_net.zero_grad()
        loss.backward()
        self.opt_net.step()

        state = {
            'g_loss': loss.item(),
            'g_l_int': l_int.item(),
            'g_l_common': l_common.item(),
            'g_l_grad': l_grad.item(),
            'g_l_mse_grad': l_mse_grad.item(),
        }
        return state

    def eval_generator(self, i1, i2, epoch) -> dict:
        """
        Eval generator
        """

        logging.debug('eval generator')

        self.net.eval()

        fusion = self.net(i1, i2)
        fusion_inverse = self.net(1-i1, 1-i2)

        gray_i1 = transforms.functional.rgb_to_grayscale(i1, 1)
        gray_i2 = transforms.functional.rgb_to_grayscale(i2, 1)

        # calculate loss towards criterion
        loss = self.loss_cal(gray_i1, gray_i2, fusion, fusion_inverse, epoch)[0]

        return loss.item()

    def test_generator(self, epoch) -> dict:
        """
        Eval generator
        """
        self.net.eval()

        save = f'result/training/{self.config.id}'

        i1_test = cv2.imread('../data/test/ir/soldier_behind_smoke_1.bmp')
        i1_test = cv2.resize(i1_test, (512, 512))
        i2_test = cv2.imread('../data/test/vi/soldier_behind_smoke_1.bmp')
        i2_test = cv2.resize(i2_test, (512, 512))

        i1_test_t = kornia.utils.image_to_tensor(i1_test / 255.0).type(torch.FloatTensor).unsqueeze(0).to(self.environment_probe.device)
        i2_test_t = kornia.utils.image_to_tensor(i2_test / 255.0).type(torch.FloatTensor).unsqueeze(0).to(self.environment_probe.device)

        gray_i1 = transforms.functional.rgb_to_grayscale(i1_test_t, 1)
        gray_i2 = transforms.functional.rgb_to_grayscale(i2_test_t, 1)

        black = torch.zeros(i1_test_t.shape).to(self.environment_probe.device)
        black[...] = 0
        
        white = torch.zeros(i1_test_t.shape).to(self.environment_probe.device)
        white[...] = 1

        fusion = self.net(i1_test_t, i2_test_t)

        fusion_black_ir = self.net(black, i2_test_t)
        fusion_white_ir = self.net(white, i2_test_t)
        
        fusion_black_vi = self.net(i1_test_t, black)
        fusion_white_vi = self.net(i1_test_t, white)
        

        def tensor2np(img):
            img = img.squeeze().cpu()
            img = kornia.utils.tensor_to_image(img) * 255.
            return img
            
        fusion = tensor2np(fusion)

        fusion_black_ir = tensor2np(fusion_black_ir)
        fusion_white_ir = tensor2np(fusion_white_ir)

        fusion_black_vi = tensor2np(fusion_black_vi)
        fusion_white_vi = tensor2np(fusion_white_vi)

        gray_i1 = tensor2np(gray_i1)
        gray_i2 = tensor2np(gray_i2)

        os.makedirs(save, exist_ok=True)
        cv2.imwrite(f'{save}/{epoch:03d}.jpg', np.vstack([np.hstack([fusion_black_ir, fusion, fusion_white_ir]), \
                                                          np.hstack([fusion_black_vi, fusion, fusion_white_vi])]))


    def run(self):
        best = float('Inf')
        best_epoch = 0

        for epoch in range(1, self.config.epochs + 1):
            if epoch == 5:
                for param_group in self.opt_net.param_groups:
                    param_group['lr'] = self.config.learning_rate*1e-2

            train_process = tqdm(enumerate(self.train_dataloader), total=len(self.train_dataloader))

            meter = AverageMeter()
            for idx, train_sample in train_process:

                train_img1 = train_sample['img1']
                train_img2 = train_sample['img2']

                train_img1 = train_img1.to(self.environment_probe.device)
                train_img2 = train_img2.to(self.environment_probe.device)

                g_loss = self.train_generator(train_img1, train_img2, epoch)

                meter.update(g_loss["g_loss"])

                train_process.set_description(f'g: {g_loss["g_loss"]:03f} | l_int: {g_loss["g_l_int"]:03f} | common_loss: {g_loss["g_l_common"]:03f} | l_grad: {g_loss["g_l_grad"]:03f} | l_mse_grad: {g_loss["g_l_mse_grad"]:03f}')

            with torch.no_grad():
                # Eval Generator
                total_loss = 0
                eval_process = tqdm(enumerate(self.eval_dataloader), total=len(self.eval_dataloader))
                for idx, eval_sample in eval_process:
                    eval_img1 = eval_sample['img1']
                    eval_img2 = eval_sample['img2']

                    eval_img1 = eval_img1.to(self.environment_probe.device)
                    eval_img2 = eval_img2.to(self.environment_probe.device)

                    g_loss = self.eval_generator(eval_img1, eval_img2, epoch)
                    total_loss += g_loss

                # self.test_generator(epoch)

            logging.info(f'[{epoch}] g_loss: {meter.avg:03f}')
            logging.info(f'[eval_{epoch}] g_loss: {total_loss/len(eval_process):03f}')

            if epoch % 1 == 0:

                if best > total_loss/len(eval_process):
                    best = total_loss/len(eval_process)
                    best_epoch = epoch

                    self.save(epoch, is_best=True)

                self.save(epoch)
                logging.info(f'best epoch is {best_epoch}, loss is {best}')

    def save(self, epoch: int, is_best=False):
        path = Path(self.config.cache) / self.config.id
        path.mkdir(parents=True, exist_ok=True)

        if is_best:
            cache = path / f'best.pth'
        else:
            cache = path / f'{epoch:03d}.pth'
        logging.info(f'save checkpoint to {str(cache)}')

        state = {
            'net': self.net.state_dict(),
        }

        torch.save(state, cache)
