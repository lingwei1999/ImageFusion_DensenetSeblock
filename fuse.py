import pathlib
import statistics
import time
import argparse
import cv2
import kornia
from thop import profile
import torch
from torch import nn
from torchvision import transforms
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
import os

# from models.FusionNet import FusionNet as Model
# from models.DenseNet import DenseNet as Model
from models.DenseNet_Seblock import DenseNet as Model


class Fuse:
    """
    fuse with infrared folder and visible folder
    """

    def __init__(self, model_path: str):
        """
        :param model_path: path of pre-trained parameters
        """

        # device
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.device = device
        # model parameters
        params = torch.load(model_path, map_location='cpu')

        self.net = Model()

        self.net.load_state_dict(params['net'])

        self.net.to(device)
        
        self.net.eval()
        for param in self.net.parameters():
            param.grad = None

    def __call__(self, i1_folder: str, i2_folder: str, dst: str, fuse_type = None):
        """
        fuse with i1 folder and vi folder and save fusion image into dst
        :param i1_folder: infrared image folder
        :param vi_folder: visible image folder
        :param dst: fusion image output folder
        """

        para = sum([np.prod(list(p.size())) for p in self.net.parameters()])
        print('Model params: {:}'.format(para))
        
        # image list
        i1_folder = pathlib.Path(i1_folder)
        i2_folder = pathlib.Path(i2_folder)
        i1_list = sorted([x for x in sorted(i1_folder.glob('*')) if x.suffix in ['.bmp', '.png', '.jpg', '.JPG']])
        i2_list = sorted([x for x in sorted(i2_folder.glob('*')) if x.suffix in ['.bmp', '.png', '.jpg', '.JPG']])

        # check image name and fuse
        fuse_time = []
        rge = tqdm(zip(i1_list, i2_list))

        for i1_path, i2_path in rge:
            start = time.time()

            # check image name
            i1_name = i1_path.stem
            i2_name = i2_path.stem
            rge.set_description(f'fusing {i1_name}')
            # assert i1_name == vi_name

            # read image
            i1, i2 = self._imread(str(i1_path), str(i2_path), fuse_type = fuse_type)
            i1 = i1.unsqueeze(0).to(self.device)
            i2 = i2.unsqueeze(0).to(self.device)

            # network forward
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            fu = self._forward(i1, i2)
            torch.cuda.synchronize() if torch.cuda.is_available() else None


            # save fusion tensor
            fu_path = pathlib.Path(dst, i1_path.name)
            self._imsave(fu_path, fu)

            end = time.time()
            fuse_time.append(end - start)
        
        # time analysis
        if len(fuse_time) > 2:
            mean = statistics.mean(fuse_time[1:])
            print('fps (equivalence): {:.2f}'.format(1. / mean))

        else:
            print(f'fuse avg time: {fuse_time[0]:.2f}')


    @torch.no_grad()
    def _forward(self, i1: torch.Tensor, i2: torch.Tensor) -> torch.Tensor:
        gray_i1 = transforms.functional.rgb_to_grayscale(i1, 1)
        gray_i2 = transforms.functional.rgb_to_grayscale(i2, 1)
        
        fusion = self.net(i1, i2)
        # calculate loss towards criterion
        
        # grad_i1 = self.sobel(self.blur(gray_i1))
        # grad_i2 = self.sobel(self.blur(gray_i2))

        # detail_grad_i1 = self.sum(grad_i1)
        # detail_grad_i1 = detail_grad_i1.view(detail_grad_i1.size(0), -1)
        # detail_grad_i1 -= detail_grad_i1.min(1, keepdim=True)[0]
        # detail_grad_i1 /= torch.clamp(detail_grad_i1.max(1, keepdim=True)[0], 1e-10)
        # detail_grad_i1 = detail_grad_i1.view(grad_i1.shape)
        
        # detail_grad_i2 = self.sum(grad_i2)
        # detail_grad_i2 = detail_grad_i2.view(detail_grad_i2.size(0), -1)
        # detail_grad_i2 -= detail_grad_i2.min(1, keepdim=True)[0]
        # detail_grad_i2 /= torch.clamp(detail_grad_i2.max(1, keepdim=True)[0], 1e-10)
        # detail_grad_i2 = detail_grad_i2.view(grad_i1.shape)
        
        # detail_i1 = detail_grad_i1
        # detail_i2 = detail_grad_i2
        # w1 = detail_i1/torch.clamp((detail_i1+detail_i2), 1e-10)
        # w2 = 1-w1
        # i_gt = gray_i1*w1 + gray_i2*w2
        # grad = torch.clamp(torch.max(grad_i1, grad_i2)*(1+torch.max(gray_i1, gray_i2)), 0, 1)
        # grad_inv = torch.clamp(torch.max(grad_i1, grad_i2)*(1+torch.max(1-gray_i1, 1-gray_i2)), 0, 1)

        # return torch.clamp(torch.max(grad_i1*torch.clamp(w1+0.5, 1), grad_i2*torch.clamp(w2+0.5, 1)), 0 , 1)
        return fusion

    @staticmethod
    def _imread(i1_path: str, i2_path: str, flags=cv2.IMREAD_GRAYSCALE, fuse_type = None) -> torch.Tensor:
        i1_cv = cv2.imread(i1_path).astype('float32')
        # i1_cv[:,:,:] = 255
        i2_cv = cv2.imread(i2_path).astype('float32')
        # i2_cv[:,:,:] = 255
        i1_cv = cv2.resize(i1_cv, (640,480))
        i2_cv = cv2.resize(i2_cv, (640,480))
        
        # fu_cr = (i1_cr*cv2.absdiff(i1_cr, 128) + i2_cr*cv2.absdiff(i2_cr, 128))/np.maximum(cv2.absdiff(i1_cr, 128) + cv2.absdiff(i2_cr, 128), 0.0001)
        # fu_cb = (i1_cb*cv2.absdiff(i1_cb, 128) + i2_cb*cv2.absdiff(i2_cb, 128))/np.maximum(cv2.absdiff(i1_cb, 128) + cv2.absdiff(i2_cb, 128), 0.0001)
        
        i1_ts = kornia.utils.image_to_tensor(i1_cv / 255.0).type(torch.FloatTensor)
        i2_ts = kornia.utils.image_to_tensor(i2_cv / 255.0).type(torch.FloatTensor)
        return i1_ts, i2_ts

    @staticmethod
    def _imsave(path: pathlib.Path, image: torch.Tensor):
        im_ts = image.squeeze().cpu()
        path.parent.mkdir(parents=True, exist_ok=True)
        im_cv = kornia.utils.tensor_to_image(im_ts) * 255.

        cv2.imwrite(str(path), im_cv)



if __name__ == '__main__':
    model = 'default_maxWeightGT'
    f = Fuse(f"./cache/{model}/best.pth")

    parser = argparse.ArgumentParser()
    parser.add_argument("--ir", default='../datasets/test/ir', help="ir path")
    parser.add_argument("--vi", default='../datasets/test/vi', help="vi path")
    parser.add_argument("--result", default=f'../result/{model}/test', help="result path")
    args = parser.parse_args()
    
    f(args.ir, args.vi, args.result)




