import pathlib
import statistics
import time

import cv2
import kornia
import torch
from torch import nn
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np

# from models.Unet import UNet as Model
# from models.Unet import UNet_lite as Model
from models.net_densefuse import DenseFuse_net as Model

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
        # self.net = DenseFuse_net()


        self.net.load_state_dict(params['net'])

        self.net.to(device)
        
        self.net.eval()

    def __call__(self, ir_folder: str, vi_folder: str, dst: str):
        """
        fuse with ir folder and vi folder and save fusion image into dst
        :param ir_folder: infrared image folder
        :param vi_folder: visible image folder
        :param dst: fusion image output folder
        """

        # encoder_params = sum(p.numel() for p in self.encoder.parameters())
        # decoder_params = sum(p.numel() for p in self.decoder.parameters())
        # print('encoder params: ', encoder_params)
        # print('decoder params: ', decoder_params)
        # print('total params:   ', decoder_params + encoder_params)

        para = sum([np.prod(list(p.size())) for p in self.net.parameters()])
        print('Model params: {:}'.format(para))

        # image list
        ir_folder = pathlib.Path(ir_folder)
        vi_folder = pathlib.Path(vi_folder)
        ir_list = sorted([x for x in sorted(ir_folder.glob('*')) if x.suffix in ['.bmp', '.png', '.jpg']])
        vi_list = sorted([x for x in sorted(vi_folder.glob('*')) if x.suffix in ['.bmp', '.png', '.jpg']])

        # check image name and fuse
        fuse_time = []
        rge = tqdm(zip(ir_list, vi_list))

        for ir_path, vi_path in rge:
            start = time.time()

            # check image name
            ir_name = ir_path.stem
            vi_name = vi_path.stem
            rge.set_description(f'fusing {ir_name}')
            assert ir_name == vi_name

            # read image
            ir, vi = [i.unsqueeze(0) for i in self._imread(str(ir_path), str(vi_path))]
            ir = ir.to(self.device)
            vi = vi.to(self.device)

            # network forward
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            fu = self._forward(ir, vi)
            torch.cuda.synchronize() if torch.cuda.is_available() else None

            # save fusion tensor
            fu_path = pathlib.Path(dst, ir_path.name)
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
    def _forward(self, ir: torch.Tensor, vi: torch.Tensor) -> torch.Tensor:
        fusion = self.net(ir, vi)
        return fusion

    @staticmethod
    def _imread(ir_path: str, vi_path: str, flags=cv2.IMREAD_GRAYSCALE) -> torch.Tensor:
        ir_cv = cv2.imread(ir_path, flags)
        
        vi_cv = cv2.imread(vi_path, flags)
        height, width = ir_cv.shape[:2]
        vi_cv = cv2.resize(vi_cv, (width, height))

        # im_cv = cv2.resize(im_cv, (640,480))
        ir_ts = kornia.utils.image_to_tensor(ir_cv / 255.0).type(torch.FloatTensor)
        vi_ts = kornia.utils.image_to_tensor(vi_cv / 255.0).type(torch.FloatTensor)
        return ir_ts, vi_ts

    @staticmethod
    def _imsave(path: pathlib.Path, image: torch.Tensor):
        im_ts = image.squeeze().cpu()
        path.parent.mkdir(parents=True, exist_ok=True)
        im_cv = kornia.utils.tensor_to_image(im_ts) * 255.
        cv2.imwrite(str(path), im_cv)


if __name__ == '__main__':

    model = 'densenet'
    f = Fuse(f"./cache/{model}/best.pth")
    # f('data/TNO/Nato/thermal', 'data/TNO/Nato/visual', f'runs/TNO/Nato/{model}')
    # f('data/TNO/Tree/thermal', 'data/TNO/Tree/visual', f'runs/TNO/Tree/{model}')
    # f('data/TNO/Duine/thermal', 'data/TNO/Duine/visual', f'runs/TNO/Duine/{model}')
    # f('data/TNO/Triclobs/thermal', 'data/TNO/Triclobs/visual', f'runs/TNO/Triclobs/{model}')
    
    # f('../datasets/Multi_spectral/infrared/val', '../datasets/Multi_spectral/visible/val', f'runs/Multi_spectral/val/{model}')
    # f('../datasets/Multi_spectral/infrared/test', '../datasets/Multi_spectral/visible/test', f'runs/Multi_spectral/test/{model}')
    # f('../datasets/Multi_spectral/infrared_list', '../datasets/Multi_spectral/visible_list', f'runs/Multi_spectral/list/{model}')
    # f('../datasets/Multi_spectral/infrared/train', '../datasets/Multi_spectral/visible/train', f'runs/Multi_spectral/train/{model}')
    
    # f('../datasets/M3FD/M3FD_Fusion/Ir', '../datasets/M3FD/M3FD_Fusion/Vis', f'runs/M3FD/M3FD_Fusion/{model}')

    f('../datasets/test/ir', '../datasets/test/vi', f'runs/test/{model}')
    # f('../datasets/LLVIP640/infrared/test', '../datasets/LLVIP640/visible/test', f'runs/LLVIP640/test/{model}')
    # f('data/clips6/ir', 'data/clips6/vi', f'runs/clips6/{model}')
