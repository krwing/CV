import gc
import os.path
from operator import getitem

import cv2
import numpy as np
import skimage.measure
from PIL import Image
import torch
from torchvision.transforms import Compose, transforms

# midas imports

from transforms_resize import Resize, NormalizeImage, PrepareForNet
# zoedepth
from zoedepth.models.builder import build_model
from zoedepth.utils.config import get_config
#
from pix2pix.models.pix2pix4depth_model import Pix2Pix4DepthModel

from pix2pix.options.test_options import TestOptions

# Our code
from src.misc import *
from src import backbone

global depthmap_device
device = torch.device("cuda:0")

from utils import *


class ModelHolder:
    def __init__(self):
        self.depth_model_type = None
        self.device = None

        opt = TestOptions().parse()
        self.pix2pix_model = Pix2Pix4DepthModel(opt)
        self.pix2pix_model.save_dir = './models/pix2pix'
        self.pix2pix_model.load_networks('latest')
        self.pix2pix_model.eval()

        self.resize_mode = None
        self.normalization = None

        model_dir = "./models/midas"

        resize_mode = "minimal"
        normalization = NormalizeImage(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])

        conf = get_config("zoedepth_nk", "infer")
        model = build_model(conf)

        model.eval()

        model.to(device)

        self.depth_model = model
        self.depth_model_type = 9
        self.resize_mode = resize_mode
        self.normalization = normalization

        self.device = device

    def update_settings(self, **kvargs):
        # Opens the pandora box
        for k, v in kvargs.items():
            setattr(self, k, v)


    def get_raw_prediction(self, input, net_width=448, net_height=448):

        device = self.device
        img = cv2.cvtColor(np.asarray(input), cv2.COLOR_BGR2RGB) / 255.0

        raw_prediction = estimateboost(img, self.depth_model, self.depth_model_type, self.pix2pix_model, self.boost_rmax)


        return raw_prediction


def read_image(image_path, if_depthmap=False):
    I = Image.open(image_path)
    if if_depthmap:
        I = I.convert('L')
    #I.show()
    #I.save('./save.png')
    I_array = np.array(I)

    return I_array

origin = read_image("origin.jpg")
model_holder = ModelHolder()
ops = backbone.gather_ops()
model_holder.update_settings(**ops)
#model_holder.ensure_models(model_type=9, device='cuda:0', boost=True)
raw_prediction  =  model_holder.get_raw_prediction(origin)

from torchvision.utils import save_image
save_image(torch.from_numpy(-raw_prediction+1), "result/raw_prediction_t_.jpg")

print("scale:",np.max(raw_prediction), np.min(raw_prediction))
print("raw_prediction shape:",raw_prediction.shape)