import torch
import importlib
import time
from PIL import Image
import torchvision.transforms as transforms
from collections import OrderedDict
import numpy as np

import Flask.CARN_pytorch.carn.model.carn_m as module

# module = importlib.import_module("model.carn_m")
# CARN_SRnet = module.Net(multi_scale=True,group=4)
# # print(json.dumps(vars(cfg), indent=4, sort_keys=True))
#
# state_dict = torch.load("./checkpoint/carn_m.pth")
# new_state_dict = OrderedDict()
# for k, v in state_dict.items():
#     name = k
#     new_state_dict[name] = v
#
# CARN_SRnet.load_state_dict(new_state_dict)
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# CARN_SRnet = CARN_SRnet.to(device)
# shave = 20
#
# def save_image(tensor, filename):
#     tensor = tensor.cpu()
#     ndarr = tensor.mul(255).clamp(0, 255).byte().permute(1, 2, 0).numpy()
#     im = Image.fromarray(ndarr)
#     im.save(filename)
#
# def CARN_SR(net,device,scale,lr):
#     t1 = time.time()
# #     lr = torch.from_numpy(lr)
#     h, w = lr.size()[1:]
#     h_half, w_half = int(h / 2), int(w / 2)
#     h_chop, w_chop = h_half + shave, w_half + shave
#
#     lr_patch = torch.zeros((4, 3, h_chop, w_chop), dtype=torch.float)
#
#     lr_patch[0].copy_(lr[:, 0:h_chop, 0:w_chop])
#     lr_patch[1].copy_(lr[:, 0:h_chop, w - w_chop:w])
#     lr_patch[2].copy_(lr[:, h - h_chop:h, 0:w_chop])
#     lr_patch[3].copy_(lr[:, h - h_chop:h, w - w_chop:w])
#     lr_patch = lr_patch.to(device)
#
#     sr = net(lr_patch,scale).detach()
#
#     h, h_half, h_chop = h * scale, h_half * scale, h_chop * scale
#     w, w_half, w_chop = w * scale, w_half * scale, w_chop * scale
#
#     result = torch.zeros((3, h, w), dtype=torch.float).to(device)
#     result[:, 0:h_half, 0:w_half].copy_(sr[0, :, 0:h_half, 0:w_half])
#     result[:, 0:h_half, w_half:w].copy_(sr[1, :, 0:h_half, w_chop - w + w_half:w_chop])
#     result[:, h_half:h, 0:w_half].copy_(sr[2, :, h_chop - h + h_half:h_chop, 0:w_half])
#     result[:, h_half:h, w_half:w].copy_(sr[3, :, h_chop - h + h_half:h_chop, w_chop - w + w_half:w_chop])
#     sr = result
#     t2 = time.time()
#     print("Transformed:({}x{} -> {}x{}, {:.3f}s)"
#             .format(lr.shape[1], lr.shape[2], sr.shape[1], sr.shape[2], t2-t1))
# #     sr = sr.numpy()
#     return sr
#
# transform = transforms.Compose([
#             transforms.ToTensor()
#         ])
# img = Image.open("./carn/0000.png")
# img = img.convert("RGB")
# img = transform(img)
#
#
# result = CARN_SR(CARN_SRnet,device,2,img)
# save_image(result,"./carn/result.png")

class CARN_SR(object):
    def __init__(self, ):
        # module = importlib.import_module("model.carn_m")
        self.net = module.Net(multi_scale=True, group=4)

        state_dict = torch.load("C:\\Users\\rht\\PycharmProjects\\STGAN\\Flask\\CARN_pytorch\\checkpoint\\carn_m.pth")
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k
            new_state_dict[name] = v

        self.net.load_state_dict(new_state_dict)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.net = self.net.to(self.device)
        self.shave = 20
        self.scale = 2

    def inference(self,lr):
        # h, w = lr.shape()[:2]
        h, w = lr.shape[0], lr.shape[1]
        lr_tmp = np.zeros(shape=(3, h, w), dtype=np.float32)
        lr_tmp[0, :, :] = lr[:, :, 0]
        lr_tmp[1, :, :] = lr[:, :, 1]
        lr_tmp[2, :, :] = lr[:, :, 2]
        lr = lr_tmp
        lr = torch.from_numpy(lr)
        h_half, w_half = int(h / 2), int(w / 2)
        h_chop, w_chop = h_half + self.shave, w_half + self.shave

        lr_patch = torch.zeros((4, 3, h_chop, w_chop), dtype=torch.float)

        lr_patch[0].copy_(lr[:, 0:h_chop, 0:w_chop])
        lr_patch[1].copy_(lr[:, 0:h_chop, w - w_chop:w])
        lr_patch[2].copy_(lr[:, h - h_chop:h, 0:w_chop])
        lr_patch[3].copy_(lr[:, h - h_chop:h, w - w_chop:w])
        lr_patch = lr_patch.to(self.device)

        sr = self.net(lr_patch, self.scale).detach()

        h, h_half, h_chop = h * self.scale, h_half * self.scale, h_chop * self.scale
        w, w_half, w_chop = w * self.scale, w_half * self.scale, w_chop * self.scale

        result = torch.zeros((3, h, w), dtype=torch.float).to(self.device)
        result[:, 0:h_half, 0:w_half].copy_(sr[0, :, 0:h_half, 0:w_half])
        result[:, 0:h_half, w_half:w].copy_(sr[1, :, 0:h_half, w_chop - w + w_half:w_chop])
        result[:, h_half:h, 0:w_half].copy_(sr[2, :, h_chop - h + h_half:h_chop, 0:w_half])
        result[:, h_half:h, w_half:w].copy_(sr[3, :, h_chop - h + h_half:h_chop, w_chop - w + w_half:w_chop])
        sr = result
        # t2 = time.time()
        # print("Transformed:({}x{} -> {}x{}, {:.3f}s)"
        #       .format(lr.shape[1], lr.shape[2], sr.shape[1], sr.shape[2], t2 - t1))
        sr = sr.numpy()
        sr_tmp = np.zeros(shape=(h * self.scale, w * self.scale, 3), dtype=np.float32)
        sr_tmp[:, :, 0] = sr[0, :, :]
        sr_tmp[:, :, 1] = sr[1, :, :]
        sr_tmp[:, :, 2] = sr[2, :, :]
        sr = sr_tmp
        return sr





    





