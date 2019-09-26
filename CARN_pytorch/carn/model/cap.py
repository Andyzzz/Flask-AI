import torch
import importlib
import time

module = importlib.import_module("model.carn_m")
CARN_SRnet = module.Net(multi_scale=True,group=4)
# print(json.dumps(vars(cfg), indent=4, sort_keys=True))

state_dict = torch.load(./checkpoint/carn_m.pth)
new_state_dict = OrderedDict()
for k, v in state_dict.items():
    name = k
    new_state_dict[name] = v

CARN_SRnet.load_state_dict(new_state_dict)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CARN_SRnet = CARN_SRnet.to(device)
shave = 20

def CARN_SR(net,device,scale,lr):
    t1 = time.time()
    lr = torch.from_numpy(lr)
    h, w = lr.size()[1:]
    h_half, w_half = int(h / 2), int(w / 2)
    h_chop, w_chop = h_half + shave, w_half + shave

    lr_patch = torch.zeros((4, 3, h_chop, w_chop), dtype=torch.float)

    lr_patch[0].copy_(lr[:, 0:h_chop, 0:w_chop])
    lr_patch[1].copy_(lr[:, 0:h_chop, w - w_chop:w])
    lr_patch[2].copy_(lr[:, h - h_chop:h, 0:w_chop])
    lr_patch[3].copy_(lr[:, h - h_chop:h, w - w_chop:w])
    lr_patch = lr_patch.to(device)

    sr = net(lr_patch,scale).detach()

    h, h_half, h_chop = h * scale, h_half * scale, h_chop * scale
    w, w_half, w_chop = w * scale, w_half * scale, w_chop * scale

    result = torch.zeros((3, h, w), dtype=torch.float).to(device)
    result[:, 0:h_half, 0:w_half].copy_(sr[0, :, 0:h_half, 0:w_half])
    result[:, 0:h_half, w_half:w].copy_(sr[1, :, 0:h_half, w_chop - w + w_half:w_chop])
    result[:, h_half:h, 0:w_half].copy_(sr[2, :, h_chop - h + h_half:h_chop, 0:w_half])
    result[:, h_half:h, w_half:w].copy_(sr[3, :, h_chop - h + h_half:h_chop, w_chop - w + w_half:w_chop])
    sr = result
    t2 = time.time()
    sr = sr.numpy()
    return sr





