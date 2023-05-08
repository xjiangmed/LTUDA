from networks.unet import UNet
from networks.unet_proto import UNet_proto

def net_factory(net_type="unet"):
    if net_type == "unet":
        net = UNet(input_channel=1,num_class=4).cuda()
    elif net_type == "unet_proto":
        net = UNet_proto(input_channel=1,num_class=4, num_prototype=5).cuda()
    else:
        net = None 
    return net