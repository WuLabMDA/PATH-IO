import torch
import torch.nn as nn
from monai.networks.nets import EfficientNet


# Example regularize_path_weights function for EfficientNet
def regularize_path_weights(model, reg_type=None):
    l1_reg = None   

    for name, param in model.named_parameters():
        if 'weight' in name:
            if l1_reg is None:
                l1_reg = torch.sum(torch.abs(param))
            else:
                l1_reg = l1_reg + torch.sum(torch.abs(param))

    return l1_reg

efficientnet_params = {
    # model_name: (width_mult, depth_mult, image_size, dropout_rate, dropconnect_rate)
    "efficientnet-b0": (1.0, 1.0, 200, 0.2, 0.2),
    "efficientnet-b1": (1.0, 1.1, 240, 0.2, 0.2),
    "efficientnet-b2": (1.1, 1.2, 260, 0.3, 0.2),
    "efficientnet-b3": (1.2, 1.4, 300, 0.3, 0.2),
    "efficientnet-b4": (1.4, 1.8, 380, 0.4, 0.2),
    "efficientnet-b5": (1.6, 2.2, 456, 0.4, 0.2),
    "efficientnet-b6": (1.8, 2.6, 528, 0.5, 0.2),
    "efficientnet-b7": (2.0, 3.1, 600, 0.5, 0.2),
    "efficientnet-b8": (2.2, 3.6, 672, 0.5, 0.2),
    "efficientnet-l2": (4.3, 5.3, 800, 0.5, 0.2),
    }
blocks_args_str = [
    "r1_k7_s11_e1_i32_o16_se0.25",
    "r2_k1_s22_e6_i16_o24_se0.25",
    "r2_k1_s22_e6_i24_o40_se0.25",
    "r3_k1_s22_e6_i40_o80_se0.25",
    "r3_k1_s11_e6_i80_o112_se0.25",
    "r4_k1_s22_e6_i112_o192_se0.25",
    "r1_k1_s11_e6_i192_o320_se0.25",
    ]
weight_coeff, depth_coeff, image_size, dropout_rate, dropconnect_rate = efficientnet_params['efficientnet-b0']
model = EfficientNet(
    blocks_args_str  = blocks_args_str,
    spatial_dims=2,
    in_channels=3,
    num_classes=1,
    width_coefficient=weight_coeff,
    depth_coefficient=depth_coeff,
    dropout_rate=dropout_rate,
    image_size=224,
    drop_connect_rate=dropconnect_rate,
)
# class CustomEfficientNet(nn.Module):
#     def __init__(self, num_classes=1):  # Adjust num_init_features to match the output shape of EfficientNet
#         super(CustomEfficientNet, self).__init__()
#         self.relu = nn.ReLU()
#         self.efficientnet = model
#         self.relu2 = nn.ReLU()
# #         self.linear = nn.Linear(64, num_classes)  # Adjust the input size and number of classes as needed

#     def forward(self, x):
#         x = self.relu(x)
#         x = self.efficientnet(x)
#         x = self.relu2(x)
# #         x = torch.flatten(x, 1)  # Flatten the tensor before passing it to the Linear layer
# #         x = self.linear(x)
# #         x = self.relu(x)
# #         x = self.linear(x)

#         return x

# Custom network that adds a ReLU layer on top of EfficientNet
# class CustomEfficientNet(nn.Module):
#     def __init__(self, num_classes =1):
#         super(CustomEfficientNet, self).__init__()
#         self.efficientnet = model
#         self.relu = nn.ReLU(inplace=True)
#         self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
#         self.linear = nn.Linear(1000, num_classes)  # Adjust the input size (1000) and number of classes as needed

#     def forward(self, x):
#         x = self.efficientnet(x)
#         print(x.shape)
#         x = self.avgpool(x)
#         x = x.view(x.size(0), -1)
#         x = self.relu(x)
#         x = self.linear(x)
#         return x

# Example usage
# num_classes = 1  # Specify the number of output classes
# model = CustomEfficientNet()
