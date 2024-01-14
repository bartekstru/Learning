import torch
import torch.nn as nn
import torch.nn.functional as F

##################################################################################
# Normalization layers
##################################################################################
class AdaptiveInstanceNorm2d(nn.Module):

    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        super().__init__()
        self.eps = eps
        self.momentum = momentum
        # affine parameters will come from MLP and the style code
        self.weight = None
        self.bias = None

        # dummy buffers, we actually do not care about these
        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_var', torch.ones(num_features))
        
    
    def forward(self, x: nn.Tens):
        assert self.weight is not None and self.bias is not None, "Affine parameters for Adaptive Instance Normalization not set."
        b,c = x.size(0), x.size(1)
        running_mean = self.running_mean.repeat(b)
        running_var = self.running_var.repeat(b)
        x = x.contiguous().view(1, b*c, *x.size()[2:])

        out = F.batch_norm(x, running_mean=running_mean, running_var=running_var, weight=self.weight, bias=self.bias, training=True, momentum=self.momentum, eps=self.eps)

        return out.view(1, b*c, *x.size()[2:])
    
class LayerNorm(nn.Module):
    pass

#################################################################################
# Basic Blocks
#################################################################################

class Conv2dBlock(nn.Module):
    
    def __init__(self, input_dim, output_dim, kernel, stride, padding=0, pad_type='zero', activation='none', norm='none'):
        super().__init__()

        # Add form of padding
        if pad_type == 'zeros':
            self.pad = nn.ZeroPad2d(padding)
        elif pad_type == 'reflect':
            self.pad = nn.ReflectionPad2d(padding)
        else:
            assert 0, "Unsupported padding type {}".format(pad_type)

        # Add form of normalization
        norm_dim = output_dim
        if norm == 'in':
            self.norm = nn.InstanceNorm2d(norm_dim)
        elif norm == 'adain':
            self.norm = AdaptiveInstanceNorm2d(norm_dim)
        elif norm == 'sn' or norm == 'none':
            self.norm = None
        elif norm == 'ln':
            self.norm = LayerNorm(norm_dim)
        else:
            assert 0, "Unsupported normalization: {}".format(norm)
        
        # Add activation
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'none':
            self.activation = None
        else:
            assert 0, "Unsupported activation function {}".format(activation)

        # Add convolution
        if norm == 'sn':
            self.conv = nn.utils.spectral_norm(
                nn.Conv2d(in_channels=input_dim, out_channels=output_dim, kernel_size=kernel, stride=stride)
            )
        else:
            self.conv = nn.Conv2d(in_channels=input_dim, out_channels=output_dim, kernel_size=kernel, stride=stride)
        

    def forward(self, x):
        x = self.conv(self.pad(x))
        if self.activation:
            x = self.activation(x)
        return x

class ResBlock(nn.Module):

    def __init__(self, dim, pad_type='zeros', activation='relu', norm='in'):
        super().__init__()
        self.model = []
        model += [Conv2dBlock(input_dim=dim, output_dim=dim, kernel=3, stride=1, padding=1, pad_type=pad_type, activation=activation, norm=norm)]
        model += [Conv2dBlock(input_dim=dim, output_dim=dim, kernel=3, stride=1, padding=1, pad_type=pad_type, activation='none', norm=norm)]
        self.model = nn.Sequential(*model)
    
    def forward(self, x):
        residual = x
        out = self.model(x)
        out += residual
        return out

# class LinearBlock(nn.Module):
#     def __init__(self, input_dim, output_dim, activation='relu', norm='none'):
#         super().__init__()

#         # Add activation
#         if activation == 'relu':
#             self.activation = nn.ReLU()
#         if activation == 'none':
#             self.activation = None
#         else:
#             assert 0, "Unsupported activation function {}".format(activation)

#         # Add normalization
#         norm_dim = output_dim
#         if norm == 'in':
#             self.norm = nn.InstanceNorm1d(norm_dim)
#         elif norm == 'none':
#             self.norm = None
#         else:
#             assert 0, "Unsupported normalization: {}".format(norm)

#         self.fc = nn.Linear(in_features=input_dim, out_features=output_dim, bias=True)

#     def forward(self, x):
#         out = self.fc(x)
#         if self.norm:
#             out = self.norm(out)
#         if self.activation:
#             out = self.activation(out)
#         return out

#################################################################################
# Sequential Blocks
#################################################################################

class ResBlocks(nn.Module):

    def __init__(self, num_blocks, dim, pad_type, activation, norm):
        super().__init__()
        self.model = []
        for i in range(num_blocks):
            self.modules += [ResBlock(dim=dim, pad_type=pad_type, activation=activation, norm=norm)]
        self.model = nn.Sequential(*self.model)

    def forward(self, x):
        return self.model(x)

#################################################################################
# Generator
#################################################################################

class StyleEncoder(nn.Module):
    
    def __init__(self, n_downsample, in_dim, hid_dim, output_dim, norm, activation, pad_type):
        super.__init__()
        self.model = []
        self.model += [Conv2dBlock(input_dim=in_dim, output_dim=hid_dim, kernel=7, stride=1, padding=3, norm=norm, activation=activation, pad_type=pad_type)]
        for i in range(2):
            self.model += [Conv2dBlock(hid_dim, output_dim=hid_dim*2, kernel=4, stride=2, padding=1, norm=norm, activation=activation, pad_type=pad_type)]
            hid_dim *= 2
        for i in range(n_downsample - 2):
            self.model += [Conv2dBlock(input_dim=hid_dim, output_dim=hid_dim, kernel=4, stride=2, padding=1, pad_type=pad_type, activation=activation, norm=norm)]
        self.model += [nn.AdaptiveAvgPool2d(1)]
        self.model += [nn.Conv2d(in_channels=hid_dim, out_channels=output_dim, kernel_size=1, stride=1, padding=0)]
        self.model = nn.Sequential(*self.model)

    def forward(self, x):
        return self.model(x)

class ContentEncoder(nn.Module):
    
    def __init__(self, n_downsample, n_res_blocks, in_dim, hid_dim, norm, activation, pad_type):
        super.__init__()
        self.model = []
        self.model += [Conv2dBlock(input_dim=in_dim, output_dim=hid_dim, kernel=7, stride=1, padding=3, norm=norm, activation=activation, pad_type=pad_type)]
        for i in range(n_downsample):
            self.model += [Conv2dBlock(input_dim=hid_dim, output_dim=hid_dim*2, kernel=4, stride=2, padding=1, pad_type=pad_type, norm=norm, activation=activation)]
            hid_dim *= 2
        self.model += [ResBlocks(num_blocks=n_res_blocks, dim=hid_dim, pad_type=pad_type, activation=activation, norm=norm)]
        self.model = nn.Sequential(*self.model)
    
    def forward(self, x):
        return self.model(x)

class Decoder(nn.Module):
    
    def __init__(self, n_upsample, n_res_blocks, hid_dim, output_dim, pad_type, activation, res_norm="adain"):
        super().__init__()
        self.model = []
        # AdaIN residual blocks
        self.model += [ResBlocks(num_blocks=n_res_blocks, dim=hid_dim, pad_type=pad_type, activation=activation, norm=res_norm)]

        # upsampling blocks
        for i in range(n_upsample):
            self.model += [
                nn.UpsamplingNearest2d(scale_factor=2),
                Conv2dBlock(input_dim=hid_dim, output_dim=hid_dim//2, kernel=5, stride=1, padding=2, pad_type=pad_type, activation=activation, norm='ln')
            ]
            hid_dim //= 2
        self.model += [Conv2dBlock(input_dim=hid_dim, output_dim=output_dim, kernel=7, stride=1, padding=3, pad_type=pad_type, norm='none', activation='tanh')]
        self.model = nn.Sequential(*self.model)

    def forward(self, x):
        return self.model(x)



class AdaINGenerator(nn.Module):
    pass        

#################################################################################
# Discriminator
#################################################################################

class Discriminator(nn.Module):
    pass