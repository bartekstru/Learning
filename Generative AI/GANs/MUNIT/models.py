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
        self.num_features = num_features
        # dummy buffers, we actually do not care about these
        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_var', torch.ones(num_features))

    def forward(self, x):                                                                                                               # (B, C, W, H)
        assert self.weight is not None and self.bias is not None, "Affine parameters for Adaptive Instance Normalization not set."
        b, c = x.size(0), x.size(1)
        running_mean = self.running_mean.repeat(b)
        running_var = self.running_var.repeat(b)
        x = x.contiguous().view(1, b*c, *x.size()[2:])

        out = F.batch_norm(x, running_mean=running_mean, running_var=running_var, weight=self.weight,
                           bias=self.bias, training=True, momentum=self.momentum, eps=self.eps)

        return out.view(1, b*c, *x.size()[2:])


class LayerNorm(nn.Module):

    def __init__(self, num_features, affine, eps):
        super().__init__()
        self.num_features = num_features
        self.affine = affine
        self.eps = eps

        if self.affine:
            self.gamma = nn.Parameter(torch.Tensor(num_features).uniform_())
            self.beta = nn.Parameter(torch.zeros())

    def forward(self, x):
        shape = [1] + [1] * (x.dim() - 1)

        if x.size(0) == 1:
            mean = x.view(-1).mean().view(*shape)
            std = x.view(-1).std().view(*shape)
        else:
            mean = x.view(x.shape[0], -1).mean().view(*shape)
            std = x.view(x.shape[0], -1).std().view(*shape)

        x = (x - mean) / (std + self.eps)  # B, C, H, W

        if self.affine:
            shape = [1, -1] + [1] * (x.dim() - 2)  # 1, C, 1, 1
            x = x * self.gamma.view(*shape) + x.beta.view(*shape)
        return x


#################################################################################
# Basic Blocks
#################################################################################

class Conv2dBlock(nn.Module):

    def __init__(self, 
                 input_channels, 
                 output_channels, 
                 kernel_size, 
                 stride, 
                 padding=0, 
                 pad_type='zero', 
                 activation='none', 
                 norm='none'):
        
        super().__init__()

        # Add form of padding
        if pad_type == 'zero':
            self.pad = nn.ZeroPad2d(padding)
        elif pad_type == 'reflect':
            self.pad = nn.ReflectionPad2d(padding)
        else:
            assert 0, "Unsupported padding type {}".format(pad_type)

        # Add form of normalization
        norm_dim = output_channels
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
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'none':
            self.activation = None
        else:
            assert 0, "Unsupported activation function {}".format(activation)

        # Add convolution
        if norm == 'sn':
            self.conv = nn.utils.spectral_norm(
                nn.Conv2d(in_channels=input_channels, 
                          out_channels=output_channels,
                          kernel_size=kernel_size, 
                          stride=stride)
            )
        else:
            self.conv = nn.Conv2d(in_channels=input_channels, 
                                  out_channels=output_channels, 
                                  kernel_size=kernel_size, 
                                  stride=stride)

    def forward(self, x):
        x = self.conv(self.pad(x))
        if self.activation:
            x = self.activation(x)
        return x


class ResBlock(nn.Module):

    def __init__(self, 
                 channels, 
                 pad_type='zero', 
                 activation='relu', 
                 norm='in'):
        
        super().__init__()
        self.model = []

        model += [Conv2dBlock(input_channels=channels, 
                              output_channels=channels, 
                              kernel_size=3,
                              stride=1, 
                              padding=1, 
                              pad_type=pad_type, 
                              activation=activation, 
                              norm=norm)]
        
        model += [Conv2dBlock(input_channels=channels, 
                              output_channels=channels, 
                              kernel_size=3,
                              stride=1, 
                              padding=1, 
                              pad_type=pad_type, 
                              activation='none', 
                              norm=norm)]
        
        self.model = nn.Sequential(*model)

    def forward(self, x):
        residual = x
        out = self.model(x)
        out += residual
        return out


class LinearBlock(nn.Module):
    def __init__(self, 
                 input_dim, 
                 output_dim, 
                 activation='relu', 
                 norm='none'):
        
        super().__init__()

        # Add activation
        if activation == 'relu':
            self.activation = nn.ReLU()
        if activation == 'none':
            self.activation = None
        else:
            assert 0, "Unsupported activation function {}".format(activation)

        # Add normalization
        norm_dim = output_dim
        if norm == 'in':
            self.norm = nn.InstanceNorm1d(norm_dim)
        elif norm == 'none':
            self.norm = None
        else:
            assert 0, "Unsupported normalization: {}".format(norm)

        self.fc = nn.Linear(in_features=input_dim,
                            out_features=output_dim, bias=True)

    def forward(self, x):
        out = self.fc(x)
        if self.norm:
            out = self.norm(out)
        if self.activation:
            out = self.activation(out)
        return out

#################################################################################
# Sequential Blocks
#################################################################################


class ResBlocks(nn.Module):

    def __init__(self, 
                 num_blocks, 
                 channels, 
                 pad_type, 
                 activation, 
                 norm):
        
        super().__init__()

        self.model = []

        for _ in range(num_blocks):
            self.modules += [ResBlock(channels=channels, 
                                      pad_type=pad_type, 
                                      activation=activation, 
                                      norm=norm)]
            
        self.model = nn.Sequential(*self.model)

    def forward(self, x):
        return self.model(x)


class MLP(nn.Module):

    def __init__(self, 
                 input_dim, 
                 output_dim, 
                 hid_dim, 
                 num_blocks, 
                 norm='none', 
                 activation='relu'):
        
        super().__init__()

        self.model = []

        self.model += [LinearBlock(input_dim=input_dim,
                                   output_dim=hid_dim, 
                                   activation=activation, 
                                   norm=norm)]
        
        for _ in range(num_blocks - 2):
            self.model += [LinearBlock(input_dim=hid_dim,
                                       output_dim=hid_dim, 
                                       activation=activation, 
                                       norm=norm)]
            
        self.model += [LinearBlock(input_dim=hid_dim,
                                   output_dim=output_dim, 
                                   activation=activation, 
                                   norm=norm)]
        
        self.model = nn.Sequential(*self.model)

    def forward(self, x):
        return self.model(x.view(x.size(0), -1))


#################################################################################
# Generator
#################################################################################

class StyleEncoder(nn.Module):
    def __init__(self,
                 input_channels: int = 3,       # the channels in the input tensor
                 base_channels: int = 64,       # number of channels in the first convolutional layer
                 n_downsample: int = 4,         # number of down-sampling layers
                 s_dim: int = 8,                # output dimension ot the style tensor
                 norm: str = "none",            # type of normalization to apply to the layers
                 activ: str = "relu",           # activation function
                 pad_type: str = "reflect"):    # type of padding

        super.__init__()

        channels = base_channels
        n_deepen_layers = 2         # number of layers which increase the number of channels

        self.model = []
        self.model += [Conv2dBlock(input_channels=input_channels, 
                                   output_channels=channels, 
                                   kernel_size=7, stride=1, 
                                   padding=3, 
                                   norm=norm, 
                                   activation=activ, 
                                   pad_type=pad_type)]
        
        for _ in range(n_deepen_layers):
            self.model += [Conv2dBlock(input_channels=channels, 
                                       output_channels=channels*2, 
                                       kernel_size=4, 
                                       stride=2, 
                                       padding=1, 
                                       norm=norm, 
                                       activation=activ, 
                                       pad_type=pad_type)]
            channels *= 2

        for _ in range(n_downsample - 2):
            self.model += [Conv2dBlock(input_channels=channels, 
                                       output_channels=channels,
                                       kernel_size=4, 
                                       stride=2, 
                                       padding=1, 
                                       pad_type=pad_type, 
                                       activation=activ, 
                                       norm=norm)]
            
        self.model += [nn.AdaptiveAvgPool2d(1)]
        self.model += [nn.Conv2d(in_channels=channels,
                                 out_channels=s_dim, 
                                 kernel_size=1, 
                                 stride=1, 
                                 padding=0)]
        
        self.model = nn.Sequential(*self.model)

    def forward(self, x):
        return self.model(x)


class ContentEncoder(nn.Module):
    def __init__(self,
             input_channels: int = 3,       # the channels in the input tensor
             base_channels: int = 64,       # number of channels in the first convolutional layer
             n_downsample: int = 2,         # number of down-sampling layers
             norm: str = "in",              # type of normalization to apply to the layers
             activ: str = "relu",           # activation function
             pad_type: str = "reflect",     # type of padding
             n_res_blocks = 4):             # number of residual blocks

        super.__init__()

        channels = base_channels

        self.model = []
        self.model += [Conv2dBlock(input_channels=input_channels, 
                                   output_channels=base_channels,
                                   kernel_size=7, 
                                   stride=1, 
                                   padding=3, 
                                   norm=norm, 
                                   activation=activ, 
                                   pad_type=pad_type)]
        
        for _ in range(n_downsample):
            self.model += [Conv2dBlock(input_channels=base_channels, 
                                       output_channels=base_channels*2,
                                       kernel_size=4, 
                                       stride=2, 
                                       padding=1, 
                                       pad_type=pad_type, 
                                       norm=norm, 
                                       activation=activ)]
            channels *= 2

        self.model += [ResBlocks(num_blocks=n_res_blocks, 
                                 channels=channels,
                                 pad_type=pad_type, 
                                 activation=activ, 
                                 norm=norm)]
        
        self.output_channels = channels
        self.model = nn.Sequential(*self.model)

    def forward(self, x):
        return self.model(x)


class Decoder(nn.Module):

    def __init__(self,
                 input_channels: int = 256, 
                 n_upsample: int = 2, 
                 n_res_blocks: int = 4, 
                 output_channels: int = 3, 
                 pad_type: str = "reflect", 
                 activ: str = "relu", 
                 res_norm: str = "adain"):
        
        super().__init__()

        self.model = []

        channels = input_channels

        # AdaIN residual blocks
        self.model += [ResBlocks(num_blocks=n_res_blocks, 
                                 channels=input_channels,
                                 pad_type=pad_type, 
                                 activation=activ, 
                                 norm=res_norm)]

        # upsampling blocks
        for _ in range(n_upsample):
            self.model += [
                nn.Upsample(scale_factor=2),
                Conv2dBlock(input_channels=input_channels, 
                            output_channels=channels // 2, 
                            kernel_size=5,
                            stride=1,
                            padding=2,
                            pad_type=pad_type,
                            activation=activ,
                            norm='ln')
            ]
            channels //= 2

        # produce three dimensional images
        self.model += [Conv2dBlock(input_channels=channels, 
                                   output_channels=output_channels, 
                                   kernel_size=7,
                                   stride=1, 
                                   padding=3, 
                                   pad_type=pad_type, 
                                   norm='none', 
                                   activation='tanh')]
        
        self.model = nn.Sequential(*self.model)

    def forward(self, x):
        return self.model(x)


class AdaINGenerator(nn.Module):

    def __init__(self, 
                 base_channels: int = 64, 
                 n_c_downsample: int = 2, 
                 n_s_downsample: int = 4, 
                 n_res_blocks: int = 4, 
                 s_dim: int = 8, 
                 activ: str = "relu", 
                 pad_type: str = "reflect",
                 mlp_dim: int = 256,
                 mlp_layers: int =3 
                 ):
        super().__init__()
        
        self.StyleEncoder = StyleEncoder(3, base_channels, n_s_downsample, s_dim, "none", activ, pad_type)
        self.ContentEncoder = ContentEncoder(3, 64, n_c_downsample, 'in', activ, pad_type, n_res_blocks)
        self.Decoder = Decoder(self.ContentEncoder.output_channels, n_c_downsample, n_res_blocks, 3, pad_type, activ)
        self.MLP = MLP(s_dim, self.get_num_adain_params(), mlp_dim, mlp_layers, "none", "relu")


    def forward(self, x):
        style, content = self.encode(x)
        return self.decode(content, style)
        

    def encode(self, x):
        return self.StyleEncoder(x), self.ContentEncoder(x)

    def decode(self, content, style):
        adain_params = self.MLP(style)
        self.assign_adain_params(adain_params, self.Decoder)
        images = self.Decoder(content)
        return images

    def get_num_adain_params(self):
        decoder = self.Decoder
        adain_params = 0
        for m in decoder.modules():
            if m.__class__.__name__ == "AdaptiveInstanceNorm2d":
                adain_params += 2*m.num_features
        return adain_params
    
    def assign_adain_params(self, adain_params, model):
        for m in model.modules():
            if m.__class__.__name__ == "AdaptiveInstanceNorm2d":
                shift = adain_params[:, :m.num_features]
                scale = adain_params[:, :m.num_features:2*m.num_features]
                m.weight = shift.contiguous().view(-1)
                m.bias = scale.contiguous().view(-1)
                if adain_params.size() > 2*m.num_features:
                    adain_params = adain_params[:, 2*m.num_features:]

#################################################################################
# Discriminator
#################################################################################


class Discriminator(nn.Module):
    pass



class MUNIT(nn.Module):
    def __init__(
        self,
        gen_channels: int = 64,
        n_c_downsample: int = 2,
        n_s_downsample: int = 4,
        n_res_blocks: int = 4,
        s_dim: int = 8,
        h_dim: int = 256,
        dis_channels: int = 64,
        n_layers: int = 3,
        n_discriminators: int = 3,
    ):
        super().__init__()

        self.gen_a = AdaINGenerator()

        self.gen_a = AdaINGenerator(
            base_channels=gen_channels, n_c_downsample=n_c_downsample, n_s_downsample=n_s_downsample, n_res_blocks=n_res_blocks, s_dim=s_dim, h_dim=h_dim,
        )
        self.gen_b = Generator(
            base_channels=gen_channels, n_c_downsample=n_c_downsample, n_s_downsample=n_s_downsample, n_res_blocks=n_res_blocks, s_dim=s_dim, h_dim=h_dim,
        )
        self.dis_a = Discriminator(
            base_channels=dis_channels, n_layers=n_layers, n_discriminators=n_discriminators,
        )
        self.dis_b = Discriminator(
            base_channels=dis_channels, n_layers=n_layers, n_discriminators=n_discriminators,
        )
        self.s_dim = s_dim
        self.loss = GinormousCompositeLoss

    def forward(self, x_a, x_b):
        s_a = torch.randn(x_a.size(0), self.s_dim, 1, 1, device=x_a.device).to(x_a.dtype)
        s_b = torch.randn(x_b.size(0), self.s_dim, 1, 1, device=x_b.device).to(x_b.dtype)

        # Encode real x and compute image reconstruction loss
        x_a_loss, c_a, s_a_fake = self.loss.image_recon_loss(x_a, self.gen_a)
        x_b_loss, c_b, s_b_fake = self.loss.image_recon_loss(x_b, self.gen_b)

        # Decode real (c, s) and compute latent reconstruction loss
        c_b_loss, s_a_loss, x_ba = self.loss.latent_recon_loss(c_b, s_a, self.gen_a)
        c_a_loss, s_b_loss, x_ab = self.loss.latent_recon_loss(c_a, s_b, self.gen_b)

        # Compute adversarial losses
        gen_a_adv_loss = self.loss.adversarial_loss(x_ba, self.dis_a, True)
        gen_b_adv_loss = self.loss.adversarial_loss(x_ab, self.dis_b, True)

        # Sum up losses for gen
        gen_loss = (
            10 * x_a_loss + c_b_loss + s_a_loss + gen_a_adv_loss + \
            10 * x_b_loss + c_a_loss + s_b_loss + gen_b_adv_loss
        )

        # Sum up losses for dis
        dis_loss = (
            self.loss.adversarial_loss(x_ba.detach(), self.dis_a, False) + \
            self.loss.adversarial_loss(x_a.detach(), self.dis_a, True) + \
            self.loss.adversarial_loss(x_ab.detach(), self.dis_b, False) + \
            self.loss.adversarial_loss(x_b.detach(), self.dis_b, True)
        )

        return gen_loss, dis_loss, x_ab, x_ba
    
