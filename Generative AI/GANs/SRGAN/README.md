# Super-Resolution GAN (SRGAN) Documentation

Welcome to the SRGAN project! This repository explores the Super-Resolution Generative Adversarial Network (SRGAN) introduced by Ledig et al. in their paper [Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network](https://arxiv.org/abs/1609.04802) (2017). The primary goal here is to enhance the resolution of images by 4x.

## Prerequisites

Before diving into this project, it is assumed that you have some familiarity with the following concepts:
- Residual blocks, as discussed in [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385) (He et al. 2015)
- Perceptual loss, detailed in [Perceptual Losses for Real-Time Style Transfer and Super-Resolution](https://arxiv.org/abs/1603.08155) (Johnson et al. 2016)
- VGG architecture, covered in [Very Deep Convolutional Networks for Large-Scale Image Recognition](https://arxiv.org/abs/1409.1556) (Simonyan et al. 2015)

## Project Overview

The SRGAN project aims to implement and train a Generative Adversarial Network for super-resolution tasks using the CIFAR dataset. The core architecture is based on the SRResNet, a super-resolution residual network.

## Key Components

### Parametric ReLU (PReLU)

To address limitations in the traditional ReLU activation function, the project employs the Parametric ReLU (PReLU), as introduced by He et al. in [Delving Deep into Rectifiers: Surpassing Human-Level Performance on ImageNet Classification](https://arxiv.org/abs/1502.01852). The PReLU is conveniently implemented in PyTorch as [torch.nn.PReLU](https://pytorch.org/docs/stable/generated/torch.nn.PReLU.html).

### Residual Blocks

Residual blocks, a crucial element in many computer vision models, are extensively used in SRGAN. If you're not familiar with residual blocks, refer to the documentation [here](https://paperswithcode.com/method/residual-block).

### PixelShuffle

The PixelShuffle operation, proposed in [Real-Time Single Image and Video Super-Resolution Using an Efficient Sub-Pixel Convolutional Neural Network](https://arxiv.org/abs/1609.05158) (Shi et al. 2016), offers an alternative method for upsampling images. While the underlying details can be complex, the operation is conveniently implemented in PyTorch as `torch.nn.PixelShuffle`.

> ![Efficient Sub-pixel CNN](https://github.com/https-deeplearning-ai/GANs-Public/blob/master/SRGAN-PixelShuffle.png?raw=true) *Efficient sub-pixel CNN, taken from Figure 1 of [Real-Time Single Image and Video Super-Resolution Using an Efficient Sub-Pixel Convolutional Neural Network](https://arxiv.org/abs/1609.05158) (Shi et al. 2016). The PixelShuffle operation (also known as sub-pixel convolution) is shown as the last step on the right.*

### Generator (SRResNet)

The SRResNet, serving as the generator, is a relatively simple network composed of convolutional layers, residual blocks, and pixel shuffling layers.

> ![SRGAN Generator](https://github.com/https-deeplearning-ai/GANs-Public/blob/master/SRGAN-Generator.png?raw=true) *SRGAN Generator, taken from Figure 4 of [Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network](https://arxiv.org/abs/1609.04802) (Ledig et al. 2017).*

### Discriminator

The discriminator architecture is also relatively straightforward, just one big sequential model - see the diagram below for reference!

> ![SRGAN Generator](https://github.com/https-deeplearning-ai/GANs-Public/blob/master/SRGAN-Discriminator.png?raw=true) *SRGAN Discriminator, taken from Figure 4 of [Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network](https://arxiv.org/abs/1609.04802) (Ledig et al. 2017).*

## Loss Functions

The project formulates the perceptual loss as a weighted sum of content loss (based on the VGG19 network) and adversarial loss. The content loss mitigates blurriness issues associated with traditional Mean Squared Error (MSE) loss, incorporating an additional MSE loss term on VGG19 feature maps.

**Content Loss**
\[ \mathcal{L}_{VGG} = \left|\left|\phi_{5,4}(I^{\text{HR}}) - \phi_{5,4}(G(I^{\text{LR}}))\right|\right|_2^2 \]

**Adversarial Loss**
\[ \mathcal{L}_{ADV} = \sum_{n=1}^N -\log D(G(I^{\text{LR}})) \]

Feel free to explore the code and experiment with the SRGAN architecture. If you encounter any challenges, don't hesitate to dive into the documentation and resources linked throughout the project. Happy coding!