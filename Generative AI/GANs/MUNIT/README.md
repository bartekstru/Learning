## Multi-modal Unsupervised Image-to-Image Translation (MUNIT) Overview

This document provides an overview of the MUNIT framework, a technique for unsupervised image-to-image translation introduced by Huang et al. in their paper, ["Multimodal Unsupervised Image-to-Image Translation"](https://arxiv.org/abs/1804.04732) (2018). Please note that this content is optional and delves into more advanced concepts.

## Prerequisites

It is assumed that readers are already familiar with Layer Normalization, as discussed in the paper by Ba et al. (2016) [Layer Normalization](https://arxiv.org/abs/1607.06450).

## Background

MUNIT extends the idea of a shared latent space from the UNIT framework. However, in MUNIT, only the content latent space is shared, while the style latent spaces are unique to each domain.

## Overview

Let's begin with a brief recap of the UNIT framework before delving into the specifics of MUNIT.

### UNIT Framework

[Unsupervised Image-to-Image Translation Networks](https://arxiv.org/abs/1703.00848) (Liu et al. 2018) introduced UNIT, a method for image translation that assumes images from different domains share a latent distribution. The process involves mapping images from domains $\mathcal{A}$ and $\mathcal{B}$ to a shared latent space $\mathcal{Z}$ using encoders $E_a$ and $E_b$. Generators $G_a$ and $G_b$ then produce synthetic images for their respective domains. Discriminators $D_a$ and $D_b$ evaluate the authenticity of the generated and real images.

### MUNIT Framework

In MUNIT, a pair of corresponding images $(x_a, x_b)$ from domains $\mathcal{A}$ and $\mathcal{B}$ can be generated using content and style vectors. The content vector $c$ is shared, while style vectors $s_a$ and $s_b$ are domain-specific. Decoders $F_a$ and $F_b$ synthesize images based on these vectors.

During training, inverters $E_a^c$, $E_a^s$, $E_b^c$, and $E_b^s$ are assumed to exist. These inverters extract content and style information, facilitating the translation of images between domains by mixing content and style vectors.

![Same- and cross-domain interaction of encoders and decoders](https://github.com/https-deeplearning-ai/GANs-Public/blob/master/MUNIT-Domains.png?raw=true)

*Figure 2: Model overview from [Multimodal Unsupervised Image-to-Image Translation](https://arxiv.org/abs/1804.04732) (Huang et al. 2018). Red and blue arrows indicate encoders-decoder pairs within the same domain. Left: same domain image reconstruction. Right: cross-domain latent (content and style) vector reconstruction.*

## Subcomponents: Layers and Blocks

MUNIT incorporates several key subcomponents essential to its model architecture. Implementation of these smaller components is recommended for a smoother overall integration.

### Adaptive Instance Normalization (AdaIN)

The AdaIN layer, previously covered in StyleGAN and resembling its counterpart class-conditional batch normalization in the BigGAN components notebook, undergoes enhancement by the authors. This involves the use of a multi-layer perceptron (MLP) to augment linear layers responsible for scale and shift. Refer to the figure in **Submodules** and additional notes in **Submodules: Decoder** for deeper insights.

### Layer Normalization

Layer normalization is employed in the upsampling layers of MUNIT's decoder. Unlike batch normalization, it normalizes across channels per minibatch example, as proposed in [Layer Normalization](https://arxiv.org/abs/1607.06450) (Ba et al. 2016). Noteworthy is its prevalence in NLP and relative rarity in computer vision. Due to specific constraints in MUNIT, batch normalization is impractical, and instance normalization's normalization of statistics to a standard Gaussian at each position is undesirable. In PyTorch, this normalization is implemented as [nn.LayerNorm](https://pytorch.org/docs/stable/generated/torch.nn.LayerNorm.html), requiring precomputed spatial size for initialization, tailored for 1D, 2D, and 3D inputs.

### Residual Block

The residual block, a familiar concept by this point, is implemented to support both adaptive and non-adaptive instance normalization layers. This flexibility accommodates the varied usage of these layers throughout the model.

## Submodules: Encoders and Decoder

With the foundational building blocks in place, let's delve into the content encoder, style encoder, and decoder, pivotal components in the generator.

![Same- and cross-domain interaction of encoders and decoders](https://github.com/https-deeplearning-ai/GANs-Public/blob/master/MUNIT-Generator.png?raw=true)

*Figure 3 from [Multimodal Unsupervised Image-to-Image Translation](https://arxiv.org/abs/1804.04732) (Huang et al. 2018) depicts the generator architecture. Content encoder generates a downsampled representation, style encoder produces a style code, and decoder synthesizes an image from content and style codes.*

*Note: The official implementation integrates a multi-layer perceptron (MLP) to adjust scale and shift parameters in instance normalization layers. To align with previous definitions of* `AdaIN`, *your implementation will apply the MLP within* `AdaptiveInstanceNorm2d`.

### Content Encoder

The content encoder, following the pattern of other encoders, downsamples the input image and processes it through residual blocks to create a condensed representation.

### Style Encoder

Similar to the content encoder but distinct in its use of global pooling and fully-connected layers, the style encoder distills the input image to its style vector. Notably, no normalization layers are applied, as they could erase the feature statistics encoding style. This style code is then passed to the decoder for image synthesis.

### Decoder

As a crucial part of the encoder-decoder framework, the decoder synthesizes images using latent information from both content and style encodings. In this context, the content encoder and decoder form the backbone, with style information injected into residual blocks through `AdaIN` layers.

*Note: The official implementation incorporates an MLP to process style codes and assigns the resulting values to scale and shift parameters in instance normalization layers. To conform with the prior definitions of `AdaIN` in this course, your implementation will apply the MLP within `AdaptiveInstanceNorm2d`.*

Let's explore the implementation!

## Modules: Generator, Discriminator, and Loss

Now, you're prepared to implement the MUNIT generator and discriminator, along with the composite loss function that ties everything together during training.

### Generator

The generator consists of the two encoders and one decoder discussed in the preceding sections. Let's encapsulate these components within a `Generator` module.

### Discriminator

The discriminator, mirroring the one used in Pix2PixHD, comprises several PatchGAN discriminators operating at different scales. For a detailed understanding of their collaborative functioning, refer to the Pix2PixHD optional notebook. The discriminator is trained with the least squares objective.

### MUNIT Loss Overview

In MUNIT, various components interact to achieve the model's objectives. The notation used includes image domains, encoders, decoders, generators, and discriminators. Let's briefly outline these components:

- **Image Domains:**
  - $a \in \mathcal{A}$
  - $b \in \mathcal{B}$

- **Encoders ($E$):**
  - $E_a^c: a \mapsto c_a$
  - $E_a^s: a \mapsto s_a$
  - $E_b^c: b \mapsto c_b$
  - $E_b^s: b \mapsto s_b$

- **Decoders ($F$):**
  - $F_a: (c_*, s_a) \mapsto a'$
  - $F_b: (c_*, s_b) \mapsto b'$

- **Generators ($G$):**
  - $G_a(a, b) = F_a(E_b^c(b), E_a^s(a))$
  - $G_b(b, a) = F_b(E_a^c(a), E_b^s(b))$

- **Discriminators ($D$):**
  - $D_a: a' \mapsto p \in \mathbb{R}$
  - $D_b: b' \mapsto p \in \mathbb{R}$

Now, let's delve into the loss components:

**Image Reconstruction Loss:**
- For domain $\mathcal{A}$:
  $\mathcal{L}_{\text{recon}}^a = \mathbb{E}_{a\sim p(a)}\left|\left|F_a(E_a^c(a), E_a^s(a)) - a\right|\right|_1$
- For domain $\mathcal{B}$:
  $\mathcal{L}_{\text{recon}}^b = \mathbb{E}_{b\sim p(b)}\left|\left|F_b(E_b^c(b), E_b^s(b)) - b\right|\right|_1$

**Latent Reconstruction Loss:**
- For domain $\mathcal{A}$:
  $\mathcal{L}_{\text{recon}}^{c_b} = \mathbb{E}_{c_b\sim p(c_b),s_a\sim q(s_a)}\left|\left|E_a^c(F_a(c_b, s_a)) - c_a\right|\right|_1$
  $\mathcal{L}_{\text{recon}}^{s_a} = \mathbb{E}_{c_b\sim p(c_b),s_a\sim q(s_a)}\left|\left|E_a^s(F_a(c_b, s_a)) - s_b\right|\right|_1$
- For domain $\mathcal{B}$:
  $\mathcal{L}_{\text{recon}}^{c_a} = \mathbb{E}_{c_a\sim p(c_a),s_b\sim q(s_b)}\left|\left|E_b^c(F_b(c_a, s_b)) - c_b\right|\right|_1$
  $\mathcal{L}_{\text{recon}}^{s_b} = \mathbb{E}_{c_a\sim p(c_a),s_b\sim q(s_b)}\left|\left|E_b^s(F_b(c_a, s_b)) - s_a\right|\right|_1$

**Adversarial Loss:**
- For domain $\mathcal{A}$:
  $\mathcal{L}_{\text{GAN}}^a = \mathbb{E}_{c_b\sim p(c_b),s_a\sim q(s_a)}\left[(1 - D_a(G_a(c_b, s_a)))^2\right] + \mathbb{E}_{a\sim p(a)}\left[D_a(a)^2\right]$
- For domain $\mathcal{B}$:
  $\mathcal{L}_{\text{GAN}}^b = \mathbb{E}_{c_a\sim p(c_a),s_b\sim q(s_b)}\left[(1 - D_b(G_b(c_a, s_b)))^2\right] + \mathbb{E}_{b\sim p(b)}\left[D_b(b)^2\right]$

**Total Loss:**
$\mathcal{L}(E_a, E_b, F_a, F_b, D_a, D_b) = \mathcal{L}_{\text{GAN}}^a + \mathcal{L}_{\text{GAN}}^b + \lambda_x(\mathcal{L}_{\text{recon}}^a + \mathcal{L}_{\text{recon}}^b) + \lambda_c(\mathcal{L}_{\text{recon}}^{c_a} + \mathcal{L}_{\text{recon}}^{c_b}) + \lambda_s(\mathcal{L}_{\text{recon}}^{s_a} + \mathcal{L}_{\text{recon}}^{s_b})$

In summary, MUNIT combines image reconstruction, latent reconstruction, and adversarial losses to train its components. The comprehensive loss function encapsulates the various aspects of the model's objectives.

### Exploring Additional Loss Functions

The paper introduces a couple of supplementary loss functions designed to enhance convergence. Here, we provide an overview of these functions without delving into step-by-step tutorials.

**Style-augmented Cycle Consistency**

Building upon the concept of cycle consistency introduced by CycleGAN, where an image translated to the target domain and back should match the original, style-augmented cycle consistency adds a twist. This variant asserts that translating an image to the target domain and back, while preserving the original style, should yield the initial image. Although the reconstruction losses implicitly promote this idea, the authors suggest that explicit enforcement might offer advantages in specific scenarios.

**Domain Invariant Perceptual Loss**

Perceptual loss, commonly implemented through Mean Squared Error (MSE) loss between feature maps of fake and real images, is a familiar concept. However, when dealing with unpaired domains, pixel-wise loss may not be ideal due to the lack of spatial correspondence. To address this challenge, the authors introduce the Domain Invariant Perceptual Loss. This approach involves applying instance normalization to the feature maps, ensuring that MSE loss penalizes differences in statistics rather than raw pixel values.