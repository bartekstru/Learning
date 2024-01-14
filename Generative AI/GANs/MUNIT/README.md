# Introduction
This project implements Multimodal Unsupervised Image-to-Image Translation, as proposed in the paper [Multimodal Unsupervised Image-to-Image Translation](https://arxiv.org/abs/1804.04732) by Huang et al. (2018).

MUNIT represents an advancement over UNIT, which oversimplifies the assumption of modeling image-to-image translation as a one-to-one mapping. MUNIT proposes that image representation can be decomposed into a domain-invariant content code and domain-specific style code, capturing domain-specific properties.

## UNIT

UNIT, introduced in the paper [Unsupervised Image-to-Image Translation Networks](https://arxiv.org/abs/1703.00848) by Liu et al. (2018), is an image-to-image translation method based on the assumption that images from different domains share a common latent distribution.

In this method, let's consider two image domains, $\mathcal{A}$ and $\mathcal{B}$. Images $(x_a, x_b) \in (\mathcal{A}, \mathcal{B})$ are encoded into a shared latent space, $\mathcal{Z}$, through encoders $E_a: x_a \mapsto z$ and $E_b: x_b \mapsto z$. Generators $G_a: z \mapsto x_a'$ and $G_b: z \mapsto x_b'$ are utilized to produce synthetic images. Note that these generators can create self-reconstructed or domain-translated images for their respective domains.

As with other GAN frameworks, synthetic and real images, $(x_a',x_a)$, and $(x_b', x_b)$, are input into discriminators, $D_a$ and $D_b$, respectively.

## MUNIT

Now, considering two image domains, $\mathcal{A}$ and $\mathcal{B}$, a pair of corresponding images $(x_a, x_b) \in (\mathcal{A}, \mathcal{B})$ can be generated as $x_a = F_a(c, s_a)$ and $x_b = F_b(c, s_b)$. Here, $c$ is a content vector from a shared distribution, $s_a, s_b$ are style vectors from distinct distributions, and $F_a, F_b$ are decoders that synthesize images from the content and style vectors.

The fundamental idea is that while the content between two domains can be shared (e.g., interchange horses and zebras in an image), the styles differ between the two (e.g., drawing horses and zebras differently).

To learn the content and style distributions during training, the authors assume that $E_a, E_b$ invert $F_a, F_b$, respectively. In particular, $E_a^c: x_a \mapsto c$ extracts content, and $E_a^s: x_a \mapsto s_a$ extracts style from images in domain $\mathcal{A}$. The same applies to $E_b^c(x_b)$ and $E_b^s(x_b)$ with images in domain $\mathcal{B}$. The ability to mix and match content and style vectors from the two domains allows for translating images between them.

For instance, by taking content $b$, $c_b = E_b^c(x_b)$, and style $a$, $s_a = E_a^s(x_a)$, and passing these through the horse decoder as $F_a(c_b, s_a)$, one should obtain an image $b$ drawn with characteristics of image $a$.

