# Neural Radiance Fields (NeRF)

*Please note that this is an optional notebook meant to introduce more advanced concepts. If you’re up for a challenge, take a look and don’t worry if you can’t follow everything. There is no code to implement—only some cool code for you to learn and run!*

### Goals

In this notebook, you'll learn how to use Neural Radiance Fields to generate new views of a complex 3D scene using only a couple input views, first proposed by [NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis](https://arxiv.org/abs/2003.08934) (Mildenhall et al. 2020). Though 2D GANs have seen success in high-resolution image synthesis, NeRF has quickly become a popular technique to enable high-resolution 3D-aware GANs.


## Overview

NeRF is an approach for **novel view synthesis**, where given some input images of a scene and cooresponding camera poses, we want to generate new images of the same scene from arbitrary camera poses. Because training a full NeRF can take hours to days, we will study a feature-limited tiny NeRF ([official GitHub](https://colab.research.google.com/github/bmild/nerf/blob/master/tiny_nerf.ipynb)) to train faster, while highlighting the major differences.

### How does NeRF work?

NeRF represents a scene as a function mapping position and direction to color and volumetric density (how opaque is this object?), $F_\Theta : (x, y, z, \theta, \phi) \mapsto (R, G, B, \sigma)$. The authors then use these colors and densities with classic volume rendering techniques to compose these values into an image.

NeRF represents this mapping with a simple Multilayer Perceptron (MLP), which is differentiable and thus allows for explicit optimization by comparing the synthesized with the ground truth images.

![NeRF Pipeline](https://github.com/bmild/nerf/blob/master/imgs/pipeline.jpg?raw=true)
Image credit: [Official GitHub](https://github.com/bmild/nerf)

### Positional Encoding

To better model high-frequency functions, the authors use an encoding function defined

$$\gamma(p) = (\sin(2^0 \pi p), \cos(2^0 \pi p), \dots, \sin(2^{L-1} \pi p), \cos(2^{L-1} \pi p)).$$

In the NeRF architecture, $\gamma$ is applied to each of the 5 input dimensions $(x, y, z, \theta, \phi)$.

### Architecture

NeRF is a simple MLP with ReLU activations, concatenations at specific layers, and outputs at different stages shown below:

<img src="https://miro.medium.com/max/1129/1*q3fLvJFfoUdtVhsXeeTNXw.png" alt="NeRF Architecture" width="800"/>

For training speed, we instead implement a smaller MLP in the same spirit:

### Volume Rendering

Consider a camera ray $\mathbf r(t) = \mathbf o + t \mathbf d$, with origin $\mathbf o$ and direction $\mathbf d$. If each 3D point along this ray is associated with a color $\mathbf c(\mathbf r(t), \mathbf d)$ and density $\sigma(\mathbf r(t)$, then the authors render the expected color at $\mathbf r(t)$ as

$$C(\mathbf r) = \int_{t_n}^{t_f} T(t) \sigma(\mathbf r(t)) \mathbf c(\mathbf r(t), \mathbf d) dt,$$

where

$$T(t) = \exp \left( - \int_{t_n}^t \sigma(\mathbf r(s)) ds \right),$$

and $t_n$ and $t_f$ are the near and far bounds of what we wish to render. We can interpret $T(t)$ as the probability that the ray travels from $t_n$ to $t$ without hitting any other particle.

This formulation is continuous, so the authors discretize it with stratified sampling: divide the interval $[t_n, t_f]$ into $N$ even bins, and then sample uniformly over each bin. The sampling is critical so we can roughly cover the whole interval over the course of training. This yields the discretization

$$\hat C(\mathbf r) = \sum_{i=1}^N T_i (1 - \exp(- \sigma_i \delta_i)) \mathbf c_i,$$

with

$$T_i = \exp \left( - \sum_{j=1}^{i-1} \sigma_j \delta_j \right),$$

and $\delta_i = t_{i+1} - t_i$ as the difference series between sample points.

### Loss Function and Optimizer

The authors formulate a simple loss function as the total squared error between the rendered and ground truth RGB image values. The authors use the Adam optimizer with default parameters and learning rate that begins at $5 \times 10^{-4}$ and exponentially decays to $5 \times 10^{-5}$.

### Modifications

* **Input dimension**: Unlike the 5D input of the original NeRF model, our TinyNeRF only uses 3 input dimensions $(x, y, z)$, omitting the camera angle.
* **Architecture**: We simplified the NeRF architecture to train faster.
* **Optimizer**: We use the default Adam optimizer (lr= $5 \times 10^{-3}$) without exponential decay.
* **Hierarchical Volume Sampling**: Because sampling uniformly across a camera ray is not efficient (since much of the ray does not intersect with the object) NeRF actually uses a coarse network to determine where to sample, and then a fine network that samples from points that will contribute most to the output image. We use the simpler stratified sampling.

## Extensions

Tying this back to GANs, many papers have found success in applying NeRFs to 3D aware GANs. This is a very active research area, so here's a couple pointers:

* [GRAF: Generative Radiance Fields for 3D-Aware Image Synthesis](https://arxiv.org/abs/2007.02442) (Schwarz et al. 2020) demonstrated a NeRF model that can generate classes of objects by conditioning on shape and appearance.

* [pi-GAN: Periodic Implicit Generative Adversarial Networks for 3D-Aware Image Synthesis](https://arxiv.org/abs/2012.00926) (Chan et al. 2021) proposed the current, as of writing, state-of-the-art 3D aware image synthesis technique through more expressive architectures. A comparison of these techniques is below, credit [Marco Monteiro](https://marcoamonteiro.github.io/pi-GAN-website/).

    ![Comparison of 3D-aware GANs](https://raw.githubusercontent.com/alexxke/nerf-images/main/comparison.gif)

* [GIRAFFE: Representing Scenes as Compositional Generative Neural Feature Fields](https://arxiv.org/abs/2011.12100) (Niemeyer and Geiger 2021) developed an approach that can model multi-object scenes as compositions of NeRFs. Check out some visualizations of object translation from a 2D GAN (left) and GIRAFFE (right), credit to [Michael Niemeyer](https://m-niemeyer.github.io/project-pages/giraffe/index.html).

    ![2D GAN single object translation](https://raw.githubusercontent.com/alexxke/nerf-images/main/2dgan.gif)
![GIRAFFE single object translation](https://raw.githubusercontent.com/alexxke/nerf-images/main/giraffe.gif)

* [Block-NeRF: Scalable Large Scene Neural View Synthesis](https://arxiv.org/abs/2202.05263) (Tancik et al. 2022) takes scene decomposition even further by introducing tweaks that allow Block-NeRF to render an entire neighborhood of San Francisco (credit to [Waymo Research](https://waymo.com/research/block-nerf/)).

    ![Block-NeRF on Grace Cathedral](https://raw.githubusercontent.com/alexxke/nerf-images/main/grace_cathedral.gif)


For more general improvements on this technique, Frank Dellaert put together a great [anthology](https://dellaert.github.io/NeRF/).
