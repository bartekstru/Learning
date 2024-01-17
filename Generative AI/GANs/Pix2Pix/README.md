### Goals
This project involves implementing a generative model inspired by Pix2Pix, a seminal paper in image-to-image translation titled [*Image-to-Image Translation with Conditional Adversarial Networks*](https://arxiv.org/abs/1611.07004) by Isola et al. (2017). The primary objective is to train a model that can convert aerial satellite imagery (input) into map routes (output) using the U-Net architecture, a key element in Pix2Pix.

![pix2pix example](pix2pix_ex.png)

### Pix2Pix Overview
Pix2Pix introduces conditional adversarial networks, combining a generative model with an adversarial loss function. This approach allows the model to learn a mapping from input to output images by simultaneously training a generator and a discriminator. The discriminator assesses the realism of generated images, while the generator aims to produce outputs that convincingly deceive the discriminator.

### Learning Objectives
1. **Loss Function Implementation:** Implement the loss function specific to Pix2Pix, differentiating it from a traditional supervised U-Net.
2. **Generator Training Dynamics:** Observe the evolution of generator priorities during training, witnessing a shift from prioritizing reconstruction to emphasizing realism.

#### U-Net Code

The U-Net implementation closely resembles the previous assignment but incorporates optional dropout and batch normalization. Adjustments in structure cater to the Pix2Pix context, ensuring the final image size aligns closely with the input image.

### PatchGAN Discriminator

Define a discriminator leveraging the contracting path of the U-Net. This allows for the evaluation of the realism of generated images. The discriminator outputs a one-channel matrix of classifications instead of a single value. The final layer maps from the final number of hidden channels to a prediction for every pixel, enhancing its suitability for adversarial training.

### Training Preparation
Assemble all components for training by defining the following parameters:

- **real_dim:** Number of channels in the real and expected output image.
- **adv_criterion:** Adversarial loss function tracking GAN performance.
- **recon_criterion:** Loss function rewarding similarity to ground truth for image "reconstruction."
- **lambda_recon:** Weight parameter for reconstruction loss.
- **n_epochs:** Number of iterations through the dataset during training.
- **input_dim:** Number of channels in the input image.
- **display_step:** Frequency of image visualization.
- **batch_size:** Number of images per pass.
- **lr:** Learning rate.
- **target_shape:** Output image size in pixels.
- **device:** Device type.

### Additional Insights
- The Pix2Pix model involves training a generator to produce realistic images and a discriminator to distinguish between real and generated images.
- A pre-trained checkpoint is available for faster model visualization, but training from scratch is also an option.
- The Pix2Pix architecture enables various image-to-image translation tasks beyond the specific aerial satellite imagery to map routes in this project.
- Experimenting with different parameters and training strategies can lead to improved model performance.