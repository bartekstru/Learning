# Generative Teaching Networks (GTN) Overview

Generative Teaching Networks (GTN) present a unique approach to accelerate Neural Architecture Search (NAS) by introducing a cooperative model consisting of a generator (teacher) and a student. This notebook explores the implementation of GTNs, as initially proposed in the work by Such et al. in [Generative Teaching Networks: Accelerating Neural Architecture Search by Learning to Generate Synthetic Training Data](https://arxiv.org/abs/1912.07768) (2019).

## Goals

The primary objective is to implement a GTN, where the generator produces synthetic data, and the student is trained on this data for a specific task. Unlike traditional Generative Adversarial Networks (GANs), GTNs operate cooperatively. Throughout this exploration, we delve into the following concepts:

1. **End-to-End Data Augmentation:** The generator synthesizes data as extra training data, and both the generator and student are trained collaboratively. Real data plays a minor role, used sporadically to update the generator based on the student's performance.

2. **Curriculum Learning:** The generator not only generates data but learns the optimal curriculum or random noise through backpropagation. This curriculum is designed to enhance student learning.

3. **Meta-Learning:** GTNs engage in meta-learning, understanding how the student learns, demonstrated through curriculum learning.

4. **Neural Architecture Search (NAS):** The generator not only guides student training but also aids in determining the optimal student architecture, a concept known as Neural Architecture Search.

## Learning Objectives

By the end of this exploration, the goals are to:

1. Grasp the concepts of teaching networks, meta-learning, and neural architecture search, understanding their connection to data augmentation.
2. Implement and train a GTN on the MNIST dataset, observing how GTNs can expedite the training process.

## Simple MNIST NAS

The central idea revolves around evaluating the performance of larger networks on teacher-generated data as a proxy for real data performance. Such et al. found that achieving similar predictive power on GTN-generated data required significantly fewer steps compared to real data. The notebook proceeds to implement a basic NAS experiment with a GTN, optimizing the number of convolutional filters for a two-layer student network trained on the teacher-generated data.

![Figure 1(a) from the GTN paper](https://github.com/https-deeplearning-ai/GANs-Public/blob/master/gtn_fig1.png?raw=true)

*Figure 1(a) from the [GTN paper](https://arxiv.org/pdf/1912.07768.pdf), providing an overview of the method.*