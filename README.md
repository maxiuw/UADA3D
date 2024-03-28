# UADA3D: Unsupervised Adversarial Domain Adaptation for 3D Object Detection with Sparse LiDAR and Large Domain Gaps

[![arXiv](https://img.shields.io/badge/arXiv-2403.17633-b31b1b.svg?style=plastic)](https://arxiv.org/abs/2403.17633)

![Method Overview](https://raw.githubusercontent.com/maxiuw/UADA3D/main/.github/main.png)

## Summary 
In this study, we address a gap in existing unsupervised domain adaptation approaches on LiDAR-based 3D object detection, which have predominantly concentrated on adapting between established, high-density autonomous driving datasets. We focus on sparser point clouds, capturing scenarios from different perspectives: not just from vehicles on the road but also from mobile robots on sidewalks, which encounter significantly different environmental conditions and sensor configurations. We introduce Unsupervised Adversarial Domain Adaptation for 3D Object Detection (UADA3D). UADA3D does not depend on pre-trained source models or teacher-student architectures. Instead, it uses an adversarial approach to directly learn domain-invariant features. We demonstrate its efficacy in various adaptation scenarios, showing significant improvements in both self-driving car and mobile robot domains.


## Method overview 

provides a schematic overview of our method UADA3D. In each iteration, a batch of samples $Q$ from source $D_s$ and target $D_t$ domain is fed to the feature extractor $f_{\theta_f}$. Next, for each sample, features are extracted, and fed to the detection head $h_{\theta_y}$ that predicts 3D bounding boxes. The object detection loss (described in supplementary materials) is calculated only for the labeled samples from source domain. The probability distribution alignment branch uses the domain discriminator $g_{\theta_D}$ to predict from which domain samples came from, based on the extracted features $X$ and predicted labels $\hat{Y}$. The domain loss $L_C$ is calculated for all samples. Next, the $L_C$ is backpropagated through the discriminators, and through the gradient reversal layer (GRL) with the coefficient $\lambda$, that reverses the gradient during backpropagation, to detection head and feature extractor. This adversarial training scheme works towards creating domain invariant features. Thus, our network learns how to extract features that will be domain invariant but also how to provide accurate predictions. Therefore, we seek for the optimal parameters $\theta_f*$, $\theta_y*$, and $\theta_D*$.

## Cite our Paper
```
@article{wozniak2024uada3d,
  title={UADA3D: Unsupervised Adversarial Domain Adaptation for 3D Object Detection with Sparse LiDAR and Large Domain Gaps},
  author={Wozniak, K Maciej, and Hansson, Mattias and Thiel, Marko and Jensfelt, Patric},
  journal={arXiv preprint arXiv:2403.17633},
  year={2024}
}
```
