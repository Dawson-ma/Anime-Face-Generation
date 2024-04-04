# Anime Face Generation

Generating anime faces using a generative adversarial network (GAN). The model has been recognized as the best example among 1000 students, showcasing a remarkable Frechet inception distance score of **7293.72** and an anime face detection rate of **0.697**.

## Model Architecture

The model employed in this project is [StyleGAN2](https://github.com/lucidrains/stylegan2-pytorch), incorporating several advanced techniques:

1. **Progressive Growing**: Like the original StyleGAN, StyleGAN2 utilizes progressive growing, gradually increasing the resolution of both the generator and discriminator networks during training. This technique stabilizes the training process and facilitates the generation of high-resolution images.

2. **Disentangled Latent Space**: StyleGAN2 introduces disentangled latent space representations, allowing separate control over various attributes of the generated images. This enables precise manipulation of features such as pose, expression, and appearance, resulting in diverse and controllable image synthesis.

3. **Adaptive Instance Normalization (AdaIN)**: AdaIN is employed in StyleGAN2 to adaptively normalize the activations of intermediate layers in the generator network based on the input latent code. This enhances style transfer and improves the diversity of generated images.

4. **Skip Connections**: StyleGAN2 incorporates skip connections between the generator's latent inputs and intermediate layers to facilitate information flow and gradient propagation. This improves training stability and convergence speed.

5. **Path Length Regularization**: This regularization technique encourages smooth and continuous changes in the latent space by penalizing deviations from the expected path length. It helps prevent mode collapse and promotes the generation of diverse images.

6. **Improved Normalization Layers**: StyleGAN2 utilizes improved normalization layers such as group normalization and equalized learning rate to stabilize training and enhance overall performance.

7. **Efficient Architecture Design**: The architecture of StyleGAN2 is optimized for memory efficiency and computational speed, enabling faster training and inference on both CPU and GPU platforms.

8. **Transfer Learning**: StyleGAN2 supports transfer learning, allowing users to fine-tune pre-trained models on specific datasets. This facilitates faster convergence and improved performance when working with limited data.

These techniques collectively contribute to StyleGAN2's superior image synthesis capabilities, stability, and efficiency compared to its predecessors.

## Dataset

The [Crypko dataset](https://drive.google.com/file/d/1-ZmrU2qHQWDEG6wnweUYK3h7cEfN5_2h/view?usp=sharing) serves as the primary data source for this project, originally obtained from [Crypko](https://crypko.ai/en/).

For detailed implementation and usage instructions, please refer to the provided [code](https://github.com/Dawson-ma/Anime-Face-Generation/blob/main/main.py).
