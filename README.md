# Implement diffusion denoising models from scratch

- run setup.sh to setup your environment
- run code in download.ipynb to gather ~1000 images from miyazaki movies
- use training.ipynb to train a model and generate sample imagery

## References:
The basic functions for training are taken from the [original ddpm paper](https://arxiv.org/pdf/2006.11239.pdf). The code here is commented with references to the specific parts the paper's algorithm.

For the Unet architecture, two are tried here

- this: https://github.com/dome272/Diffusion-Models-pytorch/blob/main/modules.py
- and this: https://huggingface.co/blog/annotated-diffusion

Both are similar and gave similar results but ultimately the former was chosen since it's a bit simpler/smaller and easier to play with

In addition, some of the image transforms are taken from those sources as well.