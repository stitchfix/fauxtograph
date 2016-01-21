# Changelog

## v1.0.x
* Now has 3 different model classes available (`VAE`, `GAN`, `VAEGAN`)

* All models have both `convolution` and `linear` mode architectures.

* Python3 Compatibility

* Updated to use Chainer 1.6.0

* Will output intermediate generated images to give users the ability to inspect training progress when run in a Jupyter notebook.



# fauxtograph
This package contains classes for training three different unsupervised, generative image models. Namely Variational Auto-encoders, Generative Adversarial Networks, and the newly developed combination of the two (VAE/GAN). Descriptions of the inner workings of these algorithms can be found in

1. Kingma, Diederik P., and Max Welling. "Auto-encoding variational bayes." arXiv preprint arXiv:1312.6114 (2013).
2. Radford, Alec et. al.; "Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks" arXiv preprint arxiv:1511.06434 (2015).
3. Boesen Lindbo Larsen, Anders et. al.; "Autoencoding Beyond Pixels Using a Learned Similarity Metric" arXiv preprint arxiv:1512.09300 (2015).

respectively.


All models take in a series of images and can be trained to perform either an encoding `transform` step or a generative `inverse_transform` step (or both). It's built on top of the Chainer framework and has an easy to use command line interface for training and generating images with a Variational Auto-encoder. 

Both the module itself as well as the training script are available by installing this package through PyPI. Otherwise the module itself containing the main class which does all the heavy lifting is in  `fauxtograph/fauxtograph.py` which has dependencies in `fauxtograph/vaegan.py`, while the training/generation CLI script is in `fauxtograph/fauxto.py`

To learn more about the command line tool functionality and to get a better sense of how one might use it, please see the blog post on the Stitch Fix tech blog, [multithreaded](http://multithreaded.stitchfix.com/blog/2015/09/17/deep-style/).



##Installation

The simplest step to using the module is to install via pip:

```bash
$ pip install fauxtograph
```
this should additionally grab all necessary dependencies including the main backend NN framework, Chainer. However, if you plan on using CUDA to train the model with a GPU you'll need to additionally install the Chainer CUDA dependencies with

```bash
$ pip install chainer-cuda-deps
```

##Usage

To get started, you can either find your own image set to use or use the downloading tool to grab some of the [Hubble/ESA space images](https://www.spacetelescope.org/images/viewall/), which I've found make for interesting results. 

To grab the images and place them in an `images` folder run

```bash
$ fauxtograph download ./images
```

*This process can take some time depending on your internet connection.*

Then you can train a model and output it to disk with 
```bash
$ fauxtograph train ./images ./models/model_name 
```

Finally, you can generate new images based on your trained model with

```bash
$ fauxtograph generate ./models/model_name_model.h5 ./models/model_name_opt.h5 ./models/model_name_meta.h5 ./generated_images_folder
```
Each command comes with a `--help` option to see possible optional arguments. 



## Tips
### Using the CLI
* In order to get the best results for generated images it'll be necessary to either have a rather large number of images (say on the order of several hundred thousand or more), or images that are all quite similar with minimal backgrounds. 

* As the model trains you should see the output of the KL Divergence average over the batches and the reconstruction loss average as well. You might wish to adjust the ratio of these two terms with the `--kl_ratio` option in order to get better performance should you find that the learning rate is driving one or the other terms to zero too quickly(slowly).


* If you have an CUDA capable Nvidia GPU, use it. The model can train over 10 times faster by taking advantage of GPU processing. 

* Sometimes you will want to brighten your images when saving them, which can be done with the `--image_multiplier` argument. 

* If you manage to train a particularly interesting model and generate some neat images, then we'd like to see them. Use #fauxtograph if you decide to put them up on social media.

### Generally
* When training GAN and VAEGAN models, they are highly sensitive to the relative learning rate of the subnetworks. Particularly the learning rate of the generator to the discriminator. If you notice highly oscillatory behavior in your training losses it might be helpful to turn down the Adam `alpha` and `beta1` parameters of either network (usually the discriminator) to help train them at a similar rate.


ENJOY