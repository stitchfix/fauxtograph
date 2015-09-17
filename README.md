# fauxtograph
This code can be used for using a **variational auto-encoder** for latent image encoding and generation. It implements the algorithm found in 

Kingma, Diederik P., and Max Welling. "Auto-encoding variational bayes." arXiv preprint arXiv:1312.6114 (2013).


The model takes in a series of images and trains an auto-encoder to perform an encoding `transform` step as well as a generative `inverse_transform` step. It's built on top of the Chainer framework and has an easy to use command line interface for training and generating images. 

Both the module itself as well as the training script are available by installing this package through PyPI. Otherwise the module itself containing the main class which does all the heavy lifting is in  `fauxtograph/fauxtograph.py` while the training/generation CLI script is in `fauxtograph/fauxto.py`

To learn more about its functionality and to get a better sense of how one might us it, please see the blog post on the Stitch Fix tech blog, [multithreaded](http://multithreaded.stitchfix.com/blog/2015/09/17/vizualizing-style-post/).



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
$ fauxtograph generate ./models/model_name ./generated_images_folder
```
Each command comes with a `--help` option to see possible optional arguments. 



##Tips

* In order to get the best results for generated images it'll be necessary to either have a rather large number of images (say on the order of several hundred thousand or more), or images that are all quite similar with minimal backgrounds. 

* As the model trains you should see the output of the KL Divergence average over the batches and the reconstruction loss average as well. You might wish to adjust the ratio of these two terms with the `--kl_ratio` option in order to get better performance should you find that the learning rate is driving one or the other terms to zero too quickly(slowly).


* If you have an CUDA capable Nvidia GPU, use it. The model can train over 10 times faster by taking advantage of GPU processing. 

* If you manage to train a particularly interesting model and generate some neat images, then we'd like to see them. Use #fauxtograph if you decide to put them up on social media.


ENJOY