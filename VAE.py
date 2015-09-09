import chainer
from chainer import cuda
import chainer.functions as F
from chainer import optimizers
import numpy as np
from PIL import Image, ImageChops
import time
import joblib
import json
import os
#import pickle
from tqdm import *

if not os.path.exists('cache'):
    os.mkdir('cache')
mem = joblib.Memory('cache')

# TODO: unit test
#   TODO: unit test for pickle / depickle, c/gpu
#   TODO: test that before training the RMSE is as random as you'd expect
#   TODO: deccode / encode on an easy example
# TODO: setup.py & PyPI Instllation
# TODO: Downloader for images
# TODO: docstrings for public functions
# TODO: private funcs tend to have underscores
# TODO: watch the 80 character limit


class ImageAutoEncoder():
    '''Variational Auto-encoder Class for image data.

    Using linear transformations with ReLU activations, this class peforms
    an encoding and then decoding step to form a full generative model for
    image data. Images can thus be encoded to a latent representation-space or
    decoded/generated from latent vectors.

    See  Kingma, Diederik and Welling, Max; "Auto-Encoding Variational Bayes" (2013)

    Given a set of input images train an artificial neural network, respampling at the latent
    stage from an approximate posterior multivariate gaussian distribution
    with unit covariance with mean and variance trained by the encoding step.

    Attributes
    ----------
    encode_sizes : List[int]
        List of layer sizes for hiden linear encoding layers of the model.
    decode_sizes : List[int]
        List of layer sizes for hiden linear decoding layers of the model.
    latent_dim : int
        Dimension of latent encoding space.
    img_width : int
        Width of the desired image representation.
    img_height : int
        Height of the desired image representation.
    color_channels : int
        Number of color channels in the input images.
    img_length : int
        Total dimensionality of input image `img_length = img_width *
        img_height * color_channels`.
    flag_gpu : bool
        Flag to toggle GPU training functionality.
    flag_dropout : bool
        Flag to toggle image dropout functionality.
    flag_autocrop : bool
        Flag to toggle image autocropping from background functionality.
    flag_grayscale : bool
        Flag to toggle image color channel averaging to grayscale.
    model : chainer.FunctionSet
        FunctionSet of chainer model layers for encoding and decoding.
    optimizer : chainer.Optimizer
        Chiner optimizer used to do backpropagation.
    x_all : numpy.array
        Numpy array used to hold training image data.





    '''

    def __init__(self, img_width=75, img_height=100, color_channels=3,
                 encode_layers=[1000, 600, 300],
                 decode_layers=[300, 800, 1000],
                 latent_width=100, flag_gpu=None, flag_dropout=False,
                 flag_autocrop=False, flag_grayscale=False):
        '''Setup for the variational auto-encoder.

        Inititalizes the layer setup for the NN to the defined dimensions.

        Parameters
        ----------

        img_width : int
            Width of the desired image representation.
        img_height : int
            Height of the desired image representation.
        color_channels : int
            Number of color channels in the input images (set to 1 automatically
                if `flag_grayscale = True`).
        encode_layers : List[int]
            List of layer sizes for hiden linear encoding layers of the model.
        decode_layers : List[int]
            List of layer sizes for hiden linear decoding layers of the model.
        latent_width : int
            Dimension of latent encoding space.
        flag_gpu : bool
            Flag to toggle GPU training functionality.
        flag_dropout : bool
            Flag to toggle image dropout functionality.
        flag_autocrop : bool
            Flag to toggle image autocropping from background functionality.
        flag_grayscale : bool
            Flag to toggle image color channel averaging to grayscale.


        '''
        self.encode_sizes = encode_layers
        self.decode_sizes = decode_layers
        self.latent_dim = latent_width
        self.img_width = img_width
        self.img_height = img_height
        if flag_grayscale:
            slef.color_channels = 1
        else:
            self.color_channels = color_channels
        self.img_len = self.img_width*self.img_height*self.color_channels
        self.flag_gpu = flag_gpu
        self.flag_dropout = flag_dropout
        self.flag_autocrop = flag_autocrop
        self.flag_grayscale = flag_grayscale
        self.model = self._layer_setup()
        self.optimizer = optimizers.Adam()
        self.optimizer.setup(self.model)
        self.x_all = np.array([], dtype=np.float32)

    def _layer_setup(self):
        # setup chainer layers for NN.

        layers = {}

        # Encoding Steps
        encode_layer_pairs = [(self.img_len, self.encode_sizes[0])]
        encode_layer_pairs += zip(self.encode_sizes[:-1],
                                  self.encode_sizes[1:])
        encode_layer_pairs += [(self.encode_sizes[-1], self.latent_dim * 2)]
        for i, (n_in, n_out) in enumerate(encode_layer_pairs):
            layers['encode_%i' % i] = F.Linear(n_in, n_out)

        # Decoding Steps
        decode_layer_pairs = [(self.latent_dim, self.decode_sizes[0])]
        decode_layer_pairs += zip(self.decode_sizes[:-1],
                                  self.decode_sizes[1:])
        decode_layer_pairs += [(self.decode_sizes[-1], self.img_len)]

        for i, (n_in, n_out) in enumerate(decode_layer_pairs):
            layers['decode_%i' % i] = F.Linear(n_in, n_out)
        model = chainer.FunctionSet(**layers)

        if self.flag_gpu:
            cuda.init()
            model.to_gpu()

        return model

    def _encode(self, img_batch):

        batch = img_batch

        if self.flag_dropout:
            batch = F.dropout(batch)
        n_layers = len(self.encode_sizes)
        for i in xrange(n_layers):
            batch = F.relu(getattr(self.model, 'encode_%i' % i)(batch))
        batch = F.relu(getattr(self.model, 'encode_%i' % n_layers)(batch))

        return batch

    def _decode(self, latent_vec):

        batch = latent_vec
        n_layers = len(self.decode_sizes)
        for i in xrange(n_layers):
            batch = F.relu(getattr(self.model, 'decode_%i' % i)(batch))
        batch = F.sigmoid(getattr(self.model, 'decode_%i' % n_layers)(batch))
        return batch

    def forward(self, img_batch):
        # TODO: underscore this method if it isn't public
        batch = chainer.Variable(img_batch / 255.)
        encoded = self._encode(batch)

        #Split latent space into `\mu` and `\sigma` parameters
        mean, std = F.split_axis(encoded, 2, 1)

        #create `latent_dim` N(0,1) normal samples.
        samples = np.random.standard_normal(mean.data.shape).astype('float32')
        if self.flag_gpu:
            samples = cuda.to_gpu(samples)
        samples = chainer.Variable(samples)

        #Scale samples to model trained parameters.
        sample_set = samples * F.exp(0.5*std) + mean
        output = self._decode(sample_set)
        reconstruction_loss = F.mean_squared_error(output, batch)

        #construct and scale KL Divergence loss.
        kl_div = -0.5 * F.sum(1 + std - mean ** 2 - F.exp(std))
        kl_div /= (img_batch.shape[1] * img_batch.shape[0])

        return reconstruction_loss, kl_div, output

    #@mem.cache
    def load_images(self, files):

        # TODO: docstring for all public functions
        # TODO: use logger instead of print
        # https://docs.python.org/2/library/logging.html
        shape = (self.img_width, self.img_height)
        print("Loading Image Files...")

        #helper function to crop images from background
        def trim(im):
            bg = Image.new(im.mode, im.size, im.getpixel((0, 0)))
            diff = ImageChops.difference(im, bg)
            diff = ImageChops.add(diff, diff, 2.0, -50)
            bbox = diff.getbbox()
            if bbox:
                return im.crop(bbox)

        #helper function to quickly resize images
        def resize(fname):
            im = Image.open(fname)
            #on the fly image cropping
            if self.flag_autocrop:
                im = trim(im)
            im.thumbnail(shape, Image.ANTIALIAS)
            im = im.resize(shape, Image.ANTIALIAS)
            im = np.float32(im)
            if self.flag_grayscale:
                im = im.mean(axis=2)
            return im
        self.x_all = np.array([resize(fname) for fname in tqdm(files)]).astype('float32')
        print("Image Files Loaded!")

    def fit(self, n_epochs=200, batch_size=100):
        # TODO: docstring for all public functions
        # TODO: rename to use sklearn's fit / predict / transform convention
        if self.x_all.shape[0] == 0:
            raise ValueError('Images not yet loaded, call load_images() first.')
        n_samp = self.x_all.shape[0]
        x_train = self.x_all.flatten().reshape((n_samp, -1))
        # Train Model#
        print("\n Training for %i epochs. \n" % n_epochs)
        for epoch in xrange(1, n_epochs + 1):
            print('epoch: %i' % epoch)
            t1 = time.time()
            indexes = np.random.permutation(x_train.shape[0])
            last_loss_kl = 0.
            last_loss_rec = 0.
            q = len(range(0, x_train.shape[0], batch_size))
            for i in tqdm(xrange(0, x_train.shape[0], batch_size)):
                if self.flag_gpu:
                    x_batch = cuda.to_gpu(x_train[indexes[i: i + batch_size]])
                else:
                    x_batch = x_train[indexes[i: i + batch_size]]
                self.optimizer.zero_grads()
                r_loss, kl_div, out = self.forward(x_batch)
                loss = r_loss + kl_div
                loss.backward()
                self.optimizer.update()
                last_loss_kl += kl_div.data
                last_loss_rec += r_loss.data

            msg = "r_loss = {0} , kl_div = {1}"
            print(msg.format(last_loss_rec/q, last_loss_kl/q))
            t_diff = time.time()-t1
            print("time: %f\n\n" % t_diff)

    def dump(self, filepath):
        # TODO: docstring for all public functions
        if not os.path.exists(os.path.dirname(filepath)):
            os.makedirs(os.path.dirname(filepath))
        print("Dumping model to file: %s " % filepath)
        cls_data = self.__dict__
        model = cls_data.pop('model')
        model.to_cpu()
        cls_data.pop('optimizer')
        cls_data.pop('x_all')
        meta = json.dumps(cls_data)
        with open(filepath+'_meta.json', 'wb') as f:
            f.write(meta)
        joblib.dump(model, filepath)

    @classmethod
    def load(cls, filepath, flag_gpu=False):
        # TODO: docstring for all public functions
        modpath = filepath
        metapath = filepath + '_meta.json'

        mess = "Model file {0} does not exist. Please check the file path."
        if not os.path.exists(modpath):
            print(mess.format(modpath))
        elif not os.path.exists(metapath):
            print(mess.format(metapath))
        else:
            with open(metapath, 'r') as f:
                meta = json.load(f)
            new_vae = cls(img_width=meta['img_width'],
                          img_height=meta['img_height'],
                          color_channels=meta['color_channels'],
                          encode_layers=meta['encode_sizes'],
                          decode_layers=meta['decode_sizes'],
                          latent_width=meta['latent_dim'],
                          flag_gpu=flag_gpu,
                          flag_dropout=meta['flag_dropout'],
                          flag_autocrop=meta['flag_autocrop'],
                          flag_grayscale=meta['flag_grayscale'])
            new_vae.model = joblib.load(modpath)
            new_vae.optimizer = optimizers.Adam()
            new_vae.optimizer.setup(new_vae.model)
            return new_vae
