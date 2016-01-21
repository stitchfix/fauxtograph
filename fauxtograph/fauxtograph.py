from PIL import Image
import chainer.functions as F
from chainer import Variable
import chainer.optimizers as O
from chainer import serializers
import matplotlib.pyplot as plt
from vaegan import *
import tqdm
import time
import numpy as np
import os
from IPython.display import display
import json


class VAE(object):
    '''Variational Auto-encoder Class for image data.

    Using linear/convolutional transformations with ReLU activations, this
    class peforms an encoding and then decoding step to form a full generative
    model for image data. Images can thus be encoded to a latent representation-space
    or decoded/generated from latent vectors.

    See  Kingma, Diederik and Welling, Max; "Auto-Encoding Variational Bayes"
    (2013)

    Given a set of input images train an artificial neural network, resampling
    at the latent stage from an approximate posterior multivariate gaussian
    distribution with unit covariance with mean and variance trained by the
    encoding step.

    Attributes
    ----------
    encode_layers : List[int]
        List of layer sizes for hidden linear encoding layers of the model.
        Only taken into account when mode='linear'.
    decode_layers : List[int]
        List of layer sizes for hidden linear decoding layers of the model.
        Only taken into account when mode='linear'.
    latent_width : int
        Dimension of latent encoding space.
    img_width : int
        Width of the desired image representation.
    img_height : int
        Height of the desired image representation.
    color_channels : int
        Number of color channels in the input images.
    kl_ratio : float
            Multiplicative factor on and KL Divergence term.
    flag_gpu : bool
        Flag to toggle GPU training functionality.
    mode: str
        Mode to set the encoder and decoder architectures. Can be either
        'convolution' or 'linear'.
    adam_alpha : float
        Alpha parameter for the adam optimizer.
    adam_beta1 : float
        Beta1 parameter for the adam optimizer.
    rectifier : str
        Rectifier option for the output of the decoder. Can be either
        'clipped_relu' or 'sigmoid'.
    model : chainer.Chain
        Chain of chainer model links for encoding and decoding.
    opt : chainer.Optimizer
        Chiner optimizer used to do backpropagation. Set to Adam.
    '''
    def __init__(self, img_width=64, img_height=64, color_channels=3,
                 encode_layers=[1000, 600, 300],
                 decode_layers=[300, 800, 1000],
                 latent_width=100, kl_ratio=1.0, flag_gpu=True,
                 mode='convolution', adam_alpha=0.001, adam_beta1=0.9,
                 rectifier='clipped_relu'):

        self.img_width = img_width
        self.img_height = img_height
        self.color_channels = color_channels
        self.encode_layers = encode_layers
        self.decode_layers = decode_layers
        self.latent_width = latent_width
        self.kl_ratio = kl_ratio
        self.flag_gpu = flag_gpu
        self.mode = mode
        self.adam_alpha = adam_alpha
        self.adam_beta1 = adam_beta1
        self.rectifier = rectifier
        if self.mode == 'convolution':
            self._check_dims()

        self.model = EncDec(
            img_width=self.img_width,
            img_height=self.img_height,
            color_channels=self.color_channels,
            encode_layers=self.encode_layers,
            decode_layers=self.decode_layers,
            latent_width=self.latent_width,
            mode=self.mode,
            flag_gpu=self.flag_gpu,
            rectifier=self.rectifier
        )
        if self.flag_gpu:
            self.model = self.model.to_gpu()

        self.opt = O.Adam(alpha=self.adam_alpha, beta1=self.adam_beta1)

    def _check_dims(self):
        h, w = calc_fc_size(self.img_height, self.img_width)[1:]
        h, w = calc_im_size(h, w)

        assert (h == self.img_height) and (w == self.img_width),\
            "To use convolution, please resize images " + \
            "to nearest target height, width = %d, %d" % (h, w)

    def _save_meta(self, filepath):
        if not os.path.exists(os.path.dirname(filepath)):
            os.makedirs(os.path.dirname(filepath))
        d = self.__dict__.copy()
        d.pop('model')
        d.pop('opt')
        # d.pop('xp')
        meta = json.dumps(d)
        with open(filepath+'.json', 'wb') as f:
            f.write(meta)

    def transform(self, data, test=False):
        '''Transform image data to latent space.

        Parameters
        ----------
        data : array-like shape (n_images, image_width, image_height,
                                   n_colors)
            Input numpy array of images.
        test [optional] : bool
            Controls the test boolean for batch normalization.

        Returns
        -------
        latent_vec : array-like shape (n_images, latent_width)
        '''
        #make sure that data has the right shape.
        if not type(data) == Variable:
            if len(data.shape) < 4:
                data = data[np.newaxis]
            if len(data.shape) != 4:
                raise TypeError("Invalid dimensions for image data. Dim = %s.\
                     Must be 4d array." % str(data.shape))
            if data.shape[1] != self.color_channels:
                if data.shape[-1] == self.color_channels:
                    data = data.transpose(0, 3, 1, 2)
                else:
                    raise TypeError("Invalid dimensions for image data. Dim = %s"
                                    % str(data.shape))
            data = Variable(data)
        else:
            if len(data.data.shape) < 4:
                data.data = data.data[np.newaxis]
            if len(data.data.shape) != 4:
                raise TypeError("Invalid dimensions for image data. Dim = %s.\
                     Must be 4d array." % str(data.data.shape))
            if data.data.shape[1] != self.color_channels:
                if data.data.shape[-1] == self.color_channels:
                    data.data = data.data.transpose(0, 3, 1, 2)
                else:
                    raise TypeError("Invalid dimensions for image data. Dim = %s"
                                    % str(data.shape))

        # Actual transformation.
        if self.flag_gpu:
            data.to_gpu()
        z = self.model.encode(data, test=test)[0]

        z.to_cpu()

        return z.data

    def inverse_transform(self, data, test=False):
        '''Transform latent vectors into images.

        Parameters
        ----------
        data : array-like shape (n_images, latent_width)
            Input numpy array of images.
        test [optional] : bool
            Controls the test boolean for batch normalization.

        Returns
        -------
        images : array-like shape (n_images, image_width, image_height,
                                   n_colors)
        '''
        if not type(data) == Variable:
            if len(data.shape) < 2:
                data = data[np.newaxis]
            if len(data.shape) != 2:
                raise TypeError("Invalid dimensions for latent data. Dim = %s.\
                     Must be a 2d array." % str(data.shape))
            data = Variable(data)

        else:
            if len(data.data.shape) < 2:
                data.data = data.data[np.newaxis]
            if len(data.data.shape) != 2:
                raise TypeError("Invalid dimensions for latent data. Dim = %s.\
                     Must be a 2d array." % str(data.data.shape))
        assert data.data.shape[-1] == self.latent_width,\
            "Latent shape %d != %d" % (data.data.shape[-1], self.latent_width)

        if self.flag_gpu:
            data.to_gpu()
        out = self.model.decode(data, test=test)

        out.to_cpu()

        if self.mode == 'linear':
            final = out.data
        else:
            final = out.data.transpose(0, 2, 3, 1)

        return final

    def load_images(self, filepaths):
        '''Load in image files from list of paths.

        Parameters
        ----------

        filepaths : List[str]
            List of file paths of images to be loaded.

        Returns
        -------
        images : array-like shape (n_images, n_colors, image_width, image_height)
            Images normalized to have pixel data range [0,1].

        '''
        def read(fname):
            im = Image.open(fname)
            im = np.float32(im)
            return im/255.
        x_all = np.array([read(fname) for fname in tqdm.tqdm(filepaths)])
        x_all = x_all.astype('float32')
        if self.mode == 'convolution':
            x_all = x_all.transpose(0, 3, 1, 2)
        print("Image Files Loaded!")
        return x_all

    def fit(
        self,
        img_data,
        save_freq=-1,
        pic_freq=-1,

        n_epochs=100,
        batch_size=50,
        weight_decay=True,
        model_path='./VAE_training_model/',
        img_path='./VAE_training_images/',
        img_out_width=10
    ):

        '''Fit the VAE model to the image data.

        Parameters
        ----------

        img_data : array-like shape (n_images, n_colors, image_width, image_height)
            Images used to fit VAE model.
        save_freq [optional] : int
            Sets the number of epochs to wait before saving the model and optimizer states.
            Also saves image files of randomly generated images using those states in a
            separate directory. Does not save if negative valued.
        pic_freq [optional] : int
            Sets the number of batches to wait before displaying a picture or randomly
            generated images using the current model state.
            Does not display if negative valued.
        n_epochs [optional] : int
            Gives the number of training epochs to run through for the fitting
            process.
        batch_size [optional] : int
            The size of the batch to use when training. Note: generally larger
            batch sizes will result in fater epoch iteration, but at the const
            of lower granulatity when updating the layer weights.
        weight_decay [optional] : bool
            Flag that controls adding weight decay hooks to the optimizer.
        model_path [optional] : str
            Directory where the model and optimizer state files will be saved.
        img_path [optional] : str
            Directory where the end of epoch training image files will be saved.
        img_out_width : int
            Controls the number of randomly genreated images per row in the output
            saved imags.
        '''
        width = img_out_width
        self.opt.setup(self.model)

        if weight_decay:
            self.opt.add_hook(chainer.optimizer.WeightDecay(0.00001))

        n_data = img_data.shape[0]

        batch_iter = list(range(0, n_data, batch_size))
        n_batches = len(batch_iter)
        save_counter = 0
        for epoch in range(1, n_epochs + 1):
            print('epoch: %i' % epoch)
            t1 = time.time()
            indexes = np.random.permutation(n_data)
            last_loss_kl = 0.
            last_loss_rec = 0.
            count = 0
            for i in tqdm.tqdm(batch_iter):

                x_batch = Variable(img_data[indexes[i: i + batch_size]])

                if self.flag_gpu:
                    x_batch.to_gpu()

                out, kl_loss, rec_loss = self.model.forward(x_batch)
                total_loss = rec_loss + kl_loss*self.kl_ratio

                self.opt.zero_grads()
                total_loss.backward()
                self.opt.update()

                last_loss_kl += kl_loss.data
                last_loss_rec += rec_loss.data
                plot_pics = Variable(img_data[indexes[:width]])
                count += 1
                if pic_freq > 0:
                    assert type(pic_freq) == int, "pic_freq must be an integer."
                    if count % pic_freq == 0:
                        fig = self._plot_img(
                            plot_pics,
                            img_path=img_path,
                            epoch=epoch
                        )
                        display(fig)

            if save_freq > 0:
                save_counter += 1
                assert type(save_freq) == int, "save_freq must be an integer."
                if epoch % save_freq == 0:
                    name = "vae_epoch%s" % str(epoch)
                    if save_counter == 1:
                        save_meta = True
                    else:
                        save_meta = False
                    self.save(model_path, name, save_meta=save_meta)
                    fig = self._plot_img(
                        plot_pics,
                        img_path=img_path,
                        epoch=epoch,
                        batch=n_batches,
                        save=True
                        )

            msg = "rec_loss = {0} , kl_loss = {1}"
            print(msg.format(last_loss_rec/n_batches, last_loss_kl/n_batches))
            t_diff = time.time()-t1
            print("time: %f\n\n" % t_diff)

    def _plot_img(self, data, img_path='./', epoch=1, batch=1, save=False):

        if data.data.shape[0] < 10:
            width = data.data.shape[0]
        else:
            width = 10
        x = Variable(data.data[:width])
        if self.flag_gpu:
            x.to_gpu()
        rec = self.model.forward(x)[0]
        rec.to_cpu()
        x.to_cpu()
        fig = plt.figure(figsize=(16.0, 3.0))

        orig = x.data.transpose(0, 2, 3, 1)
        rec = rec.data.transpose(0, 2, 3, 1)

        for i in range(width):
            plt.subplot(2, width, i+1)
            plt.imshow(orig[i])
            plt.axis("off")
        for i in range(width):
            plt.subplot(2, width, width+i+1)
            plt.imshow(rec[i])
            plt.axis("off")
        if save:
            if img_path[-1] != '/':
                img_path += '/'
            if not os.path.exists(os.path.dirname(img_path)):
                os.makedirs(os.path.dirname(img_path))
            save_str = 'image_vae_epoch%d_batch%d.png' % (epoch, batch)
            plt.savefig(os.path.join(img_path, save_str))
        plt.close()
        return fig

    def save(self, path, name, save_meta=True):
        '''Saves model as a sequence of files in the format:
            {path}/{name}_{'model', 'opt', 'meta'}.h5

        Parameters
        ----------
        path : str
            The directory of the file you wish to save the model to.
        name : str
            The name prefix of the model and optimizer files you wish
            to save.
        save_meta [optional] : bool
            Flag that controls whether to save the class metadata along with
            the encoder, decoder, and respective optimizer states.
        '''
        _save_model(self.model, str(path), "%s_model" % str(name))
        _save_model(self.opt, str(path), "%s_opt" % str(name))
        if save_meta:
            self._save_meta(os.path.join(path, "%s_meta" % str(name)))

    @classmethod
    def load(cls, model, opt, meta, flag_gpu=None):
        '''Loads in model as a class instance with with the specified
           model and optimizer states.

        Parameters
        ----------
        model : str
            Path to the model state file.
        opt : str
            Path to the optimizer state file.
        meta : str
            Path to the class metadata state file.
        flag_gpu : bool
            Specifies whether to load the model to use gpu capabilities.

        Returns
        -------

        class instance of self.
        '''
        mess = "Model file {0} does not exist. Please check the file path."
        assert os.path.exists(model), mess.format(model)
        assert os.path.exists(opt), mess.format(opt)
        assert os.path.exists(meta), mess.format(meta)
        with open(meta, 'r') as f:
            meta = json.load(f)
        if flag_gpu is not None:
            meta['flag_gpu'] = flag_gpu

        loaded_class = cls(**meta)

        serializers.load_hdf5(model, loaded_class.model)
        loaded_class.opt.setup(loaded_class.model)
        serializers.load_hdf5(opt, loaded_class.opt)

        if meta['flag_gpu']:
            loaded_class.model = loaded_class.model.to_gpu()

        return loaded_class


class GAN(object):
    '''Generative Adversarial Networks Class for image data.

    Using linear/convolutional transformations with ReLU activations, this class
    uses a system of adversarial generator and discriminator networks to generate
    images from latent space representations.

    See  Radford, Alec et. al.; "Unsupervised Representation Learning with Deep
    Convolutional Generative Adversarial Networks"; (2015).

    The generator network is trained to generate images that resemble the input
    data while the discriminator is simultaneously trained to discern the difference
    between ground truth images in the training set and images made by the generator.

    Attributes
    ----------
    decode_layers : List[int]
        List of layer sizes for hidden linear decoding layers of the model.
        Only taken into account when mode='linear'.
    disc_layers : List[int]
        List of layer sizes for hidden linear discriminator layers of the model.
        Only taken into account when mode='linear'.
    latent_width : int
        Dimension of latent encoding space.
    img_width : int
        Width of the desired image representation.
    img_height : int
        Height of the desired image representation.
    color_channels : int
        Number of color channels in the input images.
    kl_ratio : float
            Multiplicative factor on and KL Divergence term.
    flag_gpu : bool
        Flag to toggle GPU training functionality.
    mode: str
        Mode to set the encoder and decoder architectures. Can be either
        'convolution' or 'linear'.
    dec_adam_alpha : float
        Alpha parameter for the adam optimizer training the generator.
    dec_adam_beta1 : float
        Beta1 parameter for the adam optimizer training the generator.
    disc_adam_alpha : float
        Alpha parameter for the adam optimizer training the discriminator.
    disc_adam_beta1 : float
        Beta1 parameter for the adam optimizer training the discriminator.
    rectifier : str
        Rectifier option for the output of the decoder. Can be either
        'clipped_relu' or 'sigmoid'.
    dropout_ratio : float
        Specifies the dropout probability for the convolutional discriminator
        layers. Range is [0,1].
    dec : chainer.Chain
        Chain of chainer links for the generator network.
    disc : chainer.Chain
        Chain of chainer links for the discriminator network.
    dec_opt : chainer.Optimizer
        Chiner optimizer used to do backpropagation on the generator.
        Set to Adam.
    disc_opt : chainer.Optimizer
        Chiner optimizer used to do backpropagation on the discriminator.
        Set to Adam.
    '''
    def __init__(self, img_width=64, img_height=64, color_channels=3,
                 decode_layers=[300, 800, 1000],
                 disc_layers=[1000, 600, 300],
                 latent_width=100, flag_gpu=True,
                 mode='convolution', dec_adam_alpha=0.0002, dec_adam_beta1=0.5,
                 disc_adam_alpha=0.0002, disc_adam_beta1=0.5, rectifier='clipped_relu',
                 dropout_ratio=0.5):
        self.img_width = img_width
        self.img_height = img_height
        self.color_channels = color_channels
        self.decode_layers = decode_layers
        self.disc_layers = disc_layers
        self.latent_width = latent_width
        self.flag_gpu = flag_gpu
        self.mode = mode
        self.dec_adam_alpha = dec_adam_alpha
        self.dec_adam_beta1 = dec_adam_beta1
        self.disc_adam_alpha = disc_adam_alpha
        self.disc_adam_beta1 = disc_adam_beta1
        self.rectifier = rectifier
        self.dropout_ratio = dropout_ratio
        if self.mode == 'convolution':
            self._check_dims()

        self.dec = Decoder(img_width=self.img_width,
                           img_height=self.img_height,
                           color_channels=self.color_channels,
                           decode_layers=self.decode_layers,
                           latent_width=self.latent_width,
                           mode=self.mode)
        self.disc = Discriminator(img_width=self.img_width,
                                  img_height=self.img_height,
                                  color_channels=self.color_channels,
                                  disc_layers=self.disc_layers,
                                  latent_width=self.latent_width,
                                  mode=self.mode)
        if self.flag_gpu:
            self.dec = self.dec.to_gpu()
            self.disc = self.disc.to_gpu()

        self.dec_opt = O.Adam(alpha=self.dec_adam_alpha, beta1=self.dec_adam_beta1)
        self.disc_opt = O.Adam(alpha=self.disc_adam_alpha, beta1=self.disc_adam_beta1)

    def _check_dims(self):
        h, w = calc_fc_size(self.img_height, self.img_width)[1:]
        h, w = calc_im_size(h, w)

        assert (h == self.img_height) and (w == self.img_width),\
            "To use convolution, please resize images " + \
            "to nearest target height, width = %d, %d" % (h, w)

    def _save_meta(self, filepath):
        if not os.path.exists(os.path.dirname(filepath)):
            os.makedirs(os.path.dirname(filepath))
        d = self.__dict__.copy()
        poplist = ['dec', 'disc', 'dec_opt', 'disc_opt']
        for name in poplist:
            d.pop(name)
        meta = json.dumps(d)
        with open(filepath+'.json', 'wb') as f:
            f.write(meta)

    def _forward(self, batch):
        shape = (batch.data.shape[0], self.latent_width)
        samp = np.random.standard_normal(shape).astype('float32')
        samp = Variable(samp)
        if self.flag_gpu:
            samp.to_gpu()

        decoded = self.dec(samp, rectifier=self.rectifier)
        disc_samp = self.disc(decoded, dropout_ratio=self.dropout_ratio)[0]

        disc_batch = self.disc(batch, dropout_ratio=self.dropout_ratio)[0]

        return disc_samp, disc_batch

    def inverse_transform(self, data, test=False):
        '''Transform latent vectors into images.

        Parameters
        ----------
        data : array-like shape (n_images, latent_width)
            Input numpy array of images.
        test [optional] : bool
            Controls the test boolean for batch normalization.

        Returns
        -------
        images : array-like shape (n_images, image_width, image_height,
                                   n_colors)
        '''
        if not type(data) == Variable:
            if len(data.shape) < 2:
                data = data[np.newaxis]
            if len(data.shape) != 2:
                raise TypeError("Invalid dimensions for latent data. Dim = %s.\
                     Must be a 2d array." % str(data.shape))
            data = Variable(data)

        else:
            if len(data.data.shape) < 2:
                data.data = data.data[np.newaxis]
            if len(data.data.shape) != 2:
                raise TypeError("Invalid dimensions for latent data. Dim = %s.\
                     Must be a 2d array." % str(data.data.shape))
        assert data.data.shape[-1] == self.latent_width,\
            "Latent shape %d != %d" % (data.data.shape[-1], self.latent_width)

        if self.flag_gpu:
            data.to_gpu()
        out = self.dec(data, test=test, rectifier=self.rectifier)

        out.to_cpu()

        if self.mode == 'linear':
            final = out.data
        else:
            final = out.data.transpose(0, 2, 3, 1)

        return final

    def load_images(self, filepaths):
        '''Load in image files from list of paths.

        Parameters
        ----------

        filepaths : List[str]
            List of file paths of images to be loaded.

        Returns
        -------
        images : array-like shape (n_images, n_colors, image_width, image_height)
            Images normalized to have pixel data range [0,1].

        '''
        def read(fname):
            im = Image.open(fname)
            im = np.float32(im)
            return im/255.
        x_all = np.array([read(fname) for fname in tqdm.tqdm(filepaths)])
        x_all = x_all.astype('float32')
        if self.mode == 'convolution':
            x_all = x_all.transpose(0, 3, 1, 2)
        print("Image Files Loaded!")
        return x_all

    def fit(
        self,
        img_data,
        save_freq=-1,
        pic_freq=-1,
        n_epochs=100,
        batch_size=50,
        weight_decay=True,
        model_path='./GAN_training_model/',
        img_path='./GAN_training_images/',
        img_out_width=10,
        mirroring=False
    ):
        '''Fit the GAN model to the image data.

        Parameters
        ----------

        img_data : array-like shape (n_images, n_colors, image_width, image_height)
            Images used to fit VAE model.
        save_freq [optional] : int
            Sets the number of epochs to wait before saving the model and optimizer states.
            Also saves image files of randomly generated images using those states in a
            separate directory. Does not save if negative valued.
        pic_freq [optional] : int
            Sets the number of batches to wait before displaying a picture or randomly
            generated images using the current model state.
            Does not display if negative valued.
        n_epochs [optional] : int
            Gives the number of training epochs to run through for the fitting
            process.
        batch_size [optional] : int
            The size of the batch to use when training. Note: generally larger
            batch sizes will result in fater epoch iteration, but at the const
            of lower granulatity when updating the layer weights.
        weight_decay [optional] : bool
            Flag that controls adding weight decay hooks to the optimizer.
        model_path [optional] : str
            Directory where the model and optimizer state files will be saved.
        img_path [optional] : str
            Directory where the end of epoch training image files will be saved.
        img_out_width : int
            Controls the number of randomly genreated images per row in the output
            saved imags.
        mirroring [optional] : bool
            Controls whether images are randomly mirrored along the verical axis with
            a .5 probability. Artificially increases images variance for training set.
        '''
        width = img_out_width
        self.dec_opt.setup(self.dec)
        self.disc_opt.setup(self.disc)

        if weight_decay:
            self.dec_opt.add_hook(chainer.optimizer.WeightDecay(0.00001))
            self.disc_opt.add_hook(chainer.optimizer.WeightDecay(0.00001))

        n_data = img_data.shape[0]

        batch_iter = list(range(0, n_data, batch_size))
        n_batches = len(batch_iter)

        c_samples = np.random.standard_normal((width, self.latent_width)).astype(np.float32)
        save_counter = 0

        for epoch in range(1, n_epochs + 1):
            print('epoch: %i' % epoch)
            t1 = time.time()
            indexes = np.random.permutation(n_data)
            last_loss_dec = 0.
            last_loss_disc = 0.
            count = 0
            for i in tqdm.tqdm(batch_iter):
                x = img_data[indexes[i: i + batch_size]]
                size = x.shape[0]
                if mirroring:
                    for j in range(size):
                        if np.random.randint(2):
                            x[j, :, :, :] = x[j, :, :, ::-1]
                x_batch = Variable(x)
                zeros = Variable(np.zeros(size, dtype=np.int32))
                ones = Variable(np.ones(size, dtype=np.int32))

                if self.flag_gpu:
                    x_batch.to_gpu()
                    zeros.to_gpu()
                    ones.to_gpu()

                disc_samp, disc_batch = self._forward(x_batch)

                L_dec = F.softmax_cross_entropy(disc_samp, ones)

                L_disc = F.softmax_cross_entropy(disc_samp, zeros)
                L_disc += F.softmax_cross_entropy(disc_batch, ones)
                L_disc /= 2.

                self.dec_opt.zero_grads()
                L_dec.backward()
                self.dec_opt.update()

                self.disc_opt.zero_grads()
                L_disc.backward()
                self.disc_opt.update()

                last_loss_dec += L_dec.data
                last_loss_disc += L_disc.data
                count += 1
                if pic_freq > 0:
                    assert type(pic_freq) == int, "pic_freq must be an integer."
                    if count % pic_freq == 0:
                        fig = self._plot_img(
                            c_samples,
                            img_path=img_path,
                            epoch=epoch
                        )
                        display(fig)

            if save_freq > 0:
                save_counter += 1
                assert type(save_freq) == int, "save_freq must be an integer."
                if epoch % save_freq == 0:
                    name = "gan_epoch%s" % str(epoch)
                    if save_counter == 1:
                        save_meta = True
                    else:
                        save_meta = False
                    self.save(model_path, name, save_meta=save_meta)
                    fig = self._plot_img(
                        c_samples,
                        img_path=img_path,
                        epoch=epoch,
                        batch=n_batches,
                        save_pic=True
                        )

            msg = "dec_loss = {0} , disc_loss = {1}"
            print(msg.format(last_loss_dec/n_batches, last_loss_disc/n_batches))
            t_diff = time.time()-t1
            print("time: %f\n\n" % t_diff)

    def _plot_img(self, samples, img_path='./', epoch=1, batch=1, save_pic=False):

        if samples.shape[0] < 10:
            width = samples.shape[0]
        else:
            width = 10

        x = Variable(samples[:width])
        y = Variable(np.random.standard_normal((width, self.latent_width)).astype(np.float32))
        if self.flag_gpu:
            x.to_gpu()
            y.to_gpu()
        x_pics = self.dec(x, rectifier=self.rectifier)
        y_pics = self.dec(y, rectifier=self.rectifier)
        x_pics.to_cpu()
        y_pics.to_cpu()

        fig = plt.figure(figsize=(16.0, 3.0))

        x_pics = x_pics.data.transpose(0, 2, 3, 1)
        y_pics = y_pics.data.transpose(0, 2, 3, 1)

        for i in range(width):
            plt.subplot(2, width, i+1)
            plt.imshow(x_pics[i])
            plt.axis("off")
        for i in range(width):
            plt.subplot(2, width, width+i+1)
            plt.imshow(y_pics[i])
            plt.axis("off")
        if save_pic:
            if img_path[-1] != '/':
                img_path += '/'
            if not os.path.exists(os.path.dirname(img_path)):
                os.makedirs(os.path.dirname(img_path))
            save_str = 'image_gan_epoch%d_batch%d.png' % (epoch, batch)
            plt.savefig(os.path.join(img_path, save_str))
        plt.close()
        return fig

    def save(self, path, name, save_meta=True):
        '''Saves model as a sequence of files in the format:
            {path}/{name}_{'dec', 'disc', 'dec_opt',
            'disc_opt', 'meta'}.h5

        Parameters
        ----------
        path : str
            The directory of the file you wish to save the model to.
        name : str
            The name prefix of the model and optimizer files you wish
            to save.
        save_meta [optional] : bool
            Flag that controls whether to save the class metadata along with
            the generator, discriminator, and respective optimizer states.
        '''
        _save_model(self.dec, str(path), "%s_dec" % str(name))
        _save_model(self.disc, str(path), "%s_disc" % str(name))
        _save_model(self.dec_opt, str(path), "%s_dec_opt" % str(name))
        _save_model(self.disc_opt, str(path), "%s_disc_opt" % str(name))
        if save_meta:
            self._save_meta(os.path.join(path, "%s_meta" % str(name)))

    @classmethod
    def load(cls, dec, disc, dec_opt, disc_opt, meta, flag_gpu=None):
        '''Loads in model as a class instance with with the specified
           model and optimizer states.

        Parameters
        ----------
        dec : str
            Path to the generator state file.
        disc : str
            Path to the discriminator state file.
        dec_opt : str
            Path to the generator optimizer state file.
        disc_opt : str
            Path to the discriminator optimizer state file.
        meta : str
            Path to the class metadata state file.
        flag_gpu : bool
            Specifies whether to load the model to use gpu capabilities.

        Returns
        -------

        class instance of self.
        '''
        mess = "Model file {0} does not exist. Please check the file path."
        assert os.path.exists(dec), mess.format(dec)
        assert os.path.exists(disc), mess.format(disc)
        assert os.path.exists(dec_opt), mess.format(dec_opt)
        assert os.path.exists(disc_opt), mess.format(disc_opt)
        assert os.path.exists(meta), mess.format(meta)
        with open(meta, 'r') as f:
            meta = json.load(f)
        if flag_gpu is not None:
            meta['flag_gpu'] = flag_gpu

        loaded_class = cls(**meta)

        serializers.load_hdf5(dec, loaded_class.dec)
        serializers.load_hdf5(disc, loaded_class.disc)
        loaded_class.dec_opt.setup(loaded_class.dec)
        loaded_class.disc_opt.setup(loaded_class.disc)
        serializers.load_hdf5(dec_opt, loaded_class.dec_opt)
        serializers.load_hdf5(disc_opt, loaded_class.disc_opt)

        if meta['flag_gpu']:
            loaded_class.dec.to_gpu()
            loaded_class.disc.to_gpu()

        return loaded_class


class VAEGAN(object):
    '''Variational Auto-encoding Generative Adversarial Networks Class for image data.

    Using linear/convolutional transformations with ReLU activations, this class
    pairs the finctionality of both Varitational Auto-encoders and Adversarial Networks
    in order to create an adversarially trained network that is capable of encoding images
    to latent representations.

    See Boesen Lindbo Larsen, Anders et. al.; "Autoencoding Beyond Pixels Using a Learned
    Similarity Metric"; (2015). And personal work by TJ Torres on network architecture
    and hyperparameters.

    Here the generator network is trained to generate images that resemble the input
    data while the discriminator is simultaneously trained to discern the difference
    between ground truth images in the training set and images made by the generator.
    With the added caveat that reconstructed auto-encoded images are also used to
    train the entire encoder, generator, and discriminator netowrk.

    Attributes
    ----------
    encode_layers : List[int]
        List of layer sizes for hidden linear encoding layers of the model.
        Only taken into account when mode='linear'.
    decode_layers : List[int]
        List of layer sizes for hidden linear decoding layers of the model.
        Only taken into account when mode='linear'.
    disc_layers : List[int]
        List of layer sizes for hidden linear discriminator layers of the model.
        Only taken into account when mode='linear'.
    kl_ratio : float
        Sets the multiplicative factor on the kl divergence term used to regularize
        the encoder training.
    latent_width : int
        Dimension of latent encoding space.
    img_width : int
        Width of the desired image representation.
    img_height : int
        Height of the desired image representation.
    color_channels : int
        Number of color channels in the input images.
    kl_ratio : float
            Multiplicative factor on and KL Divergence term.
    flag_gpu : bool
        Flag to toggle GPU training functionality.
    mode: str
        Mode to set the encoder and decoder architectures. Can be either
        'convolution' or 'linear'.
    enc_adam_alpha : float
        Alpha parameter for the adam optimizer training the encoder.
    enc_adam_beta1 : float
        Beta1 parameter for the adam optimizer training the encoder.
    dec_adam_alpha : float
        Alpha parameter for the adam optimizer training the decoder/generator.
    dec_adam_beta1 : float
        Beta1 parameter for the adam optimizer training the decoder/generator.
    disc_adam_alpha : float
        Alpha parameter for the adam optimizer training the discriminator.
    disc_adam_beta1 : float
        Beta1 parameter for the adam optimizer training the discriminator.
    rectifier : str
        Rectifier option for the output of the decoder. Can be either
        'clipped_relu' or 'sigmoid'.
    dropout_ratio : float
        Specifies the dropout probability for the convolutional discriminator
        layers. Range is [0,1].
    enc : chainer.Chain
        Chain of chainer links for the encoder network.
    dec : chainer.Chain
        Chain of chainer links for the decoder/generator network.
    disc : chainer.Chain
        Chain of chainer links for the discriminator network.
    enc_opt : chainer.Optimizer
        Chiner optimizer used to do backpropagation on the encoder.
        Set to Adam.
    dec_opt : chainer.Optimizer
        Chiner optimizer used to do backpropagation on the decoder/generator.
        Set to Adam.
    disc_opt : chainer.Optimizer
        Chiner optimizer used to do backpropagation on the discriminator.
        Set to Adam.
    '''
    def __init__(self, img_width=64, img_height=64, color_channels=3,
                 encode_layers=[1000, 600, 300],
                 decode_layers=[300, 800, 1000],
                 disc_layers=[1000, 600, 300],
                 kl_ratio=1.0,
                 latent_width=500, flag_gpu=True,
                 mode='convolution',
                 enc_adam_alpha=0.0002, enc_adam_beta1=0.5,
                 dec_adam_alpha=0.0002, dec_adam_beta1=0.5,
                 disc_adam_alpha=0.0001, disc_adam_beta1=0.5,
                 rectifier='clipped_relu', dropout_ratio=0.5):
        self.img_width = img_width
        self.img_height = img_height
        self.color_channels = color_channels
        self.encode_layers = encode_layers
        self.decode_layers = decode_layers
        self.disc_layers = disc_layers
        self.kl_ratio = kl_ratio
        self.latent_width = latent_width
        self.flag_gpu = flag_gpu
        self.mode = mode
        self.enc_adam_alpha = enc_adam_alpha
        self.enc_adam_beta1 = enc_adam_beta1
        self.dec_adam_alpha = dec_adam_alpha
        self.dec_adam_beta1 = dec_adam_beta1
        self.disc_adam_alpha = disc_adam_alpha
        self.disc_adam_beta1 = disc_adam_beta1
        self.rectifier = rectifier
        self.dropout_ratio = dropout_ratio
        if self.mode == 'convolution':
            self._check_dims()
        self.enc = Encoder(img_width=self.img_width,
                           img_height=self.img_height,
                           color_channels=self.color_channels,
                           encode_layers=self.encode_layers,
                           latent_width=self.latent_width,
                           mode=self.mode)
        self.dec = Decoder(img_width=self.img_width,
                           img_height=self.img_height,
                           color_channels=self.color_channels,
                           decode_layers=self.decode_layers,
                           latent_width=self.latent_width,
                           mode=self.mode)
        self.disc = Discriminator(img_width=self.img_width,
                                  img_height=self.img_height,
                                  color_channels=self.color_channels,
                                  disc_layers=self.disc_layers,
                                  latent_width=self.latent_width,
                                  mode=self.mode)
        if self.flag_gpu:
            self.enc = self.enc.to_gpu()
            self.dec = self.dec.to_gpu()
            self.disc = self.disc.to_gpu()

        self.enc_opt = O.Adam(alpha=self.enc_adam_alpha, beta1=self.enc_adam_beta1)
        self.dec_opt = O.Adam(alpha=self.dec_adam_alpha, beta1=self.dec_adam_beta1)
        self.disc_opt = O.Adam(alpha=self.disc_adam_alpha, beta1=self.disc_adam_beta1)

    def _check_dims(self):
        h, w = calc_fc_size(self.img_height, self.img_width)[1:]
        h, w = calc_im_size(h, w)

        assert (h == self.img_height) and (w == self.img_width),\
            "To use convolution, please resize images " + \
            "to nearest target height, width = %d, %d" % (h, w)

    def _save_meta(self, filepath):
        if not os.path.exists(os.path.dirname(filepath)):
            os.makedirs(os.path.dirname(filepath))
        d = self.__dict__.copy()
        poplist = ['enc', 'dec', 'disc',
                   'enc_opt', 'dec_opt', 'disc_opt',
                   ]
        for name in poplist:
            d.pop(name)
        meta = json.dumps(d)
        with open(filepath+'.json', 'wb') as f:
            f.write(meta)

    def _encode(self, data, test=False):
        x = self.enc(data, test=test)
        mean, ln_var = F.split_axis(x, 2, 1)
        samp = np.random.standard_normal(mean.data.shape).astype('float32')
        samp = Variable(samp)
        if self.flag_gpu:
            samp.to_gpu()
        z = samp * F.exp(0.5*ln_var) + mean

        return z, mean, ln_var

    def _decode(self, z, test=False):
        x = self.dec(z, test=test, rectifier=self.rectifier)

        return x

    def _forward(self, batch, test=False):

        encoded, means, ln_vars = self._encode(batch, test=test)
        rec = self._decode(encoded, test=test)
        normer = reduce(lambda x, y: x*y, means.data.shape)
        kl_loss = F.gaussian_kl_divergence(means, ln_vars)/normer

        samp_p = np.random.standard_normal(means.data.shape).astype('float32')
        z_p = chainer.Variable(samp_p)

        if self.flag_gpu:
            z_p.to_gpu()

        rec_p = self._decode(z_p)

        disc_rec, conv_layer_rec = self.disc(rec, test=test, dropout_ratio=self.dropout_ratio)

        disc_batch, conv_layer_batch = self.disc(batch, test=test, dropout_ratio=self.dropout_ratio)

        disc_x_p, conv_layer_x_p = self.disc(rec_p, test=test, dropout_ratio=self.dropout_ratio)

        dif_l = F.mean_squared_error(conv_layer_rec, conv_layer_batch)

        return kl_loss, dif_l, disc_rec, disc_batch, disc_x_p

    def transform(self, data, test=False):
        '''Transform image data to latent space.

        Parameters
        ----------
        data : array-like shape (n_images, image_width, image_height,
                                   n_colors)
            Input numpy array of images.
        test [optional] : bool
            Controls the test boolean for batch normalization.

        Returns
        -------
        latent_vec : array-like shape (n_images, latent_width)
        '''
        #make sure that data has the right shape.
        if not type(data) == Variable:
            if len(data.shape) < 4:
                data = data[np.newaxis]
            if len(data.shape) != 4:
                raise TypeError("Invalid dimensions for image data. Dim = %s.\
                     Must be 4d array." % str(data.shape))
            if data.shape[1] != self.color_channels:
                if data.shape[-1] == self.color_channels:
                    data = data.transpose(0, 3, 1, 2)
                else:
                    raise TypeError("Invalid dimensions for image data. Dim = %s"
                                    % str(data.shape))
            data = Variable(data)
        else:
            if len(data.data.shape) < 4:
                data.data = data.data[np.newaxis]
            if len(data.data.shape) != 4:
                raise TypeError("Invalid dimensions for image data. Dim = %s.\
                     Must be 4d array." % str(data.data.shape))
            if data.data.shape[1] != self.color_channels:
                if data.data.shape[-1] == self.color_channels:
                    data.data = data.data.transpose(0, 3, 1, 2)
                else:
                    raise TypeError("Invalid dimensions for image data. Dim = %s"
                                    % str(data.shape))

        # Actual transformation.
        if self.flag_gpu:
            data.to_gpu()
        z = self._encode(data, test=test)[0]

        z.to_cpu()

        return z.data

    def inverse_transform(self, data, test=False):
        '''Transform latent vectors into images.

        Parameters
        ----------
        data : array-like shape (n_images, latent_width)
            Input numpy array of images.
        test [optional] : bool
            Controls the test boolean for batch normalization.

        Returns
        -------
        images : array-like shape (n_images, image_width, image_height, n_colors)
        '''
        if not type(data) == Variable:
            if len(data.shape) < 2:
                data = data[np.newaxis]
            if len(data.shape) != 2:
                raise TypeError("Invalid dimensions for latent data. Dim = %s.\
                     Must be a 2d array." % str(data.shape))
            data = Variable(data)

        else:
            if len(data.data.shape) < 2:
                data.data = data.data[np.newaxis]
            if len(data.data.shape) != 2:
                raise TypeError("Invalid dimensions for latent data. Dim = %s.\
                     Must be a 2d array." % str(data.data.shape))
        assert data.data.shape[-1] == self.latent_width,\
            "Latent shape %d != %d" % (data.data.shape[-1], self.latent_width)

        if self.flag_gpu:
            data.to_gpu()
        out = self._decode(data, test=test)

        out.to_cpu()

        if self.mode == 'linear':
            final = out.data
        else:
            final = out.data.transpose(0, 2, 3, 1)

        return final

    def load_images(self, filepaths):
        '''Load in image files from list of paths.

        Parameters
        ----------

        filepaths : List[str]
            List of file paths of images to be loaded.

        Returns
        -------
        images : array-like shape (n_images, n_colors, image_width, image_height)
            Images normalized to have pixel data range [0,1].

        '''
        def read(fname):
            im = Image.open(fname)
            im = np.float32(im)
            return im/255.
        x_all = np.array([read(fname) for fname in tqdm.tqdm(filepaths)])
        x_all = x_all.astype('float32')
        if self.mode == 'convolution':
            x_all = x_all.transpose(0, 3, 1, 2)
        print("Image Files Loaded!")
        return x_all

    def fit(
        self,
        img_data,
        gamma=1.0,
        save_freq=-1,
        pic_freq=-1,
        n_epochs=100,
        batch_size=50,
        weight_decay=True,
        model_path='./VAEGAN_training_model/',
        img_path='./VAEGAN_training_images/',
        img_out_width=10,
        mirroring=False
    ):
        '''Fit the VAE/GAN model to the image data.

        Parameters
        ----------

        img_data : array-like shape (n_images, n_colors, image_width, image_height)
            Images used to fit VAE model.
        gamma [optional] : float
            Sets the multiplicative factor that weights the relative importance of
            reconstruction loss vs. ability to fool the discriminator. Higher weight
            means greater focus on faithful reconstruction.
        save_freq [optional] : int
            Sets the number of epochs to wait before saving the model and optimizer states.
            Also saves image files of randomly generated images using those states in a
            separate directory. Does not save if negative valued.
        pic_freq [optional] : int
            Sets the number of batches to wait before displaying a picture or randomly
            generated images using the current model state.
            Does not display if negative valued.
        n_epochs [optional] : int
            Gives the number of training epochs to run through for the fitting
            process.
        batch_size [optional] : int
            The size of the batch to use when training. Note: generally larger
            batch sizes will result in fater epoch iteration, but at the const
            of lower granulatity when updating the layer weights.
        weight_decay [optional] : bool
            Flag that controls adding weight decay hooks to the optimizer.
        model_path [optional] : str
            Directory where the model and optimizer state files will be saved.
        img_path [optional] : str
            Directory where the end of epoch training image files will be saved.
        img_out_width : int
            Controls the number of randomly genreated images per row in the output
            saved imags.
        mirroring [optional] : bool
            Controls whether images are randomly mirrored along the verical axis with
            a .5 probability. Artificially increases images variance for training set.
        '''
        width = img_out_width
        self.enc_opt.setup(self.enc)
        self.dec_opt.setup(self.dec)
        self.disc_opt.setup(self.disc)

        if weight_decay:
            self.enc_opt.add_hook(chainer.optimizer.WeightDecay(0.00001))
            self.dec_opt.add_hook(chainer.optimizer.WeightDecay(0.00001))
            self.disc_opt.add_hook(chainer.optimizer.WeightDecay(0.00001))

        n_data = img_data.shape[0]

        batch_iter = list(range(0, n_data, batch_size))
        n_batches = len(batch_iter)

        c_samples = np.random.standard_normal((width, self.latent_width)).astype(np.float32)
        save_counter = 0

        for epoch in range(1, n_epochs + 1):
            print('epoch: %i' % epoch)
            t1 = time.time()
            indexes = np.random.permutation(n_data)
            sum_l_enc = 0.
            sum_l_dec = 0.
            sum_l_disc = 0.

            sum_l_gan = 0.
            sum_l_like = 0.
            sum_l_prior = 0.
            count = 0
            for i in tqdm.tqdm(batch_iter):
                x = img_data[indexes[i: i + batch_size]]
                size = x.shape[0]
                if mirroring:
                    for j in range(size):
                        if np.random.randint(2):
                            x[j, :, :, :] = x[j, :, :, ::-1]
                x_batch = Variable(x)
                zeros = Variable(np.zeros(size, dtype=np.int32))
                ones = Variable(np.ones(size, dtype=np.int32))

                if self.flag_gpu:
                    x_batch.to_gpu()
                    zeros.to_gpu()
                    ones.to_gpu()

                kl_loss, dif_l, disc_rec, disc_batch, disc_samp = self._forward(x_batch)

                L_batch_GAN = F.softmax_cross_entropy(disc_batch, ones)
                L_rec_GAN = F.softmax_cross_entropy(disc_rec, zeros)
                L_samp_GAN = F.softmax_cross_entropy(disc_samp, zeros)

                l_gan = (L_batch_GAN + L_rec_GAN + L_samp_GAN)/3.
                l_like = dif_l
                l_prior = kl_loss

                enc_loss = self.kl_ratio*l_prior + l_like
                dec_loss = gamma*l_like - l_gan
                disc_loss = l_gan

                self.enc_opt.zero_grads()
                enc_loss.backward()
                self.enc_opt.update()

                self.dec_opt.zero_grads()
                dec_loss.backward()
                self.dec_opt.update()

                self.disc_opt.zero_grads()
                disc_loss.backward()
                self.disc_opt.update()

                sum_l_enc += enc_loss.data
                sum_l_dec += dec_loss.data
                sum_l_disc += disc_loss.data

                sum_l_gan += l_gan.data
                sum_l_like += l_like.data
                sum_l_prior += l_prior.data
                count += 1

                plot_data = img_data[indexes[:width]]
                if pic_freq > 0:
                    assert type(pic_freq) == int, "pic_freq must be an integer."
                    if count % pic_freq == 0:
                        fig = self._plot_img(
                            plot_data,
                            c_samples,
                            img_path=img_path,
                            epoch=epoch
                        )
                        display(fig)

            if save_freq > 0:
                save_counter += 1
                assert type(save_freq) == int, "save_freq must be an integer."
                if epoch % save_freq == 0:
                    name = "vaegan_epoch%s" % str(epoch)
                    if save_counter == 1:
                        save_meta = True
                    else:
                        save_meta = False
                    self.save(model_path, name, save_meta=save_meta)
                    fig = self._plot_img(
                        plot_data,
                        c_samples,
                        img_path=img_path,
                        epoch=epoch,
                        batch=n_batches,
                        save_pic=True
                        )
            sum_l_enc /= n_batches
            sum_l_dec /= n_batches
            sum_l_disc /= n_batches
            sum_l_gan /= n_batches
            sum_l_like /= n_batches
            sum_l_prior /= n_batches
            msg = "enc_loss = {0}, dec_loss = {1} , disc_loss = {2}"
            msg2 = "gan_loss = {0}, sim_loss = {1}, kl_loss = {2}"
            print(msg.format(sum_l_enc, sum_l_dec, sum_l_disc))
            print(msg2.format(sum_l_gan, sum_l_like, sum_l_prior))
            t_diff = time.time()-t1
            print("time: %f\n\n" % t_diff)

    def _plot_img(self, img_data, samples, img_path='./', epoch=1, batch=1, save_pic=False):

        if samples.shape[0] < 10:
            width = samples.shape[0]
        else:
            width = 10

        x = Variable(samples[:width])
        y = Variable(np.random.standard_normal((width, self.latent_width)).astype(np.float32))
        z = img_data[:width]
        if self.flag_gpu:
            x.to_gpu()
            y.to_gpu()
        x_pics = self._decode(x)
        y_pics = self.dec(y)
        z_data = self.transform(z)
        z_rec = self.inverse_transform(z_data)
        x_pics.to_cpu()
        y_pics.to_cpu()

        fig = plt.figure(figsize=(16.0, 6.0))

        x_pics = x_pics.data.transpose(0, 2, 3, 1)
        y_pics = y_pics.data.transpose(0, 2, 3, 1)
        z = z.transpose(0, 2, 3, 1)

        for i in range(width):
            plt.subplot(4, width, i+1)
            plt.imshow(z[i])
            plt.axis("off")
        for i in range(width):
            plt.subplot(4, width, width+i+1)
            plt.imshow(z_rec[i])
            plt.axis("off")
        for i in range(width):
            plt.subplot(4, width, 2*width+i+1)
            plt.imshow(x_pics[i])
            plt.axis("off")
        for i in range(width):
            plt.subplot(4, width, 3*width+i+1)
            plt.imshow(y_pics[i])
            plt.axis("off")
        if save_pic:
            if img_path[-1] != '/':
                img_path += '/'
            if not os.path.exists(os.path.dirname(img_path)):
                os.makedirs(os.path.dirname(img_path))
            save_str = 'image_gan_epoch%d_batch%d.png' % (epoch, batch)
            plt.savefig(os.path.join(img_path, save_str))
        plt.close()
        return fig

    def save(self, path, name, save_meta=True):
        '''Saves model as a sequence of files in the format:
            {path}/{name}_{'enc', 'dec', 'disc', 'enc_opt',
            'dec_opt', 'disc_opt', 'meta'}.h5

        Parameters
        ----------
        path : str
            The directory of the file you wish to save the model to.
        name : str
            The name prefix of the model and optimizer files you wish
            to save.
        save_meta [optional] : bool
            Flag that controls whether to save the class metadata along with
            the encoder, decoder, discriminator, and respective optimizer states.
        '''
        _save_model(self.enc, str(path), "%s_enc" % str(name))
        _save_model(self.dec, str(path), "%s_dec" % str(name))
        _save_model(self.disc, str(path), "%s_disc" % str(name))
        _save_model(self.enc_opt, str(path), "%s_enc_opt" % str(name))
        _save_model(self.dec_opt, str(path), "%s_dec_opt" % str(name))
        _save_model(self.disc_opt, str(path), "%s_disc_opt" % str(name))
        if save_meta:
            self._save_meta(os.path.join(path, "%s_meta" % str(name)))

    @classmethod
    def load(cls, enc, dec, disc, enc_opt, dec_opt, disc_opt, meta, flag_gpu=None):
        '''Loads in model as a class instance with with the specified
           model and optimizer states.

        Parameters
        ----------
        enc : str
            Path to the encoder state file.
        dec : str
            Path to the decoder/generator state file.
        disc : str
            Path to the discriminator state file.
        enc_opt : str
            Path to the encoder optimizer state file.
        dec_opt : str
            Path to the decoder/generator optimizer state file.
        disc_opt : str
            Path to the discriminator optimizer state file.
        meta : str
            Path to the class metadata state file.
        flag_gpu : bool
            Specifies whether to load the model to use gpu capabilities.

        Returns
        -------

        class instance of self.
        '''
        mess = "Model file {0} does not exist. Please check the file path."
        assert os.path.exists(enc), mess.format(enc)
        assert os.path.exists(dec), mess.format(dec)
        assert os.path.exists(disc), mess.format(disc)
        assert os.path.exists(dec_opt), mess.format(dec_opt)
        assert os.path.exists(disc_opt), mess.format(disc_opt)
        assert os.path.exists(meta), mess.format(meta)
        with open(meta, 'r') as f:
            meta = json.load(f)
        if flag_gpu is not None:
            meta['flag_gpu'] = flag_gpu

        loaded_class = cls(**meta)

        serializers.load_hdf5(enc, loaded_class.enc)
        serializers.load_hdf5(dec, loaded_class.dec)
        serializers.load_hdf5(disc, loaded_class.disc)
        loaded_class.enc_opt.setup(loaded_class.enc)
        loaded_class.dec_opt.setup(loaded_class.dec)
        loaded_class.disc_opt.setup(loaded_class.disc)
        serializers.load_hdf5(enc_opt, loaded_class.enc_opt)
        serializers.load_hdf5(dec_opt, loaded_class.dec_opt)
        serializers.load_hdf5(disc_opt, loaded_class.disc_opt)

        if meta['flag_gpu']:
            loaded_class.enc.to_gpu()
            loaded_class.dec.to_gpu()
            loaded_class.disc.to_gpu()

        return loaded_class


def _save_model(model, directory, name):
    if directory[-1] != '/':
        directory += '/'

    if not os.path.exists(os.path.dirname(directory)):
        os.makedirs(os.path.dirname(directory))

    save_path = os.path.join(directory, name)

    serializers.save_hdf5("%s.h5" % save_path, model)


def get_paths(directory):
    '''Gets all the paths of non-hidden files in a directory
       and returns a list of those paths.

    Parameters
    ----------
    directory : str
        The directory whose contents you wish to grab.

    Returns
    -------
    paths : List[str]
    '''
    fnames = [os.path.join(directory, f)
              for f in os.listdir(directory)
              if os.path.isfile(os.path.join(directory, f))
              and not f.startswith('.')]
    return fnames


def image_resize(file_paths, new_dir, width, height):
    '''Resizes all images with given paths to new dimensions.
        Uses up/downscaling with antialiasing.

    Parameters
    ----------
    file_paths : List[str]
        List of path strings for image to resize.
    new_dir : str
        Directory to place resized images.
    width : int
        Target width of new resized images.
    height : int
        Target height of new resized images.
    '''
    if new_dir[-1] != '/':
        new_dir += '/'

    if not os.path.exists(os.path.dirname(new_dir)):
        os.makedirs(os.path.dirname(new_dir))

    for f in tqdm.tqdm(file_paths):
        img = Image.open(f).resize((width, height), Image.ANTIALIAS).convert('RGB')
        new = os.path.join(new_dir, os.path.basename(f))
        img.save(new)
