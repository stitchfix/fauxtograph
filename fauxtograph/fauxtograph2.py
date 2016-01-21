from PIL import Image
from chainer import cuda
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
            self.xp = cuda.cupy
            self.model = self.model.to_gpu()
        else:
            self.xp = np

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
        d.pop('xp')
        meta = json.dumps(d)
        with open(filepath+'.json', 'wb') as f:
            f.write(meta)

    def transform(self, data, test=False):
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

        return out.data.transpose(0, 2, 3, 1)

    def load_images(self, filepaths):

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
                        fig = self.plot_img(
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
                    fig = self.plot_img(
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

    def plot_img(self, data, img_path, epoch=1, batch=1, save=False):

        if data.data.shape[0] < 10:
            width = data.data.shape[0]
        else:
            width = 10
        x = Variable(data.data[:width])
        if self.flag_gpu:
            x.to_gpu()
        rec = self.model.forward(x)[0]
        rec.to_cpu()

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
        save_model(self.model, str(path), "%s_model" % str(name))
        save_model(self.opt, str(path), "%s_opt" % str(name))
        if save_meta:
            self._save_meta(os.path.join(path, "%s_meta" % str(name)))

    @classmethod
    def load(cls, model, opt, meta, flag_gpu=None):
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
            self.xp = cuda.cupy
            self.dec = self.dec.to_gpu()
            self.disc = self.disc.to_gpu()
        else:
            self.xp = np

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
        poplist = ['dec', 'disc', 'dec_opt', 'disc_opt', 'xp']
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

        decoded = self.dec(samp)
        disc_samp = self.disc(decoded, dropout_ratio=self.dropout_ratio)[0]

        disc_batch = self.disc(batch, dropout_ratio=self.dropout_ratio)[0]

        return disc_samp, disc_batch

    def inverse_transform(self, data, test=False):
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

        return out.data.transpose(0, 2, 3, 1)

    def load_images(self, filepaths):

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
                        fig = self.plot_img(
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
                    fig = self.plot_img(
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

    def plot_img(self, samples, img_path='./', epoch=1, batch=1, save_pic=False):

        if samples.shape[0] < 10:
            width = samples.shape[0]
        else:
            width = 10

        x = Variable(samples[:width])
        y = Variable(np.random.standard_normal((width, self.latent_width)).astype(np.float32))
        if self.flag_gpu:
            x.to_gpu()
            y.to_gpu()
        x_pics = self.dec(x)
        y_pics = self.dec(y)
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
        save_model(self.dec, str(path), "%s_dec" % str(name))
        save_model(self.disc, str(path), "%s_disc" % str(name))
        save_model(self.dec_opt, str(path), "%s_dec_opt" % str(name))
        save_model(self.disc_opt, str(path), "%s_disc_opt" % str(name))
        if save_meta:
            self._save_meta(os.path.join(path, "%s_meta" % str(name)))

    @classmethod
    def load(cls, dec, disc, dec_opt, disc_opt, meta, flag_gpu=None):
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
            loaded_class.model.to_gpu()

        return loaded_class


class VAEGAN(object):
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
            self.xp = cuda.cupy
            self.enc = self.dec.to_gpu()
            self.dec = self.dec.to_gpu()
            self.disc = self.disc.to_gpu()
        else:
            self.xp = np

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
                   'xp']
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

        return out.data.transpose(0, 2, 3, 1)

    def load_images(self, filepaths):

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
                        fig = self.plot_img(
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
                    fig = self.plot_img(
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

    def plot_img(self, img_data, samples, img_path='./', epoch=1, batch=1, save_pic=False):

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
        x_pics = self.dec(x)
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
        save_model(self.enc, str(path), "%s_enc" % str(name))
        save_model(self.dec, str(path), "%s_dec" % str(name))
        save_model(self.disc, str(path), "%s_disc" % str(name))
        save_model(self.enc_opt, str(path), "%s_enc_opt" % str(name))
        save_model(self.dec_opt, str(path), "%s_dec_opt" % str(name))
        save_model(self.disc_opt, str(path), "%s_disc_opt" % str(name))
        if save_meta:
            self._save_meta(os.path.join(path, "%s_meta" % str(name)))

    @classmethod
    def load(cls, enc, dec, disc, enc_opt, dec_opt, disc_opt, meta, flag_gpu=None):
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
            loaded_class.model.to_gpu()

        return loaded_class


def save_model(model, directory, name):
    if directory[-1] != '/':
        directory += '/'

    if not os.path.exists(os.path.dirname(directory)):
        os.makedirs(os.path.dirname(directory))

    save_path = os.path.join(directory, name)

    serializers.save_hdf5("%s.h5" % save_path, model)


def get_paths(directory):

    fnames = [os.path.join(directory, f)
              for f in os.listdir(directory)
              if os.path.isfile(os.path.join(directory, f))
              and not f.startswith('.')]
    return fnames


def image_resize(file_paths, new_dir, width, height):
    if new_dir[-1] != '/':
        new_dir += '/'

    if not os.path.exists(os.path.dirname(new_dir)):
        os.makedirs(os.path.dirname(new_dir))

    for f in tqdm.tqdm(file_paths):
        img = Image.open(f).resize((width, height), Image.ANTIALIAS).convert('RGB')
        new = os.path.join(new_dir, os.path.basename(f))
        img.save(new)
