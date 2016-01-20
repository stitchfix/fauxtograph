from PIL import Image
from chainer import cuda
#import chainer.functions as F
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


class VariationalAutoEncoder(object):
    def __init__(self, img_width=64, img_height=64, color_channels=3,
                 encode_layers=[1000, 600, 300],
                 decode_layers=[300, 800, 1000],
                 latent_width=100, kl_ratio=1.0, flag_gpu=True,
                 mode='linear', adam_alpha=0.0001, adam_beta1=0.9,
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

        # self.enc = Encoder(img_width=self.img_width,
        #                    img_height=self.img_height,
        #                    encode_layers=self.encode_layers,
        #                    latent_width=self.latent_width,
        #                    mode=self.mode)
        # self.dec = Decoder(img_width=self.img_width,
        #                    img_height=self.img_height,
        #                    decode_layers=self.decode_layers,
        #                    latent_width=self.latent_width,
        #                    mode=self.mode)
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
            self.enc = self.enc.to_gpu()
            self.dec = self.dec.to_gpu()
        else:
            self.xp = np

        self.opt = O.Adam(alpha=self.adam_alpha, beta1=self.adam_beta1)

    def _check_dims(self):
        h, w = calc_fc_size(self.img_height, self.img_width)[1:]
        h, w = calc_im_size(h, w)

        assert (h == self.img_height) and (w == self.img_width),\
            "To use convolution, please resize images " + \
            "to nearest target height, width = %d, %d" % (h, w)

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
        model_path='./training_model/',
        img_path='./training_images/',
    ):
        width = 10
        self.opt.setup(self.model)

        if weight_decay:
            self.opt.add_hook(chainer.optimizer.WeightDecay(0.00001))

        n_data = img_data.shape[0]

        batch_iter = list(range(0, n_data, batch_size))
        n_batches = len(batch_iter)

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
                    assert type(pic_freq) == int, "Save_freq must be an integer."
                    if count % pic_freq == 0:
                        fig = self.plot_img(
                            plot_pics,
                            img_path=img_path,
                            epoch=epoch
                        )
                        display(fig)

            if save_freq > 0:
                assert type(save_freq) == int, "Save_freq must be an integer."
                if epoch % save_freq == 0:
                    save_model(self.model, model_path, epoch, base='vae_model')
                    save_model(self.opt, model_path, epoch, base='vae_opt')
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

        if img_path[-1] != '/':
            img_path += '/'
        if not os.path.exists(os.path.dirname(img_path)):
            os.makedirs(os.path.dirname(img_path))

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
        if self.mode == 'convolution':
            orig = x.data.transpose(0, 2, 3, 1)
            rec = rec.data.transpose(0, 2, 3, 1)
        else:
            orig = x.data
            rec = rec.data
        for i in range(width):
            plt.subplot(2, width, i+1)
            plt.imshow(orig[i])
            plt.axis("off")
        for i in range(width):
            plt.subplot(2, width, width+i+1)
            plt.imshow(rec[i])
            plt.axis("off")
        if save:
            save_str = 'image_vae_epoch%d_batch%d.png' % (epoch, batch)
            plt.savefig(os.path.join(img_path, save_str))
        plt.close()
        return fig

    def save(self, path, name):
        save_model(self.model, path, epoch='_%s' % name, base='vae_model')
        save_model(self.opt, path, epoch='_%s' % name, base='vae_opt')



    



def save_model(model, model_path, epoch, base='model'):
    if model_path[-1] != '/':
        model_path += '/'

    if not os.path.exists(os.path.dirname(model_path)):
        os.makedirs(os.path.dirname(model_path))

    serializers.save_hdf5("%s/%s_epoch%s.h5" % (model_path, base, str(epoch)), model)


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
