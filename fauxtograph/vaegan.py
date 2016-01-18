import chainer.functions as F
import chainer.links as L
import chainer
import numpy as np


class Encoder(chainer.Chain):
    def __init__(
        self,
        img_width=75,
        img_height=100,
        color_channels=3,
        encode_layers=[1000, 600, 300],
        latent_width=100,
        mode='convolution',
    ):
        self.img_width = img_width
        self.img_height = img_height
        self.color_channels = color_channels
        self.encode_layers = encode_layers
        self.latent_width = latent_width
        self.mode = mode

        self.img_len = self.img_width*self.img_height*self.color_channels

        self._layers = {}

        if self.mode == 'convolution':
            self._layers['conv1'] = L.Convolution2D(self.color_channels, 32, 4, stride=2, pad=1,
                                                    wscale=0.02*np.sqrt(4*4*3))
            self._layers['conv2'] = L.Convolution2D(32, 64, 4, stride=2, pad=1,
                                                    wscale=0.02*np.sqrt(4*4*32))
            self._layers['conv3'] = L.Convolution2D(64, 128, 4, stride=2, pad=1,
                                                    wscale=0.02*np.sqrt(4*4*64))
            self._layers['conv4'] = L.Convolution2D(128, 256,  4, stride=2, pad=1,
                                                    wscale=0.02*np.sqrt(4*4*128))
            self._layers['conv5'] = L.Convolution2D(256, 512,  4, stride=2, pad=1,
                                                    wscale=0.02*np.sqrt(4*4*256))
            self._layers['bn1'] = L.BatchNormalization(32)
            self._layers['bn2'] = L.BatchNormalization(64)
            self._layers['bn3'] = L.BatchNormalization(128)
            self._layers['bn4'] = L.BatchNormalization(256)
            self._layers['bn5'] = L.BatchNormalization(512)
            self._layers['bn6'] = L.BatchNormalization(self.latent_width*2)
            self.img_len = reduce(lambda x, y: x*y, _calc_fc_size(self.img_height, self.img_width))
            self.img_width, self.img_height = _calc_fc_size(self.img_height, self.img_width)[1:]
            self.img_width, self.img_height = _calc_im_size(self.img_height, self.img_width)
            self._layers['lin'] = L.Linear(self.img_len, 2*self.latent_width)
        elif self.mode == 'linear':
            # Encoding Steps
            encode_layer_pairs = []
            if len(self.encode_layers) > 0:
                encode_layer_pairs = [(self.img_len, self.encode_layers[0])]
            if len(self.encode_layers) > 1:
                encode_layer_pairs += zip(
                    self.encode_layers[:-1],
                    self.encode_layers[1:]
                    )
            if self.encode_layers:
                encode_layer_pairs += [(self.encode_layers[-1], self.latent_width * 2)]
            else:
                encode_layer_pairs += [(self.img_len, self.latent_width * 2)]
            for i, (n_in, n_out) in enumerate(encode_layer_pairs):
                self._layers['linear_%i' % i] = L.Linear(n_in, n_out)
        else:
            raise NameError(
                "Improper mode type %s. Encoder mode must be 'linear' or 'convolution'."
                % self.mode)

        super(Encoder, self).__init__(**self._layers)

    def __call__(self, x, test=False):

        batch = x

        if self.mode == 'convolution':
            n_pics = batch.data.shape[0]

            batch = self.conv1(batch)
            batch = self.bn1(batch, test=test)
            batch = F.relu(batch)

            batch = self.conv2(batch)
            batch = self.bn2(batch, test=test)
            batch = F.relu(batch)

            batch = self.conv3(batch)
            batch = self.bn3(batch, test=test)
            batch = F.relu(batch)

            batch = self.conv4(batch)
            batch = self.bn4(batch, test=test)
            batch = F.relu(batch)

            batch = self.conv5(batch)
            batch = self.bn5(batch, test=test)
            batch = F.relu(batch)

            batch = F.reshape(batch, (n_pics, self.img_len))
            batch = F.relu(self.bn6(self.lin(batch), test=test))

        elif self.mode == 'linear':
            n_layers = len(self.encode_layers)
            for i in range(n_layers):
                batch = F.relu(getattr(self, 'linear_%i' % i)(batch))
            batch = F.relu(getattr(self, 'linear_%i' % n_layers)(batch))

        return batch


class Decoder(chainer.Chain):
    def __init__(
        self,
        img_width=75,
        img_height=100,
        color_channels=3,
        decode_layers=[300, 600, 1000],
        latent_width=100,
        mode='convolution'
    ):
        self.img_width = img_width
        self.img_height = img_height
        self.color_channels = color_channels
        self.decode_layers = decode_layers
        self.latent_width = latent_width
        self.mode = mode

        self.img_len = self.img_width*self.img_height*self.color_channels

        if self.mode == 'convolution':
            self.img_len = reduce(lambda x, y: x*y, _calc_fc_size(self.img_height, self.img_width))
            self.img_width, self.img_height = _calc_fc_size(self.img_height, self.img_width)[1:]
            self.img_width, self.img_height = _calc_im_size(self.img_height, self.img_width)

        self._layers = {}

        if self.mode == 'convolution':

            self._layers['lin'] = L.Linear(self.latent_width, self.img_len, wscale=0.02*np.sqrt(self.latent_width))
            self._layers['deconv5'] = L.Deconvolution2D(512, 256, 4, stride=2, pad=1, wscale=0.02*np.sqrt(4*4*512))
            self._layers['deconv4'] = L.Deconvolution2D(256, 128, 4, stride=2, pad=1, wscale=0.02*np.sqrt(4*4*256))
            self._layers['deconv3'] = L.Deconvolution2D(128, 64, 4, stride=2, pad=1, wscale=0.02*np.sqrt(4*4*128))
            self._layers['deconv2'] = L.Deconvolution2D(64, 32, 4, stride=2, pad=1, wscale=0.02*np.sqrt(4*4*64))
            self._layers['deconv1'] = L.Deconvolution2D(32, self.color_channels, 4, stride=2, pad=1, wscale=0.02*np.sqrt(4*4*32))
            self._layers['bn2'] = L.BatchNormalization(32)
            self._layers['bn3'] = L.BatchNormalization(64)
            self._layers['bn4'] = L.BatchNormalization(128)
            self._layers['bn5'] = L.BatchNormalization(256)
            self._layers['bn6'] = L.BatchNormalization(self.img_len)
        elif self.mode == 'linear':
            # Decoding Steps
            decode_layer_pairs = []
            if len(self.decode_layers) > 0:
                decode_layer_pairs = [(self.latent_width, self.decode_layers[0])]
            if len(self.decode_layers) > 1:
                decode_layer_pairs += zip(
                    self.decode_layers[:-1],
                    self.decode_layers[1:]
                    )
            if self.decode_layers:
                decode_layer_pairs += [(self.decode_layers[-1], self.img_len)]
            else:
                decode_layer_pairs += [(self.latent_width, self.img_len)]

            for i, (n_in, n_out) in enumerate(decode_layer_pairs):
                self._layers['linear_%i' % i] = L.Linear(n_in, n_out)

        else:
            raise NameError(
                "Improper mode type %s. Encoder mode must be 'linear' or 'convolution'."
                % self.mode)

        super(Decoder, self).__init__(**self._layers)

    def __call__(self, z, test=False, rectifier='clipped_relu'):
        batch = z

        if self.mode == 'convolution':
            batch = F.relu(self.bn6(self.lin(z), test=test))
            n_pics = batch.data.shape[0]
            start_array_shape = (n_pics,) + _calc_fc_size(self.img_height, self.img_width)
            batch = F.reshape(batch, start_array_shape)
            batch = F.relu(self.bn5(self.deconv5(batch), test=test))
            batch = F.relu(self.bn4(self.deconv4(batch), test=test))
            batch = F.relu(self.bn3(self.deconv3(batch), test=test))
            batch = F.relu(self.bn2(self.deconv2(batch), test=test))
            batch = self.deconv1(batch)

        elif self.mode == 'linear':
            n_layers = len(self.decode_layers)
            for i in range(n_layers):
                batch = F.relu(getattr(self, 'linear_%i' % i)(batch))
            batch = F.relu(getattr(self, 'linear_%i' % n_layers)(batch))
            batch = F.reshape(batch, (-1, self.color_channels, self.img_height, self.img_width))
        if rectifier == 'clipped_relu':
            batch = F.clipped_relu(batch, z=1.0)
        elif rectifier == 'sigmoid':
            batch = F.sigmoid(batch, z=1.0)
        else:
            raise NameError(
                "Unsupported rectifier type: %s, must be either 'sigmoid' or 'clipped_relu'."
                % rectifier)

        return batch


class Discriminator(chainer.Chain):
    def __init__(
        self,
        img_width=75,
        img_height=100,
        color_channels=3,
        disc_layers=[1000, 600, 300],
        latent_width=100,
        mode='convolution',
    ):
        self.img_width = img_width
        self.img_height = img_height
        self.color_channels = color_channels
        self.disc_layers = disc_layers
        self.mode = mode

        self.img_len = self.img_width*self.img_height*self.color_channels

        self._layers = {}

        if self.mode == 'convolution':
            self._layers['conv1'] = L.Convolution2D(self.color_channels, 32, 4, stride=2, pad=1,
                                                    wscale=0.02*np.sqrt(4*4*3))
            self._layers['conv2'] = L.Convolution2D(32, 64, 4, stride=2, pad=1,
                                                    wscale=0.02*np.sqrt(4*4*32))
            self._layers['conv3'] = L.Convolution2D(64, 128, 4, stride=2, pad=1,
                                                    wscale=0.02*np.sqrt(4*4*64))
            self._layers['conv4'] = L.Convolution2D(128, 256,  4, stride=2, pad=1,
                                                    wscale=0.02*np.sqrt(4*4*128))
            self._layers['conv5'] = L.Convolution2D(256, 512,  4, stride=2, pad=1,
                                                    wscale=0.02*np.sqrt(4*4*256))
            self._layers['bn2'] = L.BatchNormalization(64)
            self._layers['bn3'] = L.BatchNormalization(128)
            self._layers['bn4'] = L.BatchNormalization(256)
            self._layers['bn5'] = L.BatchNormalization(512)
            self.img_len = reduce(lambda x, y: x*y, _calc_fc_size(self.img_height, self.img_width))
            self.img_width, self.img_height = _calc_fc_size(self.img_height, self.img_width)[1:]
            self.img_width, self.img_height = _calc_im_size(self.img_height, self.img_width)
            self._layers['lin'] = L.Linear(self.img_len, 2)
        elif self.mode == 'linear':
            # Encoding Steps
            disc_layer_pairs = []
            if len(self.disc_layers) > 0:
                disc_layer_pairs = [(self.img_len, self.disc_layers[0])]
            if len(self.disc_layers) > 1:
                disc_layer_pairs += zip(
                    self.disc_layers[:-1],
                    self.disc_layers[1:]
                    )
            if self.disc_layers:
                disc_layer_pairs += [(self.disc_layers[-1], 2)]
            else:
                disc_layer_pairs += [(self.img_len, 2)]

            for i, (n_in, n_out) in enumerate(disc_layer_pairs):
                self._layers['linear_%i' % i] = L.Linear(n_in, n_out)
        else:
            raise NameError(
                "Improper mode type %s. Encoder mode must be 'linear' or 'convolution'."
                % self.mode)

        super(Discriminator, self).__init__(**self._layers)

    def __call__(self, x, test=False, dropout_ratio=0.5):

        batch = x

        if self.mode == 'convolution':
            n_pics = batch.data.shape[0]

            batch = self.conv1(batch)
            batch = F.relu(batch)

            batch = self.conv2(batch)
            batch = self.bn2(batch, test=test)
            batch = F.relu(batch)

            batch = self.conv3(batch)
            batch_out = self.bn3(batch, test=test)
            batch = F.relu(batch_out)
            batch = F.dropout(batch, ratio=dropout_ratio)

            batch = self.conv4(batch)
            batch = self.bn4(batch, test=test)
            batch = F.relu(batch)
            batch = F.dropout(batch, ratio=dropout_ratio)

            batch = self.conv5(batch)
            batch = self.bn5(batch, test=test)
            batch = F.relu(batch)
            batch = F.dropout(batch, ratio=dropout_ratio)

            batch = F.reshape(batch, (n_pics, self.img_len))
            batch = self.lin(batch)

        elif self.mode == 'linear':
            n_layers = len(self.disc_layers)
            for i in range(n_layers):
                batch = F.relu(getattr(self, 'linear_%i' % i)(batch))
                batch_out = batch
            batch = F.relu(getattr(self, 'linear_%i' % n_layers)(batch))

        return batch, batch_out


def _calc_fc_size(img_height, img_width):
    height, width = img_height, img_width
    for _ in range(5):
        height, width = _get_conv_outsize(
            (height, width),
            4, 2, 1)

    conv_out_layers = 512
    return conv_out_layers, height, width


def _calc_im_size(img_height, img_width):
    height, width = img_height, img_width
    for _ in range(5):
        height, width = _get_deconv_outsize((height, width),
                                            4, 2, 1)

    return height, width


def _get_conv_outsize(shape, k, stride, padding, pool=False):
    mod_h = (shape[0] + 2*padding - k) % stride
    mod_w = (shape[1] + 2*padding - k) % stride
    height = (shape[0] + 2*padding - k) / stride + 1
    width = (shape[1] + 2*padding - k) / stride + 1

    if pool and not mod_h == 0:
        height += 1
    if pool and not mod_w == 0:
        width += 1

    return (height, width)


def _get_deconv_outsize(shape, kh, sy, ph):
    size_h = sy * (shape[0] - 1) + kh - 2 * ph
    size_w = sy * (shape[1] - 1) + kh - 2 * ph
    return size_h, size_w
