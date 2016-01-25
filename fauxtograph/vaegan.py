import chainer.functions as F
import chainer.links as L
from chainer import Variable
import chainer
import numpy as np


class Encoder(chainer.Chain):
    '''Chainer encoder chain that has optional linear or convolutional
    structure.

    In convolutional mode, the encoder performs the folowing:

        Convolution: 32, 4x4, stride 2, pad 1
        Batch Normalization: 32
        Relu
        Convolution: 64, 4x4, stride 2, pad 1
        Batch Normalization: 64
        Relu
        Convolution: 128, 4x4, stride 2, pad 1
        Batch Normalization: 128
        Relu
        Convolution: 256, 4x4, stride 2, pad 1
        Batch Normalization: 256
        Relu
        Convolution: 512, 4x4, stride 2, pad 1
        Batch Normalization: 512
        Relu
        Linear (convolution_width, 2*latent_width)
        Batch Normalization: 2*latent_width
        Relu


    In linear mode the encoder passes forward through fully-connected linear
    transformations layers with sizes given by the encode_layers attribute.

    Attributes
    ----------
    encode_layers : List[int]
        List of layer sizes for hidden linear encoding layers of the model.
        Only taken into account when mode='linear'.
    latent_width : int
        Dimension of latent encoding space.
    img_width : int
        Width of the desired image representation.
    img_height : int
        Height of the desired image representation.
    color_channels : int
        Number of color channels in the input images.
    mode: str
        Mode to set the encoder architectures. Can be either
        'convolution' or 'linear'.
    '''
    def __init__(
        self,
        img_width=64,
        img_height=64,
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
            self.img_len = reduce(lambda x, y: x*y, calc_fc_size(self.img_height, self.img_width))
            self.img_width, self.img_height = calc_fc_size(self.img_height, self.img_width)[1:]
            self.img_width, self.img_height = calc_im_size(self.img_height, self.img_width)
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
    '''Chainer decoder chain that has optional linear or convolutional
    structure.

    In convolutional mode, the encoder performs the folowing:

        Linear (latent_width, convolution_width)
        Batch Normalization: convolution_width
        Deconvolution: 256, 4x4, stride 2, pad 1
        Batch Normalization: 256
        Relu
        Deconvolution: 128, 4x4, stride 2, pad 1
        Batch Normalization: 128
        Relu
        Deconvolution: 64, 4x4, stride 2, pad 1
        Batch Normalization: 64
        Relu
        Deconvolution: 32, 4x4, stride 2, pad 1
        Batch Normalization: 32
        Relu
        Deconvolution: 3, 4x4, stride 2, pad 1
        Batch Normalization: 3
        Selectable: Clipped Relu or Sigmoid

    In linear mode the decoder passes forward through fully-connected linear
    transformations layers with sizes given by the decode_layers attribute.

    Attributes
    ----------
    decode_layers : List[int]
        List of layer sizes for hidden linear encoding layers of the model.
        Only taken into account when mode='linear'.
    latent_width : int
        Dimension of latent encoding space.
    img_width : int
        Width of the desired image representation.
    img_height : int
        Height of the desired image representation.
    color_channels : int
        Number of color channels in the input images.
    mode: str
        Mode to set the encoder architectures. Can be either
        'convolution' or 'linear'.
    '''
    def __init__(
        self,
        img_width=64,
        img_height=64,
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

        self._layers = {}

        if self.mode == 'convolution':
            self.img_len = reduce(lambda x, y: x*y, calc_fc_size(self.img_height, self.img_width))
            self.img_width, self.img_height = calc_fc_size(self.img_height, self.img_width)[1:]
            self.img_width, self.img_height = calc_im_size(self.img_height, self.img_width)

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
            start_array_shape = (n_pics,) + calc_fc_size(self.img_height, self.img_width)
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
            batch = F.reshape(batch, (-1, self.img_height, self.img_width, self.color_channels))
        if rectifier == 'clipped_relu':
            batch = F.clipped_relu(batch, z=1.0)
        elif rectifier == 'sigmoid':
            batch = F.sigmoid(batch)
        else:
            raise NameError(
                "Unsupported rectifier type: %s, must be either 'sigmoid' or 'clipped_relu'."
                % rectifier)

        return batch


class Discriminator(chainer.Chain):
    '''Chainer discriminator chain that has optional linear or convolutional
    structure. It outputs an activation at the 3rd convolution layer as well
    as the discriminator 2d output (prior to softmax).

    In convolutional mode, the discriminator performs the folowing:

        Convolution: 32, 4x4, stride 2, pad 1
        Relu
        Convolution: 64, 4x4, stride 2, pad 1
        Batch Normalization: 64
        Relu
        Convolution: 128, 4x4, stride 2, pad 1 : Activation Output
        Batch Normalization: 128
        Relu
        Dropout
        Convolution: 256, 4x4, stride 2, pad 1
        Batch Normalization: 256
        Relu
        Dropout
        Convolution: 512, 4x4, stride 2, pad 1
        Batch Normalization: 512
        Relu
        Dropout
        Linear (convolution_width, 2)
        Relu

    In linear mode the discriminator passes forward through fully-connected linear
    transformations layers with sizes given by the disc_layers attribute.

    Attributes
    ----------
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
    mode: str
        Mode to set the encoder architectures. Can be either
        'convolution' or 'linear'.
    '''
    def __init__(
        self,
        img_width=64,
        img_height=64,
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
            self.img_len = reduce(lambda x, y: x*y, calc_fc_size(self.img_height, self.img_width))
            self.img_width, self.img_height = calc_fc_size(self.img_height, self.img_width)[1:]
            self.img_width, self.img_height = calc_im_size(self.img_height, self.img_width)
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


class EncDec(chainer.Chain):
    '''A combination of the fauxtograph.Encoder and fauxtograph.Decoder
    chains. These two chains need to be combined to avoid two optimizers
    with the Variational Auto-encoder.

    In linear mode the encoder/decoder pass forward through fully-connected linear
    transformations layers with sizes given by the encode/decode_layers attribute.

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
    mode : str
        Mode to set the encoder architectures. Can be either
        'convolution' or 'linear'.
    flag_gpu : bool
        Flag to mark whether to use the gpu.
    rectifier : str
        Sets how the decoder output is rectified. Can be either
        'clipped_relu' or 'sigmoid'.
    '''
    def __init__(
        self,
        img_width=64,
        img_height=64,
        color_channels=3,
        encode_layers=[1000, 600, 300],
        decode_layers=[300, 600, 1000],
        latent_width=100,
        mode='convolution',
        flag_gpu=True,
        rectifier='clipped_relu'
    ):
        self.flag_gpu = flag_gpu
        self.rectifier = rectifier
        super(EncDec, self).__init__(
            enc=Encoder(img_width=img_width,
                        img_height=img_height,
                        color_channels=color_channels,
                        encode_layers=encode_layers,
                        latent_width=latent_width,
                        mode=mode),
            dec=Decoder(img_width=img_width,
                        img_height=img_height,
                        color_channels=color_channels,
                        decode_layers=decode_layers,
                        latent_width=latent_width,
                        mode=mode)
        )

    def encode(self, data, test=False):
        x = self.enc(data, test=test)
        mean, ln_var = F.split_axis(x, 2, 1)
        samp = np.random.standard_normal(mean.data.shape).astype('float32')
        samp = Variable(samp)
        if self.flag_gpu:
            samp.to_gpu()
        z = samp * F.exp(0.5*ln_var) + mean

        return z, mean, ln_var

    def decode(self, z, test=False):
        x = self.dec(z, test=test, rectifier=self.rectifier)

        return x

    def forward(self, batch, test=False):

        out, means, ln_vars = self.encode(batch, test=test)
        out = self.decode(out, test=test)
        normer = reduce(lambda x, y: x*y, means.data.shape)

        kl_loss = F.gaussian_kl_divergence(means, ln_vars)/normer
        rec_loss = F.mean_squared_error(batch, out)

        return out, kl_loss, rec_loss

    def __call__(self, x, test=False):

        return self.forward(x, test=test)


def calc_fc_size(img_height, img_width):
    '''Calculates shape of data after encoding.

    Parameters
    ----------
    img_height : int
        Height of input image.
    img_width : int
        Width of input image.

    Returns
    -------
    encoded_shape : tuple(int)
        Gives back 3-tuple with new dims.
    '''
    height, width = img_height, img_width
    for _ in range(5):
        height, width = _get_conv_outsize(
            (height, width),
            4, 2, 1)

    conv_out_layers = 512
    return conv_out_layers, height, width


def calc_im_size(img_height, img_width):
    '''Calculates shape of data after decoding.

    Parameters
    ----------
    img_height : int
        Height of encoded data.
    img_width : int
        Width of encoded data.

    Returns
    -------
    encoded_shape : tuple(int)
        Gives back 2-tuple with decoded image dimensions.
    '''
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
