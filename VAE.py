import chainer
from chainer import cuda
import chainer.functions as F
from chainer import optimizers, variable
import numpy as np
import PIL.Image
import time
import joblib
import os

if not os.path.exists('cache'):
    os.mkdir('cache')
mem = joblib.Memory('cache')

# TODO: unit test
#   TODO: unit test for pickle / depickle, c/gpu
#   TODO: test that before training the RMSE is as random as you'd expect
# TODO: setup.py & PyPI Instllation
# TODO: Downloader for images
# TODO: docstrings for public functions
# TODO: private funcs tend to have underscores
# TODO: watch the 80 character limit


class ImageAutoEncoder():
    def __init__(self, picture_width=75, picture_height=100, color_channels=3,
                 encode_layers=[1000, 600, 300],
                 decode_layers=[300, 800, 1000], latent_width=100,
                 flag_gpu=None):
        '''

        '''
        # TODO Docstring see the following:
        # http://sphinxcontrib-napoleon.readthedocs.org/en/latest/example_numpy.html

        self.encode_sizes = encode_layers
        self.decode_sizes = decode_layers
        self.latent_dim = latent_width
        self.img_width = picture_width
        self.img_height = picture_height
        self.colors = color_channels
        self.img_len = self.img_width*self.img_height*self.colors
        self.flag_gpu = flag_gpu

        self.model = self.layer_setup()
        self.optimizer = optimizers.Adam()
        self.optimizer.setup(self.model)
        self.x_all = []

    def layer_setup(self):
        # TODO: underscore this method if it isn't public
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

    def encode(self, img_batch):
        # TODO: underscore this method if it isn't public
        batch = img_batch
        for i in xrange(len(self.encode_sizes)+1):
            batch = F.relu(getattr(self.model, 'encode_%i' % i)(batch))
        return batch

    def decode(self, latent_vec):
        # TODO: underscore this method if it isn't public
        batch = latent_vec
        n_layers = len(self.decode_sizes)
        for i in xrange(n_layers):
            batch = F.relu(getattr(self.model, 'decode_%i' % i)(batch))
        batch = F.sigmoid(getattr(self.model, 'decode_%i' % n_layers)(batch))
        return batch

    def forward(self, img_batch):
        # TODO: underscore this method if it isn't public
        batch = chainer.Variable(img_batch/255.)
        encoded = self.encode(batch)
        mean, std = F.split_axis(encoded, 2, 1)
        samples = np.random.standard_normal(mean.data.shape).astype('float32')
        if self.flag_gpu:
            samples = cuda.to_gpu(samples)
        samples = chainer.Variable(samples)
        sample_set = samples * F.exp(0.5*std) + mean
        output = self.decode(sample_set)
        reconstruction_loss = F.mean_squared_error(output, batch)
        kl_div = -0.5 * F.sum(1 + std - mean ** 2 - F.exp(std))
        kl_div /= (img_batch.shape[1] * img_batch.shape[0])
        return reconstruction_loss, kl_div, output

    def gaussian_kl_divergence(self, mean, ln_var):
        # TODO Just use a new version of chainer and use their KL Div
        # no need to copy it verbatim here -- just remove this fun
        """Calculate KL-divergence between given gaussian and the standard one.

        Given two variable ``mean`` representing :math:`\\mu` and ``ln_var``
        representing :math:`\\log(\\sigma^2)`, this function returns a variable
        representing KL-divergence between given multi-dimensional gaussian
        :math:`N(\\mu, S)` and the standard Gaussian :math:`N(0, I)`

        .. math::

            D_{\\mathbf{KL}}(N(\\mu, S) \\| N(0, I)),

        where :math:`S` is a diagonal matrix such that :math:`S_{ii} =
        \\sigma_i^2` and :math:`I` is an identity matrix.

        Args:
            mean (~chainer.Variable): A variable representing mean of given
                gaussian distribution, :math:`\\mu`.
            ln_var (~chainer.Variable): A variable representing logarithm of
                variance of given gaussian distribution, :math:`\\log(\\sigma^2)`.

        Returns:
            ~chainer.Variable: A variable representing KL-divergence between
                given gaussian distribution and the standard gaussian.

        """
        assert isinstance(mean, variable.Variable)
        assert isinstance(ln_var, variable.Variable)
        J = mean.data.size
        var = F.exp(ln_var)
        return (F.sum(mean * mean + var - ln_var) - J) * 0.5

    @mem.cache()
    def load_images(self, files, shape=(75, 100)):
        # TODO: docstring for all public functions
        # TODO: use logger instead of print
        # TODO: The shape was defined at initialization, 
        # shouldn't need to specify it here?
        # https://docs.python.org/2/library/logging.html
        print("Loading Image Files...")
        resize = lambda x: PIL.Image.open(x).resize(shape, PIL.Image.ANTIALIAS)
        self.x_all = np.array([resize(fname) for fname in files])
        self.x_all = self.x_all.astype(np.float32)
        print("Image Files Loaded!")

    def train(self, n_epochs=200, batch_size=100):
        # TODO: docstring for all public functions
        # TODO: rename to use sklearn's fit / predict / transform convention
        n_samp = self.x_all.shape[0]
        x_train = self.x_all.flatten().reshape((n_samp, -1))
        # Train Model#
        print("\n Training for %i epochs. \n" % n_epochs)
        for epoch in xrange(1, n_epochs + 1):
            print('epoch: %i' % epoch)
            t1 = time.time()
            indexes = np.random.permutation(x_train.shape[0])
            for i in xrange(0, x_train.shape[0], batch_size):
                if self.flag_gpu:
                    x_batch = cuda.to_gpu(x_train[indexes[i: i + batch_size]])
                else:
                    x_batch = x_train[indexes[i: i + batch_size]]
                self.optimizer.zero_grads()
                r_loss, kl_div, out = self.forward(x_batch)
                loss = r_loss + kl_div
                loss.backward()
                self.optimizer.update()
            if self.flag_gpu:
                train_set = cuda.to_gpu(x_train)
            else:
                train_set = x_train
            r_loss, kl_div, _ = self.forward(train_set)
            msg = "r_loss = {0} , kl_div = {1}"
            print(msg.format(r_loss.data, kl_div.data))
            t_diff = time.time()-t1
            print("time: %f\n\n" % t_diff)

    def dump(self, filepath):
        # TODO: docstring for all public functions
        if not os.path.exists(os.path.dirname(filepath)):
            os.makedirs(os.path.dirname(filepath))
        print("Dumping model to file: %s " % filepath)
        if self.flag_gpu:
            self.model.to_cpu()
        joblib.dump(self, filepath)

    # TODO: Just pickle the whole VAE class itself
    # TODO: Use the @staticmethod decorator to return an instantiated class
    def load(self, filepath, img_width=75, img_height=100, color_channels=3):
        # TODO: docstring for all public functions
        if not os.path.exists(filepath):
            print("Model file does not exist. Please check the file path.")
        else:
            self.img_width = img_width
            self.img_height = img_height
            self.colors = color_channels
            self.img_len = self.img_width*self.img_height*self.colors
            layer_arr = []
            for i in range(1, len(self.model.parameters), 2):
                layer_arr.append(self.model.parameters[i].shape[0])
            layer_arr = np.array(layer_arr)
            # TODO: It's unclear what these lines do -- maybe add to the for loop above?
            self.encode_sizes = layer_arr[np.where(layer_arr == self.img_len)[0][0]+1:-1].tolist()
            self.decode_sizes = layer_arr[:np.where(layer_arr == self.img_len)[0][0]].tolist()
            self.latent_dim = layer_arr[-1] / 2
            # These initiali
            self.model = self.layer_setup()
            self.optimizer = optimizers.Adam()
            self.optimizer.setup(self.model)
            self.model = joblib.load(filepath)
