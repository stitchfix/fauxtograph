from setuptools import setup

setup(
    name='fauxtograph',
    version='1.0.0',
    author='TJ Torres',
    author_email='ttorres@mit.edu',
    license='MIT',
    description='Python module for training unsupervised deep, generative models on images.',
    packages=['fauxtograph'],
    long_description='Python module for training unsupervised deep, generative models on images. ' +
                     'It uses Chainer for the Neural Network framework and implements several methods, ' +
                     'including Variational Auto-encoders, Generative Adversarial Networks, and their ' +
                     'combination. These methods are built with reference to personal notes and the following ' +
                     'papers:\n'
                     '1) Diederik P Kingma, Max Welling; "Auto-Encoding Variational Bayes"; (2013).\n' +
                     '2) Alec Radford et. al.; "Unsupervised Representation Learning with Deep ' +
                     'Convolutional Generative Adversarial Networks"; (2015).\n' +
                     '3) Anders Boesen et. al.; "Autoencoding Beyond Pixels Using a Learned ' +
                     'Similarity Metric"; (2015).',
    url='https://github.com/stitchfix/fauxtograph',
    keywords=['unsupervised', 'images', 'deep learning', 'neural networks', 'Chainer'],
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 2.6',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3.4',
    ],
    install_requires=[
        'chainer==1.6.0',
        'pillow',
        'joblib',
        'tqdm',
        'BeautifulSoup',
        'requests',
        'numpy',
        'click>=5.0',
        'matplotlib',
    ],
    entry_points={
        'console_scripts': [
            'fauxtograph = fauxtograph.fauxto:fauxtograph'
        ],
    },
)
