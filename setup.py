from setuptools import setup

setup(
    name='fauxtograph',
    version='0.1.7',
    author='TJ Torres',
    author_email='ttorres@mit.edu',
    license='MIT',
    description='Python module for building variational auto-encoder models trained on images.',
    packages=['fauxtograph'],
    long_description='Python module for building variational auto-encoder models trained on images. ' +
                     'It uses Chainer for the Neural Network framework and implements the methods ' +
                     'layed out in: Diederik P Kingma, Max Welling; "Auto-Encoding Variational Bayes"; ' +
                     '(2013).',
    url='https://github.com/stitchfix/fauxtograph',
    keywords=['autoencoder', 'images', 'deep learning', 'neural networks', 'Chainer'],
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 2.6',
        'Programming Language :: Python :: 2.7',
    ],
    install_requires=[
        'chainer==1.2.0',
        'pillow',
        'joblib',
        'tqdm',
        'BeautifulSoup',
        'requests',
        'numpy',
        'click>=5.0'
    ],
    entry_points={
        'console_scripts': [
            'fauxtograph = fauxtograph.fauxto:fauxtograph'
        ],
    },
)
