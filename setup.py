from setuptools import setup

setup(
    name='photona',
    version='0.0.7',
    author='TJ Torres',
    author_email='ttorres@mit.edu',
    license='MIT',
    description='Python module for building variational auto-encoder models trained on images.',
    packages=['photona'],
    long_description='Python module for building variational auto-encoder models trained on images. \
                      It uses Chainer for the Neural Network framework and implements the methods \
                      layed out in: Diederik P Kingma, Max Welling; "Auto-Encoding Variational Bayes"; \
                      (2013).',
    url='https://github.com/stitchfix/photona',
    keywords=['autoencoder', 'images', 'deep learning', 'neaural networks', 'Chainer'],
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
        'tqdm'
    ]
)
