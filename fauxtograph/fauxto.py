import click
from fauxtograph import VAE, image_resize, get_paths
from BeautifulSoup import BeautifulSoup
import requests as r
import os
import numpy as np
import traceback
from PIL import Image


urls = [
    'https://www.spacetelescope.org/images/archive/category/galaxies/page/{0}/',
    'https://www.spacetelescope.org/images/archive/category/starclusters/page/{0}/',
    'https://www.spacetelescope.org/images/archive/category/nebulae/page/{0}/'
]

pages = [25, 5, 12]


def try_except_none(func):
    def wrapper(*args):
        try:
            return func(*args)
        except:
            print(func, args)
            print(traceback.format_exc())
            return None
    return wrapper


@try_except_none
def download_image(item):
    img_data = r.get(item.find('img')['src'])
    return img_data


@try_except_none
def download_page(filepath, url, pic_type, page):
    try:
        req = r.get(url)
    except:
        return None
    soup = BeautifulSoup(req.content)
    attrs = {'class': 'item'}
    for i, item in enumerate(soup.findAll('a', attrs=attrs)):
        print (i, item.find('img')['src'])
        img_data = download_image(item)
        if img_data is not None and img_data.status_code == 200:
            img = img_data.content
            numb_str = '{0}{1}_{2}.jpg'.format(pic_type, page, i)
            path = os.path.join(filepath, numb_str)
            with open(path, 'w') as f:
                f.write(img)


@click.command(help='Download Hubble Space Telescope Images')
@click.argument('filedir', type=click.Path(resolve_path=False, file_okay=False,
                                           dir_okay=True))
def download(filedir):
    filepath = filedir
    if not filepath[-1] == '/':
        filepath += '/'
    if not os.path.exists(os.path.dirname(filepath)):
            os.makedirs(os.path.dirname(filepath))

    for max_page, url in zip(pages, urls):
        for page in range(1, max_page):
            download_page(filepath, url.format(page), 'galaxies', page)
    filepaths = get_paths(filedir)

    image_resize(filepaths, filedir, 96, 96)


@click.command(help='Train a Variational Auto-encoder')
@click.argument('image_path', type=click.Path(resolve_path=False,
                                              file_okay=False, dir_okay=True))
@click.argument('model_path', type=click.Path(resolve_path=False,
                                              file_okay=True, dir_okay=False))
@click.option('--gpu', is_flag=True, help="Flag that determines whether or "
                                          "not to use the gpu")
@click.option('--shape', default=(96, 96), type = (int, int),
              help="Image shape tuple for training (image_width,"
              "image_height).")
@click.option('--latent_width', default=50, type=int,
              help="Width of the latent space.")
@click.option('--color_channels', default=3, type=int,
              help="Number of colors.")
@click.option('--batch', default=100, type=int,
              help="Number of images per training batch.")
@click.option('--epoch', default=200, type=int,
              help="Number of epochs to train.")
@click.option('--kl_ratio', default=1., type=float,
              help="Ratio of KL divergence term to reconstruction loss term.")
@click.option('--mode', default='linear', type=click.Choice(['linear', 'convolution']),
              help="Chose from fully-connected linear or convolutional architectures.")
def train(image_path, model_path, gpu, latent_width, color_channels, batch,
          epoch, shape, kl_ratio, mode):
    if not image_path[-1] == '/':
        image_path += '/'
    if not os.path.exists(os.path.dirname(image_path)):
            os.makedirs(os.path.dirname(image_path))
    if model_path[-1] == '/':
        click.echo("Model Path should be a file path not a directory path. "
                   "Removing trailing '/' ...")
        model_path = model_path[:-1]

    file_paths = [os.path.join(image_path, f)
                  for f in os.listdir(image_path)
                  if os.path.isfile(os.path.join(image_path, f))
                  and not f.startswith('.')]

    vae = VAE(img_width=shape[0],
              img_height=shape[1],
              encode_layers=[1000, 700, 300],
              decode_layers=[500, 700, 1000],
              kl_ratio=kl_ratio,
              flag_gpu=gpu,
              latent_width=latent_width,
              color_channels=color_channels,
              mode=mode)

    x_all = vae.load_images(file_paths)
    vae.fit(x_all, batch_size=batch, n_epochs=epoch)
    directory = os.path.dirname(model_path)
    name = os.path.basename(model_path)
    vae.save(directory, name)


@click.command(help='Generate images from a model.')
@click.argument('model_path', type=click.Path(resolve_path=False,
                file_okay=True, dir_okay=False))
@click.argument('optimizer_path', type=click.Path(resolve_path=False,
                file_okay=True, dir_okay=False))
@click.argument('meta_path', type=click.Path(resolve_path=False,
                file_okay=True, dir_okay=False))
@click.argument('img_dir', type=click.Path(resolve_path=False,
                file_okay=False, dir_okay=True))
@click.option('--number', default=10, type=int,
              help="Number of images to generate.")
@click.option('--extremity', default=10, type=float,
              help="Extremity of random encoded values.")
@click.option('--mean', default=0, type=float,
              help="Mean of random encoded values.")
@click.option('--format', default='jpg', type=click.Choice(['jpg', 'png']),
              help="Image format.")
@click.option('--image_multiplier', default=1.0, type=float, help="Multiplies pixes to  "
              "increase/deacrese brightness.")
def generate(model_path, optimizer_path, meta_path, img_dir, number,
             extremity, mean, format, image_multiplier):
    click.echo("Loading Model...")
    vae = VAE.load(model_path, optimizer_path, meta_path)
    click.echo("Model Loaded")
    click.echo("Saving Images to {0} as {1}".format(img_dir, format))
    if not img_dir[-1] == '/':
        img_dir += '/'
    if not os.path.exists(os.path.dirname(img_dir)):
            os.makedirs(os.path.dirname(img_dir))
    variance = extremity
    vec = np.random.normal(mean, variance,
                           (number, vae.latent_width)).astype('float32')

    reconstructed = vae.inverse_transform(vec, test=True) * 255

    for i in range(number):
        im = reconstructed[i]
        im = np.clip(image_multiplier*im, 0, 255)
        im = Image.fromarray(np.squeeze(im.astype(np.uint8)))
        fname = "{0}.{1}".format(i, format)
        path = os.path.join(img_dir, fname)
        im.save(path)


@click.group(help='Fauxtograph tools for training an image auto-encoder.')
def fauxtograph():
    pass


fauxtograph.add_command(download)
fauxtograph.add_command(train)
fauxtograph.add_command(generate)

if __name__ == '__main__':
    fauxtograph()
