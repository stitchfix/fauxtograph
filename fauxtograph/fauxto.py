import click
from fauxtograph import ImageAutoEncoder
from BeautifulSoup import BeautifulSoup
import requests as r
import os
import numpy as np
from PIL import Image


@click.group(help='Fauxtograph tools for training an image auto-encoder.')
def fauxtograph():
    pass


@click.command(help='Download Hubble Space Telescope Images')
@click.argument('filedir', type=click.Path(resolve_path=False, file_okay=False,
                                           dir_okay=True))
def download(filedir):
    filepath = filedir
    if not filepath[-1] == '/':
        filepath += '/'
    if not os.path.exists(os.path.dirname(filepath)):
            os.makedirs(os.path.dirname(filepath))

    def getter(filepath, req, pic_type, page):

            soup = BeautifulSoup(req.content)
            q = 0
            attrs = {'class': 'item'}
            for i, item in enumerate(soup.findAll('a', attrs=attrs)):
                q += 1
                print (q, item.find('img')['src'])
                try:
                    img_data = r.get(item.find('img')['src'])
                    if img_data.status_code == 200:
                        img = img_data.content
                        numb_str = '{0}{1}_{2}.jpg'.format(pic_type, page, q)
                        path = os.path.join(filepath, numb_str)
                        with open(path, 'w') as f:
                            f.write(img)
                except:
                    pass
    for j in range(25):
        req = r.get('https://www.spacetelescope.org/images/' +
                    'archive/category/galaxies/page/{0}/'
                    .format(j+1))
        getter(filepath, req, 'galaxies', j+1)

    for j in range(5):
        req = r.get('https://www.spacetelescope.org/images/' +
                    'archive/category/starclusters/page/{0}/'
                    .format(j+1))
        getter(filepath, req, 'starclusters', j+1)
    for j in range(12):
        req = r.get('https://www.spacetelescope.org/images/' +
                    'archive/category/nebulae/page/{0}/'
                    .format(j+1))
        getter(filepath, req, 'nebulae', j+1)


@click.command(help='Train a Variational Auto-encoder')
@click.argument('image_path', type=click.Path(resolve_path=False,
                                              file_okay=False, dir_okay=True))
@click.argument('model_path', type=click.Path(resolve_path=False,
                                              file_okay=True, dir_okay=False))
@click.option('--gpu', is_flag=True, help="Flag that determines whether or"
                                          "not to use the gpu")
@click.option('--shape', default=(75, 100), type = (int, int),
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
def train(image_path, model_path, gpu, latent_width, color_channels, batch,
          epoch, shape, kl_ratio):
    if not image_path[-1] == '/':
        image_path += '/'
    if not os.path.exists(os.path.dirname(image_path)):
            os.makedirs(os.path.dirname(image_path))
    if model_path[-1] == '/':
        click.echo("Model Path should be a file path not a directory path. " +
                   "Removing trailing '/' ...")
        model_path = model_path[:-1]

    file_paths = [os.path.join(image_path, f)
                  for f in os.listdir(image_path)
                  if os.path.isfile(os.path.join(image_path, f))
                  and not f.startswith('.')]

    iae = ImageAutoEncoder(img_width=shape[0],
                           img_height=shape[1],
                           encode_layers=[3000, 1000, 700, 300],
                           decode_layers=[500, 700, 1000, 3000],
                           rec_kl_ratio=kl_ratio,
                           flag_gpu=gpu,
                           latent_width=latent_width,
                           color_channels=color_channels)

    iae.load_images(file_paths)
    iae.fit(batch_size=batch, n_epochs=epoch)
    iae.dump(model_path)


@click.command(help='Generate images from a model.')
@click.argument('model_path', type=click.Path(resolve_path=False,
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
def generate(model_path, img_dir, number, extremity, mean, format):
    click.echo("Loading Model...")
    model = ImageAutoEncoder.load(model_path)
    click.echo("Model Loaded")
    click.echo("Saving Images to {0} as {1}".format(img_dir, format))
    if not img_dir[-1] == '/':
        img_dir += '/'
    if not os.path.exists(os.path.dirname(img_dir)):
            os.makedirs(os.path.dirname(img_dir))
    variance = extremity
    vec = np.random.normal(mean, variance,
                           (number, model.latent_dim)).astype('float32')

    reconstructed = model.inverse_transform(vec)*255.

    for i in range(number):
        im = Image.fromarray(np.uint8(reconstructed[i]))
        fname = "{0}.{1}".format(i, format)
        path = os.path.join(img_dir, fname)
        im.save(path)


fauxtograph.add_command(download)
fauxtograph.add_command(train)
fauxtograph.add_command(generate)

if __name__ == '__main__':
    fauxtograph()
