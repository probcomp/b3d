import numpy as np
import matplotlib.pyplot as plt
import imageio
from sklearn.utils import Bunch
from PIL import Image
import io
import jax.numpy as jnp
from IPython.display import Image as IPImage, display


def fig_to_image(fig):
    """Convert a Matplotlib figure to a PIL Image and return it"""
    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    buf.seek(0)
    img = Image.open(buf)
    return img


def save_as_gif(fname, images, fps=10, loop=0):
    """Save a list of images as a gif"""
    if isinstance(images[0], np.ndarray) or isinstance(images[0], jnp.ndarray):
        images = [Image.fromarray(im) for im in images]

    if not isinstance(images[0], Image.Image):
        raise Exception("images need to be `(j)numpy.ndarray` or `PIL.Image.Image`")

    images[0].save(
        fname,
        save_all=True,
        append_images=images[1:],
        optimize=False,
        duration=1000.0 / fps,
        loop=loop,
    )

    return fname


def display_gif(fname):
    return display(IPImage(data=open(fname, "rb").read(), format="png"))


def save_and_display_gif(fname, images, fps=10, loop=0):
    return display_gif(save_as_gif(fname, images, fps=fps, loop=loop))

