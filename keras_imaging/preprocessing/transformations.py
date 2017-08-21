import numpy
import skimage.transform
from scipy.ndimage.filters import gaussian_filter
from scipy.ndimage.interpolation import map_coordinates

"""
Note: Standardization and transforms assumes that x comes in WxHxC format from the reader
"""


def flip_horizontally(prob=0.5):
    assert 0. < prob < 1.

    def f(x):
        if numpy.random.random() < prob:
            return numpy.fliplr(x)
        return x

    return f


def flip_vertically(prob=0.5):
    assert 0. < prob < 1.

    def f(x):
        if numpy.random.random() < prob:
            return numpy.flipud(x)
        return x

    return f


def rotate90(prob=0.5):
    assert 0. < prob < 1.

    def f(x):
        if numpy.random.random() < prob:
            return numpy.rot90(x, 2, axes=(0, 1))
        return x

    return f


def rescale(scale, **kwargs):
    """
    Rescales the image according to the scale ratio.
    :param scale: The scalar to rescale the image by.
    :param kwargs: Additional arguments for skimage.transform.resize.
    :return: The rescale function.
    """

    axes_scale = (scale, scale, 1.0)

    def f(x):
        return skimage.transform.resize(x, numpy.multiply(x.shape, axes_scale), **kwargs)

    return f


def rotate():
    def f(x):
        k = numpy.pi / 360 * numpy.random.uniform(-360, 360)
        return skimage.transform.rotate(x, k)

    return f


def clip_patch(size):
    assert len(size) == 2

    def f(x):
        cx = numpy.random.randint(0, x.shape[0] - size[0])
        cy = numpy.random.randint(0, x.shape[1] - size[1])
        return x[cx:cx + size[0], cy:cy + size[1]]

    return f


def ellastic_transfrom(alpha, sigma, bg_value):
    def f(x):
        dx = gaussian_filter((numpy.random.rand(*x.shape[:2]) * 2 - 1),
                             sigma, mode="constant", cval=0) * alpha
        dy = gaussian_filter((numpy.random.rand(*x.shape[:2]) * 2 - 1),
                             sigma, mode="constant", cval=0) * alpha

        cx, cy = numpy.meshgrid(numpy.arange(x.shape[0]), numpy.arange(x.shape[1]), indexing='ij')
        indices = numpy.reshape(cx + dx, (-1, 1)), numpy.reshape(cy + dy, (-1, 1))

        dest = numpy.zeros_like(x)
        for j in range(x.shape[2]):
            dest[:, :, j] = map_coordinates(x[:, :, j], indices, cval=bg_value, order=1).reshape(x.shape[:2])
        return dest

    return f
