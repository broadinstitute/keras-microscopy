import numpy
import skimage.transform
from scipy.ndimage.filters import gaussian_filter
from scipy.ndimage.interpolation import map_coordinates

"""
Note: Standardization and transforms assumes that x comes in WxHxC format from the reader
"""


def horizontal_flip(prob):
    assert 0. < prob < 1.

    def f(x):
        if numpy.random.random() < prob:
            return numpy.fliplr(x)
        return x

    return f


def vertical_flip(prob):
    assert 0. < prob < 1.

    def f(x):
        if numpy.random.random() < prob:
            return numpy.flipud(x)
        return x

    return f


def rotate90(prob):
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

    return lambda x: skimage.transform.resize(x, numpy.multiply(x.shape, axes_scale), **kwargs)



def rotate():
    def f(x):
        k = numpy.pi / 360 * numpy.random.uniform(-360, 360)
        return skimage.transform.rotate(x, k)

    return f


"""
def clip_patches(size, n):
    
    Function clips randomly n patches from the source and target image
    :param size: a tuple new size
    :param number: 
    :return: function that from WxHxC produces nxsize[0]xsize[1]xC
   
    assert isinstance(size, tuple)
    assert len(size) == 2

    def f(x):
        cx = np.random.randint(0, x.shape[0] - size[0], n)
        cy = np.random.randint(0, x.shape[1] - size[1], n)
        if 3 == len(x.shape):
            patches = np.zeros((n, size[0], size[1],
                                x.shape[1]), dtype=np.float32)
        else:
            patches = np.zeros((n, size[0], size[1]), dtype=np.float32)

        for i in range(n):
            patches[i] = x[cx[i]:cx[i] + size[0], cy[i]:cy[i] + size[1]]

        return patches

    return f
"""


def clip_patche(size):
    assert isinstance(size, tuple)
    assert len(size) == 2

    def f(x):
        cx = numpy.random.randint(0, x.shape[0] - size[0])
        cy = numpy.random.randint(0, x.shape[1] - size[1])
        return x[cx:cx + size[0], cy:cy + size[1]]

    return f


def ellastic_transfrom(alpha, sigma, bg_value):
    def f(x):
        dx = gaussian_filter((numpy.random(x.shape) * 2 - 1),
                             sigma, mode="constant", cval=0) * alpha
        dy = gaussian_filter((numpy.random(x.shape) * 2 - 1),
                             sigma, mode="constant", cval=0) * alpha

        cx, cy = numpy.meshgrid(numpy.arange(x.shape[0]), numpy.arange(x.shape[1]), indexing='ij')
        indices = numpy.reshape(cx + dx, (-1, 1)), numpy.reshape(cy + dy, (-1, 1))

        dest = numpy.zeros_like(x)
        for j in range(x.shape[2]):
            dest[:, :, j] = map_coordinates(x[:, :, j], \
                                            indices, cval=bg_value, order=1).reshape(x.shape)
        return dest

    return f
