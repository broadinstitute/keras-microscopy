import numpy as np

"""
Must output an numpy array WxHxC
"""

try:
    import skimage.io

    skimage_available = True


    def skimage_reader():
        """
        Creates a reader using skimage library
        :return: function for reading images, returning numpy array
        """

        def f(filename):
            return skimage.io.imread(filename)

        return f
except ImportError, e:
    skimage_available = False

try:
    from PIL import Image

    pillow_available = True


    def pillow_reader():
        """
        Creates a reader using pillow library
        :return: function for reading images, returning numpy array
        """

        def f(filename):
            return np.asarray(Image.open(filename))

        return f
except ImportError, e:
    pillow_available = False

try:
    from libtiff import TIFF

    libtiff_available = True


    def tiff_reader():
        """
        Creates a reader using libtiff library
        :return: function for reading images, returning numpy array
        """

        def f(filename):
            tif = TIFF.open(filename, mode='r')
            return np.array(tif.read_image())

        return f

except ImportError, e:
    libtiff_available = False


def create_reader(reader_name):
    if "skimage" == reader_name:
        if not skimage_available:
            raise RuntimeError("Skimage is not installed! Please install skimage.")
        return skimage_reader()

    if "pillow" == reader_name:
        if not pillow_available:
            raise RuntimeError("Pillow is not installed! Please install pillow.")
        return pillow_reader()

    if "tiff" == reader_name:
        if not libtiff_available:
            raise RuntimeError("Libtiff is not installed! Please install libtiff.")
        return tiff_reader()

    raise NotImplementedError("Requested reader is not implemented!")
