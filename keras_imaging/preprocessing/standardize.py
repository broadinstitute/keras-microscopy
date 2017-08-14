import keras
import numpy
import skimage.exposure
import skimage.restoration
import skimage.transform


def desaturate(ratio):
    """
    Fades the image to white. Higher ratio means more fading.
    :param ratio: The ratio to fade by, between 0 and 1.
    :return: The desaturate function.
    """
    return lambda x: x + (x.max() - x) * ratio


def equalize(**kwargs):
    """
    Equalizes the image histogram, per channel.
    :param kwargs: Additional arguments for skimage.exposure.equalize_hist.
    :return: The equalize function.
    """

    def f(x):
        if keras.backend.image_data_format() == 'channels_last':
            x = numpy.moveaxis(x, -1, 0)

        y = numpy.empty_like(x, dtype=numpy.float64)

        for index, img in enumerate(x):
            y[index] = skimage.exposure.equalize_hist(img, **kwargs)

        if keras.backend.image_data_format() == 'channels_last':
            y = numpy.moveaxis(y, 0, -1)

        return y

    return f


def reduce_noise(**kwargs):
    """
    Reduces noise in the image.
    :param kwargs: Additional arguments for skimage.restoration.denoise_bilateral.
    :return: The reduce_noise function.
    """

    def f(x):
        if keras.backend.image_data_format() == 'channels_last':
            x = numpy.moveaxis(x, -1, 0)

        y = numpy.empty_like(x, dtype=numpy.float64)

        for index, img in enumerate(x):
            y[index] = skimage.restoration.denoise_bilateral(img, **kwargs)

        if keras.backend.image_data_format() == 'channels_last':
            y = numpy.moveaxis(y, 0, -1)

        return y

    return f


def normalization(means, max_values):
    """
    normalization make the image to range [-1,1]
    :param means: means of image values across the dataset
    :param max_values: maximum value
    :return: normalized image
    """

    def f(x):
        return (x - means) / max_values

    return f


def gaussian_blur(mean_sigma, variance_sigma=0.0):
    """
    :param mean_sigma: mean of expected value
    :param variance_sigma: 
    :return: 
    """

    def f(x):
        sigma = mean_sigma
        if variance_sigma is not 0.0:
            sigma = numpy.random.normal(mean_sigma, variance_sigma)
        return skimage.filters.gaussian(x, sigma)

    return f
