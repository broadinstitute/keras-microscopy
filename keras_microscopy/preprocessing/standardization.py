import numpy as np
import skimage.filters.gaussian

"""
Note: Standardization and transforms assumes that x comes in WxHxC format
"""


# normalization make the image to range [-1,1]
def normalization(means, max_values):
    def f(x):
        return (x - means) / max_values

    return f


# TODO: add noise
# TODO: reduce noise
# TODO: contrast
# TODO: equalization
# TODO: correct_distortion,
# TODO: correct_uneven_illumination,
# TODO: correct_vignetting
# TODO: desaturate
# TODO: equalize
# TODO: remove_chromatic_aberration

# blur
def gaussian_blur(mean_sigma, variance_sigma):
    def f(x):
        sigma = np.random.normal(mean_sigma, variance_sigma)
        return skimage.filters.gaussian(x, sigma)

    return f
