import imblearn.over_sampling
import imblearn.under_sampling
import keras.backend
import numpy
import scipy.ndimage
import skimage.filters
import skimage.transform
import skimage.util

import keras_imaging.preprocessing


class ImageGenerator(object):
    def __init__(
        self,
        transformations=[],
        standardizations=[]
    ):
        """
        :param transforms:

        :param standardizations:

        :param reader_name:

        """

        self.transformations = transformations
        self.standardizations = standardizations


    def flow_from_directory(
        self,
        directory,
        batch_size=32,
        sampling_method=None,
        seed=None,
        shape=(224, 224, 3),
        shuffle=True
    ):
        """
        :param directory:

        :param batch_size: int

        :param sampling_method:

        :param seed: int or None; optional

        :param shape: tuple of ints

        :param shuffle: boolean; optional

        :return:

        :rtype: keras_imaging.preprocessing.DirectoryIterator
        """
        if sampling_method == "oversample":
            sampling_method = imblearn.over_sampling.RandomOverSampler(random_state=seed)
        elif sampling_method == "undersample":
            sampling_method = imblearn.under_sampling.RandomUnderSampler(random_state=seed)

        return keras_imaging.preprocessing.DirectoryIterator(
            batch_size=batch_size,
            directory=directory,
            generator=self,
            sampling_method=sampling_method,
            seed=seed,
            shape=shape,
            shuffle=shuffle
        )

    def standardize(self, x):
        """
        :param x:

        :return:
        """
        for s in self.standardizations:
            x = s(x)

        return x

    def transform(self, x ):
        """
        :param x: image

        :return:
        """
        for t in self.transformations:
            x = t(x)

        return x
