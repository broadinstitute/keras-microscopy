import numpy

import keras_imaging.preprocessing.transformations


def test_rescale_last(mocker):
    image_data_format = mocker.patch("keras.backend.image_data_format")
    image_data_format.return_value = "channels_last"
    rescale = keras_imaging.preprocessing.transformations.rescale(0.5, mode='reflect')
    img = numpy.random.rand(100, 100, 3)
    new = rescale(img)
    assert new.shape == (50, 50, 3)


def test_rescale_uint8(mocker):
    image_data_format = mocker.patch("keras.backend.image_data_format")
    image_data_format.return_value = "channels_first"
    rescale = keras_imaging.preprocessing.transformations.rescale(0.5, mode='reflect')
    img = numpy.random.randint(256, size=(100, 100, 3)).astype(numpy.uint8)
    new = rescale(img)
    assert new.shape == (50, 50, 3)


def test_clip():
    clipper = keras_imaging.preprocessing.transformations.clip_patche(size=(10, 10))
    img = numpy.random.randint(256, size=(100, 100, 5)).astype(numpy.uint8)
    new = clipper(img)
    assert new.shape == (10, 10, 5)


def test_vertical_flip():
    flipper = keras_imaging.preprocessing.transformations.flip_vertically(0.9999)
    img = numpy.random.randint(256, size=(50, 50, 5)).astype(numpy.uint8)
    new = flipper(img)
    assert new.shape == (50, 50, 5)
    flipper = keras_imaging.preprocessing.transformations.flip_vertically(0.00001)
    img = numpy.random.randint(256, size=(50, 50, 5)).astype(numpy.uint8)
    new = flipper(img)
    assert new.shape == (50, 50, 5)


def test_horizontal_flip():
    flipper = keras_imaging.preprocessing.transformations.flip_horizontally(0.9999)
    img = numpy.random.randint(256, size=(50, 50, 5)).astype(numpy.uint8)
    new = flipper(img)
    assert new.shape == (50, 50, 5)
    flipper = keras_imaging.preprocessing.transformations.flip_horizontally(0.00001)
    img = numpy.random.randint(256, size=(50, 50, 5)).astype(numpy.uint8)
    new = flipper(img)
    assert new.shape == (50, 50, 5)


def test_rotate90():
    rotate = keras_imaging.preprocessing.transformations.rotate90(0.0001)
    img = numpy.random.randint(256, size=(50, 50, 5)).astype(numpy.uint8)
    new = rotate(img)
    assert new.shape == (50, 50, 5)
    rotate = keras_imaging.preprocessing.transformations.rotate90(0.9999)
    img = numpy.random.randint(256, size=(50, 50, 5)).astype(numpy.uint8)
    new = rotate(img)
    assert new.shape == (50, 50, 5)


def test_rotate():
    rotate = keras_imaging.preprocessing.transformations.rotate()
    img = numpy.random.randint(256, size=(50, 50, 5)).astype(numpy.uint8)
    new = rotate(img)
    assert new.shape == (50, 50, 5)


def test_ellastic_transfrom():
    ellastic = keras_imaging.preprocessing.transformations.ellastic_transfrom(120., 2., 1.)
    img = numpy.random.randint(256, size=(50, 50, 5)).astype(numpy.uint8)
    new = ellastic(img)
    assert new.shape == (50, 50, 5)