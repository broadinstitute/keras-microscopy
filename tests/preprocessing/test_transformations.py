import numpy

from keras_imaging.preprocessing import transformations


def test_rescale_last(mocker):
    image_data_format = mocker.patch("keras.backend.image_data_format")
    image_data_format.return_value = "channels_last"
    rescale = transformations.rescale(0.5, mode='reflect')
    img = numpy.random.rand(100, 100, 3)
    new = rescale(img)
    assert new.shape == (50, 50, 3)


def test_rescale_uint8(mocker):
    image_data_format = mocker.patch("keras.backend.image_data_format")
    image_data_format.return_value = "channels_first"
    rescale = transformations.rescale(0.5, mode='reflect')
    img = numpy.random.randint(256, size=(100, 100, 3)).astype(numpy.uint8)
    new = rescale(img)
    assert new.shape == (50, 50, 3)


def test_clip():
    clipper = transformations.clip_patch(size=(10, 10))
    img = numpy.random.randint(256, size=(100, 100, 5)).astype(numpy.uint8)
    new = clipper(img)
    assert new.shape == (10, 10, 5)


def test_vertical_flip():
    flipper = transformations.flip_vertically(0.9999)
    img = numpy.random.randint(256, size=(50, 50, 5)).astype(numpy.uint8)
    new = flipper(img)
    assert new.shape == (50, 50, 5)
    flipper = transformations.flip_vertically(0.00001)
    img = numpy.random.randint(256, size=(50, 50, 5)).astype(numpy.uint8)
    new = flipper(img)
    assert new.shape == (50, 50, 5)


def test_horizontal_flip():
    flipper = transformations.flip_horizontally(0.9999)
    img = numpy.random.randint(256, size=(50, 50, 5)).astype(numpy.uint8)
    new = flipper(img)
    assert new.shape == (50, 50, 5)
    flipper = transformations.flip_horizontally(0.00001)
    img = numpy.random.randint(256, size=(50, 50, 5)).astype(numpy.uint8)
    new = flipper(img)
    assert new.shape == (50, 50, 5)


def test_rotate90():
    rotate = transformations.rotate90(0.0001)
    img = numpy.random.randint(256, size=(50, 50, 5)).astype(numpy.uint8)
    new = rotate(img)
    assert new.shape == (50, 50, 5)
    rotate = transformations.rotate90(0.9999)
    img = numpy.random.randint(256, size=(50, 50, 5)).astype(numpy.uint8)
    new = rotate(img)
    assert new.shape == (50, 50, 5)


def test_rotate():
    rotate = transformations.rotate()
    img = numpy.random.randint(256, size=(50, 50, 5)).astype(numpy.uint8)
    new = rotate(img)
    assert new.shape == (50, 50, 5)


def test_ellastic_transfrom():
    ellastic = transformations.ellastic_transfrom(120., 2., 1.)
    img = numpy.random.randint(256, size=(50, 50, 5)).astype(numpy.uint8)
    new = ellastic(img)
    assert new.shape == (50, 50, 5)
