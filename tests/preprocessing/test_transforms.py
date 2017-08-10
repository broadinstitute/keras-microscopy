import numpy

import keras_microscopy.preprocessing.transforms

def test_rescale_last(mocker):
    image_data_format = mocker.patch("keras.backend.image_data_format")
    image_data_format.return_value = "channels_last"
    rescale = keras_microscopy.preprocessing.transforms.rescale(0.5, mode='reflect')
    img = numpy.random.rand(100, 100, 3)
    new = rescale(img)
    assert new.shape == (50, 50, 3)

def test_rescale_uint8(mocker):
    image_data_format = mocker.patch("keras.backend.image_data_format")
    image_data_format.return_value = "channels_first"
    rescale = keras_microscopy.preprocessing.transforms.rescale(0.5, mode='reflect')
    img = numpy.random.randint(256, size=(100, 100,3)).astype(numpy.uint8)
    new = rescale(img)
    assert new.shape == (50, 50,3)

def test_clip():
    clipper = keras_microscopy.preprocessing.transforms.clip_patche(size=(10,10))
    img = numpy.random.randint(256, size=(100, 100, 5)).astype(numpy.uint8)
    new = clipper(img)
    assert new.shape == (10, 10, 5)