# first-captioning

A project to play around and experiment with **image captioning**. Copied directly from https://www.tensorflow.org/tutorials/text/image_captioning, based on the architecture of Show, Attend & Tell https://arxiv.org/abs/1502.03044


# About dataset

The [COCO dataset](http://cocodataset.org) (2014) is a large-scale object detection, segmentation, and captioning dataset.
We will only be focused on captioning.

The training set contains 82783 images (13GB), each with at least 5 captions. The validation set contains 40504 images (6GB).


# About the image features extraction

Each image is preprocessed by first resizing to 299px by 299px, and then normalizing its pixel values to the range of -1 to 1.

The feature extraction is performed by using the InceptionV3 architecture using the ImageNet weights, with its top layers removed (that is, up to the last convolutional layer). Details of this architecture can be found in [here](https://medium.com/@sh.tsang/review-inception-v3-1st-runner-up-image-classification-in-ilsvrc-2015-17915421f77c)

Thus a vector of shape `(8,8,2048)` is extracted from each image.

This outputs are cached to disk, as keeping them in RAM would be faster but more
memory intensive (with 8x8x2048 floats per image).
Setting CACHE_FEATURES_BATCH_SIZE = 32, thus consuming close to 7000MiB of GPU RAM, the feature extraction takes about 36s per 1000 images.


# About training

Using 30000 training instances, with a batch size of 64 on an Nvidia 1080 Ti GPU, training time per epoch takes about 50-70s, with some overhead for the first epoch.
