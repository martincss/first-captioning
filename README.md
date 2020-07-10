# first-captioning

A project to play around and experiment with **image captioning**. Copied directly from https://www.tensorflow.org/tutorials/text/image_captioning, based on the architecture of Show, Attend & Tell https://arxiv.org/abs/1502.03044


## About dataset

The [COCO dataset](http://cocodataset.org) (2014) is a large-scale object detection, segmentation, and captioning dataset.
We will only be focused on captioning.

The training set contains 82783 images (13GB), each with at least 5 captions. The validation set contains 40504 images (6GB).


## About the image features extraction

Each image is preprocessed by first resizing to 299px by 299px, and then normalizing its pixel values to the range of -1 to 1.

The feature extraction is performed by using the InceptionV3 architecture using the ImageNet weights, with its top layers removed (that is, up to the last convolutional layer). Details of this architecture can be found in [here](https://medium.com/@sh.tsang/review-inception-v3-1st-runner-up-image-classification-in-ilsvrc-2015-17915421f77c)

Thus a vector of shape `(8,8,2048)` is extracted from each image.

This outputs are cached to disk, as keeping them in RAM would be faster but more
memory intensive (with 8x8x2048 floats per image).
Setting CACHE_FEATURES_BATCH_SIZE = 32, thus consuming close to 7000MiB of GPU RAM, the feature extraction takes about 36s per 1000 images.


## About training

Using 30000 training instances, with a batch size of 64 on an Nvidia 1080 Ti GPU, training time per epoch takes about 50-70s, with some overhead for the first epoch.

## Repo Structure

### Config and parameters

```config.py``` contains the specification of directories in which the training and validation datasets are to be stored. Also specifies the directories in which both training and validation image feature caches are stored. Finally, the specification of directories for storing training checkpoints and grid search results.
If run directly, these directories are safely created when not found.

```params.py``` contains global training parameters, such as the batch sizes, number of instances for training and validation sets, and number of epochs. Also some options for processing data such as vocabulary size. Most of these options differ when running in a cluster or local pc (this is hardoded in ```utils.py``` to a hostname).


### Data preparation
```train_data_preparation.py``` and ```valid_data_preparation.py``` execute the definition of training and validation Dataset objects. These are created by using loading functions from ```data_preparation.py``` and preprocessing functions from ```preprocess_encode_images.py``` and ```preprocess_tokenize_captions.py```

### Models
Model architectures are defined in ```model.py```.

### Caption Generation
Predictions for a single batch or instance can be done with functions from ```caption_generation.py```. 

### Evaluation and vaidation
```metrics.py``` contains the Metric objects for BLEU, METEOR and other evaluation metrics. The computation of these scores in the validation dataset is defined in ```evaluation.py```.

### Training and Grid Search
```training.py``` contains the training function for a single set of hyperparameters defining a model, while saving the learning curve including validation scores.
```grid_search.py``` can be run for performing a grid search, combining the data preparation, training and evaluation for each set of hyperparameters. The gird is defined in ```hyperparameters_space.py```
The execution of a grid search script creates a directory for the whole search, storing a progress log. This folder contains a subdirectory with each model's weights and a 'results' subdir where a json is saved for each combination, containing the detailed parameter information and learning curves.


















