import os
import tensorflow as tf

# Download caption annotation files
annotation_folder = '/annotations/'
image_train_folder = '/train2014/'
image_validation_folder = '/validation2014/'

captions_url = 'http://images.cocodataset.org/annotations/annotations_trainval2014.zip'
train_url = 'http://images.cocodataset.org/zips/train2014.zip'
valid_url = 'http://images.cocodataset.org/zips/val2014.zip'


# Download annotation files
if not os.path.exists(os.path.abspath('.') + annotation_folder):

    print('Downloading annotation files')

    annotation_zip = tf.keras.utils.get_file('captions.zip',
                                          cache_subdir=os.path.abspath('.'),
                                          origin = captions_url,
                                          extract = True)
    os.remove(annotation_zip)

# Download image files
if not os.path.exists(os.path.abspath('.') + image_train_folder):

    print('Downloading image training files')

    image_zip = tf.keras.utils.get_file('train2014.zip',
                                      cache_subdir=os.path.abspath('.'),
                                      origin = train_url,
                                      extract = True)
    os.remove(image_zip)


if not os.path.exists(os.path.abspath('.') + image_validation_folder):

    print('Downloading image validation files')

    image_zip = tf.keras.utils.get_file('val2014.zip',
                                      cache_subdir=os.path.abspath('.'),
                                      origin = 'valid_url,
                                      extract = True)

    os.remove(image_zip)
