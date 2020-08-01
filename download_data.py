import os
from pathlib import Path
import tensorflow as tf

def download_coco():

    coco_dir = Path(__file__).resolve().parent / 'COCO'
    coco_dir.mkdir()
    os.chdir(coco_dir)

    # Download caption annotation files
    annotation_folder = coco_dir / 'annotations'
    image_train_folder = coco_dir /'train2014'
    image_validation_folder = coco_dir / 'validation2014'

    captions_url = 'http://images.cocodataset.org/annotations/annotations_trainval2014.zip'
    train_url = 'http://images.cocodataset.org/zips/train2014.zip'
    valid_url = 'http://images.cocodataset.org/zips/val2014.zip'

    dirs = [annotation_folder, image_train_folder, image_validation_folder]
    labels = ['annonation', 'image training', 'image validation']
    files = ['captions.zip', 'train2014.zip', 'val2014.zip']
    urls = [captions_url, train_url, valid_url]

    for folder, label, file, url in zip(dirs, labels, files, urls):

        if not folder.exists():

            print('Downloading {} files'.format(label))

            zip_file = tf.keras.utils.get_file(file,
                                              cache_subdir=coco_dir,
                                              origin = url,
                                              extract = True)
            os.remove(zip_file)

def download_iuxray():

    iu_xray_dir = Path(__file__).resolve().parent / 'IU X-ray'
    iu_xray_dir.mkdir()
    os.chdir(iu_xray_dir)

    # Download caption annotation files
    # reports_folder = iu_xray_dir / ''
    # image_train_folder = iu_xray_dir /'train2014'

    reports_url = 'https://openi.nlm.nih.gov/imgs/collections/NLMCXR_reports.tgz'
    images_url = 'https://openi.nlm.nih.gov/imgs/collections/NLMCXR_png.tgz'

    labels = ['report', 'image']
    files = ['reports.tgz', 'images.tgz']
    urls = [reports_url, images_url]

    for label, file, url in zip(labels, files, urls):

        # if not folder.exists():

        print('Downloading {} files'.format(label))

        zip_file = tf.keras.utils.get_file(file,
                                          cache_subdir=iu_xray_dir,
                                          origin = url,
                                          extract = True)
        os.remove(zip_file)



if __name__ == '__main__':
    # download_coco()
    download_iuxray()
