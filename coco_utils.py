import json


def image_fnames_captions(captions_file, images_dir, partition):
    """
    Loads annotations file and return lists with each image's path and caption

    Arguments:
        partition: string
            either 'train' or 'val'

    Returns:
        all_captions: list of strings
            list with each image caption
        all_img_paths: list of paths as strings
            list with each image's path to file

    """


    with open(captions_file, 'r') as f:
        annotations = json.load(f)

    all_captions = []
    all_img_paths = []

    for annot in annotations['annotations']:
        caption = '<start> ' + annot['caption'] + ' <end>'
        image_id = annot['image_id']
        full_coco_image_path = images_dir / ('COCO_{}2014_'.format(partition) + \
                               '{:012d}.jpg'.format(image_id))

        all_img_paths.append(full_coco_image_path)
        all_captions.append(caption)

    return all_captions, all_img_paths
