import json
from bs4 import BeautifulSoup


def get_report(fname):
    """
    Parses an xml report and extracts information on image ids, impression,
    findings, and MeSH/MTI tags

    Params:
        fname: path or filename of xml report

    Returns:
        report_info: dict
            contains the useful report information

    """

    report_info = {}

    with open(fname, 'r') as handle:

        report = BeautifulSoup(handle, features='lxml')
        report_info['impression'] = report.find('abstracttext',
                                              {'label': 'IMPRESSION'}).text
        report_info['findings'] = report.find('abstracttext',
                                              {'label': 'FINDINGS'}).text

        report_info['ids'] = [tag['id'] for tag in \
                                            report.find_all('parentimage')]

        report_info['manual_tags'] = [tag.text for tag in report.find_all('major')]
        report_info['auto_tags'] = [tag.text for tag in report.find_all('automatic')]

    return report_info


def generate_image_caption_pairs(base_dir, save_to_json=False):
    """
    Reads through all xml reports and extracts images and captions for each one,
    putting them together in an image-caption dictionary.

    Each caption joins both "IMPRESSION" and "FINDINGS" sections.

    Params
        base_dir: Path
            path to the IU X-ray directory, containing an 'images' and 'reports'
            folders
        save_to_json: boolean
            option to save the image-caption dict to a json file on base_dir

    Returns
        img_caption_pairs: dict
            dictionary mapping each full image path to its caption

    """

    images_dir = base_dir / 'images'
    reports_dir = base_dir / 'reports'

    img_caption_pairs = {}

    for report_fname in reports_dir.glob('*.xml'):

        data = get_report(report_fname)

        for id in data['ids']:
            img_path = images_dir.resolve() / (id + '.png')
            img_caption_pairs[str(img_path)] = data['impression'] + ' ' + \
                                               data['findings']

    if save_to_json:
        with (base_dir / 'image_caption_pairs.json').open(mode = 'w') as f:
            json.dump(img_caption_pairs, f)

    return img_caption_pairs
