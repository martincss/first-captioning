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
