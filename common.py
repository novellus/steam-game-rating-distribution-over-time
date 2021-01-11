import dateparser
import re
import warnings
import yaml

def warningless_yaml_load(string):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return yaml.load(string)

def parse_date(string):
    # bypass https://github.com/scrapinghub/dateparser/issues/866
    if re.search('^[\d\s年月日]*$', string):
        string = re.sub('[年月日]', ' ', string)

    return dateparser.parse(string)
