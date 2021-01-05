import datetime
import warnings
import yaml

def parse_steam_date(date_string):
    # steam api provides dates formatted in several ways, its probably an unvalidated hand-entered string field
    # returns a datetime.datetime object

    field_formats = [
        '%b %d, %Y',
        '%b %Y',
        '%d %b, %Y',
        '%Y %b %d',
        '%Y年%b月%d日',
        '%Y 年 %b 月 %d 日',
    ]

    for format_code in field_formats:
        try:
            data = datetime.datetime.strptime(date_string, format_code)
        except:
            continue
        else:
            return data

    raise ValueError('All date formats failed to parse date string, ' + date_string)


def warningless_yaml_load(string):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return yaml.load(string)
