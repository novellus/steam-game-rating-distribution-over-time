import datetime

def parse_steam_date(date_string):
    # steam api provides dates formatted in several ways, its probably an unvalidated hand-entered string field
    # returns a datetime.datetime object

    field_formats = [
        '%b %d, %Y',
        '%b %Y',
        '%d %b, %Y',
        '%Y %b %d',
        '%Y年%b月%d日',
    ]

    for format_code in field_formats:
        try:
            data = datetime.datetime.strptime(date_string, format_code)
        except:
            continue
        else:
            return data

    raise ValueError('All date formats failed to parse data string, ' + date_string)
