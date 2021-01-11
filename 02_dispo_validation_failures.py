import common  # local

import os
import re
from collections import defaultdict
from collections import Counter


if __name__ == '__main__':
    print('loading manifest')

    f = open('01_steam_manifest.yaml')
    manifest = common.warningless_yaml_load(f.read())
    f.close()

    print('iterating manifest')

    errors_by_app = defaultdict(list)  # appid -> error codes
    errors_by_code = defaultdict(list)  # error code -> appids

    coming_soon_regex = 'AssertionError: release date marked coming soon|AssertionError\n'  # from a temporary bug in code

    consolidate_error_codes = [
        'AssertionError: details is html response instead of json',
        'AssertionError: reviews is html response instead of json',
        'AssertionError: details success is false',
        'AssertionError: data is empty',
        'AssertionError: release_date is empty',
        'AssertionError: release_date -> date is empty',
        'AssertionError: dateparser returned falsey result',
        'AssertionError: name field is empty',
        'AssertionError: appid does not match queried appid',
        'AssertionError: reviews success is false',
        'AssertionError: query_summary does not exist',
        'AssertionError: query_summary -> total_positive is not an int',
        'AssertionError: query_summary -> total_positive is negative',
        'AssertionError: query_summary -> total_negative is not an int',
        'AssertionError: query_summary -> total_negative is negative',
        'AssertionError: query_summary -> total_reviews is not an int',
        'AssertionError: query_summary -> total_reviews is negative',
        'AssertionError: total reviews does not equal sum of positive and negative reviews',
        'json.decoder.JSONDecodeError: Expecting value: line 1 column 1 (char 0)',
        "requests.exceptions.ConnectionError: ('Connection aborted.', TimeoutError(10060, 'A connection attempt failed because the connected party did not properly respond after a period of time, or established connection failed because connected host has failed to respond', None, 10060, None))",
        "requests.exceptions.ConnectionError: ('Connection aborted.', ConnectionResetError(10054, 'An existing connection was forcibly closed by the remote host'",
    ]

    for app in manifest['applist']['apps']:
        appid = app['appid']
        name = app['name']

        output_folder = '02_app_info'
        validation_failure_folder = os.path.join(output_folder, f'{appid}_validation_failure')  # stores server responses which failed validation

        # read in all tracebacks for an app
        if os.path.exists(validation_failure_folder):
            for file_name in next(os.walk(validation_failure_folder))[2]:
                if re.search('_traceback\.txt$', file_name):
                    f = open(os.path.join(validation_failure_folder, file_name), encoding='utf-8')
                    traceback = f.read()
                    f.close()

                    error_code = traceback
                    for err in consolidate_error_codes:
                        if err in traceback:
                            error_code = err  # use shorthand since full traceback varies as line numbers evolve in sourcecode
                    if re.search(coming_soon_regex, traceback):
                        error_code = coming_soon_regex

                    errors_by_app[appid].append(error_code)

            # multiple compare entries for consistency
            # do not log repeated tracebacks more than once
            # log inconsistent tracebacks under a special category
            num_unique_tracebacks = len(set(errors_by_app[appid]))
            if num_unique_tracebacks == 1:
                errors_by_code[error_code].append(appid)
            else:
                # errors_by_code['multiple_unique_tracebacks'].append(appid)
                errors_by_code[tuple(sorted(set(errors_by_app[appid])))].append(appid)

    f = open('02_validation_failures_summary.txt', 'w', encoding='utf-8')
    for error_code, appids in sorted(errors_by_code.items(), key=lambda x: -len(x[1])):  # largest number of offenders listed first
        f.write(f'{len(appids)} offending apps\n')
        # f.write(f'{appids}\n')
        f.write(f'{[(appid, len(errors_by_app[appid])) for appid in appids]}\n')

        if type(error_code) == tuple:
            for err in error_code:
                f.write(f'{err.strip()}\n')
        else:
            f.write(error_code.strip())

        # if error_code == 'multiple_unique_tracebacks':
        #     for appid in appids:
        #         f.write(f'\n{appid}\n')
        #         for error, instances in sorted(Counter(errors_by_app[appid]).items(), key=lambda x: -x[1]):
        #             f.write(f'\t{instances}* {error.strip()}\n')            
        f.write('\n\n\n\n')
