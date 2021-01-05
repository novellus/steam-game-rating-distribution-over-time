import common  # local

import os
import re
from collections import defaultdict


if __name__ == '__main__':
    print('loading manifest')

    f = open('01_steam_manifest.yaml')
    manifest = common.warningless_yaml_load(f.read())
    f.close()

    print('iterating manifest')

    tracebacks = defaultdict(list)
    tracebacks_reverse = defaultdict(list)

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

                    tracebacks[appid].append(traceback)

            # multiple compare entries for consistency
            # do not log repeated tracebacks more than once
            # log inconsistent tracebacks under a special category
            num_unique_tracebacks = len(set(tracebacks[appid]))
            if num_unique_tracebacks == 1:
                tracebacks_reverse[traceback].append(appid)
            else:
                tracebacks_reverse['multiple_unique_tracebacks'].append(appid)

    f = open('02_validation_failures_summary.txt', 'w')
    for traceback, appids in sorted(tracebacks_reverse.items(), key=lambda x: -len(x[1])):  # largest number of offenders listed first
        f.write(f'--A-- {len(appids)} offending apps\n')
        f.write(f'{appids}\n')
        f.write(traceback)
        f.write('\n\n\n\n')
