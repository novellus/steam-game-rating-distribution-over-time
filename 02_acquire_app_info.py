import common  # local

import dateparser
import json
import os
import random
import re
import requests
import shutil
import sys
import time
import traceback
import yaml


def retrieve_info(appid):
    # retrieve general data, including release date
    # this api is undocumented
    app_details = requests.get('http://store.steampowered.com/api/appdetails?appids=' + str(appid)).text

    # retriev review data
    # docs at https://partner.steamgames.com/doc/store/getreviews
    # filter updated => is more of a sorting key than a filter. Will disable the automatic sliding window which limits returned reviews, so that all reviews will be included
    # purchase_type steam => eliminate non-purchasers
    # num_per_page 1 => each query_summary contains stats for all reviews, so I don't need to query all review contents
    app_reviews = requests.get('http://store.steampowered.com/appreviews/' + str(appid) + '?json=1&filter=updated&language=all&purchase_type=steam&review_type=all&cursor=*&num_per_page=1').text

    return app_details, app_reviews


def parse_responses(appid, app_details, app_reviews):
    # turns raw server responses into parsed data structure. May raise errors if data is not parsable, such as an html error response instead of json response
    app_details = json.loads(app_details)[str(appid)]  # only one entry in this dict
    app_reviews = json.loads(app_reviews)

    unified_structure = {
        'details': app_details,
        'reviews': app_reviews,
    }

    return unified_structure


def validate_text_data(appid, app_details, app_reviews):
    # will return one of
    #   valid = true, data = resolved structure
    #   valid = false, data = (short error description, formatted traceback string)

    try:
        html_openeing_tag = '<!DOCTYPE html>'
        assert not re.search(f'^.{{0,10}}{re.escape(html_openeing_tag)}\s*\n', app_details), 'details is html response instead of json'
        assert not re.search(f'^.{{0,10}}{re.escape(html_openeing_tag)}\s*\n', app_reviews), 'reviews is html response instead of json'

        # try to parse the data as json
        data = parse_responses(appid, app_details, app_reviews)
    except:
        return False, (', '.join([str(x) for x in sys.exc_info()[0:2]]), traceback.format_exc())

    # yield validity to secondary validater
    valid, data = validate_parsed_data(appid, data)

    return valid, data


def validate_parsed_data(appid, data):
    try:
        # validate details
        assert data['details']['success'] == True, 'details success is false'
        assert data['details']['data'], 'data is empty'

        assert data['details']['data']['release_date'], 'release_date is empty'
        if not data['details']['data']['release_date']['coming_soon']:  # field should always exist, and is boolean. Indicates game is not playable. Is False for early access games.
            assert data['details']['data']['release_date']['date'], 'release_date -> date is empty'
            dateparser.parse(data['details']['data']['release_date']['date'])  # try to parse date

        assert data['details']['data']['name'], 'name field is empty'
        assert data['details']['data']['steam_appid'] == appid, 'appid does not match queried appid'  # some appids redirect to another appid. This removes the duplicates

        # validate reviews
        assert data['reviews']['success'] == True, 'reviews success is false'
        assert data['reviews']['query_summary'], 'query_summary does not exist'

        assert type(data['reviews']['query_summary']['total_positive']) == int, 'query_summary -> total_positive is not an int'
        assert data['reviews']['query_summary']['total_positive'] >= 0, 'query_summary -> total_positive is negative'

        assert type(data['reviews']['query_summary']['total_negative']) == int, 'query_summary -> total_negative is not an int'
        assert data['reviews']['query_summary']['total_negative'] >= 0, 'query_summary -> total_negative is negative'

        assert type(data['reviews']['query_summary']['total_reviews']) == int, 'query_summary -> total_reviews is not an int'
        assert data['reviews']['query_summary']['total_reviews'] >= 0, 'query_summary -> total_reviews is negative'

        assert data['reviews']['query_summary']['total_reviews'] == data['reviews']['query_summary']['total_positive'] + data['reviews']['query_summary']['total_negative'], 'total reviews does not equal sum of positive and negative reviews'

    except:
        return False, (', '.join([str(x) for x in sys.exc_info()[0:2]]), traceback.format_exc())

    else:
        return True, data


def next_unique_validation_failure_number(validation_failure_folder):
    next_unique_number = 1
    if not os.path.exists(validation_failure_folder):
        os.mkdir(validation_failure_folder)
    else:
        for file_name in next(os.walk(validation_failure_folder))[2]:
            regex_match = re.search('^response(\d+)_', file_name)
            if regex_match:
                existing_number = int(regex_match.group(1))
                next_unique_number = max(next_unique_number, 1 + existing_number)
    return next_unique_number


if __name__ == '__main__':
    print('loading manifest')

    f = open('01_steam_manifest.yaml')
    manifest = common.warningless_yaml_load(f.read())
    f.close()

    # shuffle manifest order to minimize order based errors
    # for instance, if attempting to access data on a particular app results in steam api locking us out for a period of time
    # some apps always return invalid responses, but this scenario is hypothetical
    random.shuffle(manifest['applist']['apps'])

    # dev TODO remove
    # manifest['applist']['apps'] = [{'appid':427520, 'name': 'DEV FACTORIO'}]

    print('iterating manifest')

    output_folder = '02_app_info'
    if not os.path.exists(output_folder):
        os.mkdir(output_folder)

    for app in manifest['applist']['apps']:
        appid = app['appid']
        name = app['name']

        text_details_path = os.path.join(output_folder, f'{appid}_details.txt')  # stores raw server response
        text_reviews_path = os.path.join(output_folder, f'{appid}_reviews.txt')  # stores raw server response
        yaml_path = os.path.join(output_folder, f'{appid}.yaml')  # stores parsed server responses

        validation_failure_folder = os.path.join(output_folder, f'{appid}_validation_failure')  # stores server responses which failed validation
        text_details_validation_failure_path = lambda unique_num: os.path.join(validation_failure_folder, f'response{unique_num:02d}_details.txt')  # stores raw server response
        text_reviews_validation_failure_path = lambda unique_num: os.path.join(validation_failure_folder, f'response{unique_num:02d}_reviews.txt')  # stores raw server response
        traceback_validation_failure_path = lambda unique_num: os.path.join(validation_failure_folder, f'response{unique_num:02d}_traceback.txt')  # stores traceback on validation failure

        # revalidate existing data as validation is expected to evolve between runtimes
        if os.path.exists(yaml_path): # previous validation succeeded
            f=open(yaml_path)
            data = common.warningless_yaml_load(f.read())
            f.close()

            valid, data = validate_parsed_data(appid, data)  # validation expected to evolve between runtimes
            if valid:
                continue
            else:
                short_error_description, error_traceback = data

                os.remove(yaml_path)
                shutil.move(text_details_path, text_details_validation_failure_path(1))
                shutil.move(text_reviews_path, text_reviews_validation_failure_path(1))

                f = open(traceback_validation_failure_path(next_unique_number), 'w', encoding='utf-8')
                f.write(error_traceback)
                f.close()

                print(f'\nWARN: previously validated responses marked as invalid\n\t{appid}\n\t{name[:100]}\n\t{short_error_description}\n')

        # retrieve data from server
        # on request failure: log traceback, sleep for 10 minutes, and then resume
        print('retrieving', appid)
        try:
            app_details, app_reviews = retrieve_info(appid)
        except:
            short_error_description, error_traceback = (', '.join([str(x) for x in sys.exc_info()[0:2]]), traceback.format_exc())

            next_unique_number = next_unique_validation_failure_number(validation_failure_folder)
            f = open(traceback_validation_failure_path(next_unique_number), 'w', encoding='utf-8')
            f.write(error_traceback)
            f.close()

            print(f'\nWARN: retrieval exception, sleeping for 10 minutes\n\t{appid}\n\t{name[:100]}\n\t{short_error_description}\n')
            
            time.sleep(10*60)
            continue

        valid, data = validate_text_data(appid, app_details, app_reviews)

        if valid:
            f = open(text_details_path, 'w', encoding='utf-8')
            f.write(app_details)
            f.close()

            f = open(text_reviews_path, 'w', encoding='utf-8')
            f.write(app_reviews)
            f.close()
            
            f = open(yaml_path, 'w', encoding='utf-8')
            f.write(yaml.dump(data))
            f.close()

            if os.path.exists(validation_failure_folder):
                shutil.rmtree(validation_failure_folder)

        else:
            short_error_description, error_traceback = data

            # save server response under unique name
            next_unique_number = next_unique_validation_failure_number(validation_failure_folder)

            f = open(text_details_validation_failure_path(next_unique_number), 'w', encoding='utf-8')
            f.write(app_details)
            f.close()

            f = open(text_reviews_validation_failure_path(next_unique_number), 'w', encoding='utf-8')
            f.write(app_reviews)
            f.close()

            f = open(traceback_validation_failure_path(next_unique_number), 'w', encoding='utf-8')
            f.write(error_traceback)
            f.close()

            # TODO run comparisons with previous failure texts if they exist
            print(f'\nWARN: response marked as invalid\n\t{appid}\n\t{name[:100]}\n\t{short_error_description}\n')

        # steam api is limited to 100000 calls per day
        # this is slightly over one call per second, but we make multiple calls per loop
        time.sleep(2)
