import json
import requests
import yaml


def retrieve_info(appid):
    return requests.get('http://store.steampowered.com/api/appdetails?appids=' + str(appid)).text


def validate_text_data(info):
    cawlls validate_loaded_data


def validate_loaded_data(info):


f = open manifest
for 
    if yaml exists # previous validation succeeded
        load
        if validate  # validation expected to evolve between runtimes
            continue
        else
            delete yaml
            move text into _validation_failure folder
            warn

    retrieve
    if validate
        save text
        save yaml
        remove _validation_failure folder if it exists
    else:
        save text in _validation_failure folder under unique name
        run comparisons with previous failure texts if they exist
        warn

