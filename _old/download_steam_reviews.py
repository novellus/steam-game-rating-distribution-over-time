import requests, json, pprint, sys, re, time, datetime, os, yaml
import matplotlib.dates

# initialize from previous state when possible
if os.path.exists('steam_app_release_dates_and_reviews'):
	f=open('steam_app_release_dates_and_reviews', 'r', encoding='utf8')
	output_data = yaml.load(f.read())
	f.close()
else:
	output_data = {'app_data': {},
				   'appids_excluded_no_release_date':set(),
				   'appids_excluded_release_date_in_future':set(),
				   'appids_connection_failed_api':set(),
				   'appids_connection_failed_html':set(),
				   'appids_failed_to_parse_all_time_reviews':set(),
				   'appids_api_reported_failure':set()}

current_date = datetime.datetime.now()
output_data['current_date'] = current_date.isoformat()

all_apps_json = json.loads(requests.get('http://api.steampowered.com/ISteamApps/GetAppList/v0002/?key=STEAMKEY&format=json').text)
all_appids = [app['appid'] for app in all_apps_json['applist']['apps']]

just_printed_message_i_want_to_save = False
for i, appid in enumerate(all_appids):
	# skip if we already parsed this one in a previous session
	if appid in output_data['app_data']:
		continue

	# progress messages
	if just_printed_message_i_want_to_save:
		print('')
		just_printed_message_i_want_to_save = False
	else:
		print('\r', end='')  # replace last print
	print('processing appid ' + str(appid) + ' of ' + str(all_appids[-1]), end='')

	# request api data
	connection_attempts = 0
	max_connection_attempts = 3
	while 0 <= connection_attempts < max_connection_attempts:
		try:
			app_info_text = requests.get('http://store.steampowered.com/api/appdetails?appids=' + str(appid)).text
			# wait and try again if we saturated the access rate
			# check for explicit access denied
			if '<TITLE>Access Denied</TITLE>' in app_info_text or not app_info_text:
				print('\nsaturated api usage rate, waiting 5 minutes', end='')
				just_printed_message_i_want_to_save = True
				time.sleep(60*5)
				raise ValueError('saturated api')
			app_info_json = json.loads(app_info_text)
			# check for implicit saturation: null responses
			if app_info_json == None:
				print('\nsaturated api usage rate, waiting 5 minutes', end='')
				just_printed_message_i_want_to_save = True
				time.sleep(60*5)
				raise ValueError('saturated api')
			success = app_info_json[str(appid)]['success']
			connection_attempts = -1
		except:
			connection_attempts +=1
			time.sleep(0.1)
	if connection_attempts == max_connection_attempts:
		print('\nconnection failed, failed to get api data for appid ' + str(appid), end='')
		output_data['appids_connection_failed_api'].add(appid)
		just_printed_message_i_want_to_save = True
		continue
	if not success:
		print('\napi reported failure for appid ' + str(appid), end='')
		output_data['appids_api_reported_failure'].add(appid)
		just_printed_message_i_want_to_save = True
		continue

	# Parse Date - steam api has several different date formats
	release_date_string = app_info_json[str(appid)]['data']['release_date']['date']
	if not release_date_string:
		output_data['appids_excluded_no_release_date'].add(appid)
		continue
	if 'coming_soon' in app_info_json[str(appid)]['data']['release_date'] and app_info_json[str(appid)]['data']['release_date']['coming_soon'] == True:
		output_data['appids_excluded_release_date_in_future'].add(appid)
		continue
	try:
		release_date = matplotlib.dates.date2num(datetime.datetime.strptime(release_date_string, '%b %d, %Y'))
	except:
		try:
			release_date = matplotlib.dates.date2num(datetime.datetime.strptime(release_date_string, '%b %Y'))
		except:
			try:
				release_date = matplotlib.dates.date2num(datetime.datetime.strptime(release_date_string, '%d %b, %Y'))
			except:
				try:
					release_date = matplotlib.dates.date2num(datetime.datetime.strptime(release_date_string, '%Y %b %d'))
				except:
					release_date = matplotlib.dates.date2num(datetime.datetime.strptime(release_date_string, '%Y年%b月%d日'))
	if release_date > matplotlib.dates.date2num(current_date):
		output_data['appids_excluded_release_date_in_future'].add(appid)
		continue

	# request html data
	connection_attempts = 0
	max_connection_attempts = 3
	while 0 <= connection_attempts < max_connection_attempts:
		try:
			app_html = requests.get('http://store.steampowered.com/app/' + str(appid)).text
			connection_attempts = -1
		except:
			connection_attempts +=1
			time.sleep(0.1)
	if connection_attempts == max_connection_attempts:
		print('\nconnection failed, failed to get html data for appid ' + str(appid), end='')
		output_data['appids_connection_failed_html'].add(appid)
		just_printed_message_i_want_to_save = True
		continue


	# parse ratings and number_of_reviews from the html
	all_time_reviews_search = re.search('^.*user_reviews_summary_row.*data-tooltip-text="([0-9]+)\% of the ([0-9,]+) user reviews.*$\n.*All Reviews.*$', app_html, flags=re.MULTILINE)
	if not all_time_reviews_search:
		print('\nfailed to parse all time reviews data for appid ' + str(appid), end='')
		output_data['appids_failed_to_parse_all_time_reviews'].add(appid)
		just_printed_message_i_want_to_save = True
		continue
	all_time_review_rating = int(all_time_reviews_search.group(1))
	all_time_number_of_reviews = int(re.sub(',', '', all_time_reviews_search.group(2)))

	# save output data
	output_data['app_data'][appid] = {
		'api_data':app_info_json[str(appid)],
		'html_data': app_html,
		'computed_data':{'release_date': release_date,
						 'all_time_review_rating': all_time_review_rating,
						 'all_time_number_of_reviews': all_time_number_of_reviews}
	}

	if appid in output_data['appids_excluded_no_release_date']:
		output_data['appids_excluded_no_release_date'].remove(appid)
	if appid in output_data['appids_excluded_release_date_in_future']:
		output_data['appids_excluded_release_date_in_future'].remove(appid)
	if appid in output_data['appids_connection_failed_api']:
		output_data['appids_connection_failed_api'].remove(appid)
	if appid in output_data['appids_connection_failed_html']:
		output_data['appids_connection_failed_html'].remove(appid)
	if appid in output_data['appids_failed_to_parse_all_time_reviews']:
		output_data['appids_failed_to_parse_all_time_reviews'].remove(appid)
	if appid in output_data['appids_api_reported_failure']:
		output_data['appids_api_reported_failure'].remove(appid)

	# save to file periodically
	if not i%1000:
		f=open('steam_app_release_dates_and_reviews', 'w', encoding='utf8')
		f.write(yaml.dump(output_data))
		f.close()
		print('\njust saved', end='')
		just_printed_message_i_want_to_save = True

# report stats
print('appids_excluded_no_release_date'.ljust(40), len(output_data['appids_excluded_no_release_date']))
print('appids_excluded_release_date_in_future'.ljust(40), len(output_data['appids_excluded_release_date_in_future']))
print('appids_connection_failed_api'.ljust(40), len(output_data['appids_connection_failed_api']))
print('appids_connection_failed_html'.ljust(40), len(output_data['appids_connection_failed_html']))
print('appids_failed_to_parse_all_time_reviews'.ljust(40), len(output_data['appids_failed_to_parse_all_time_reviews']))
print('appids_api_reported_failure'.ljust(40), len(output_data['appids_api_reported_failure']))

# save final dataset
f=open('steam_app_release_dates_and_reviews', 'w', encoding='utf8')
f.write(yaml.dump(output_data))
f.close()

# f=open('tmp', 'w', encoding='utf8')
# f.write(pprint.pformat(output_data, width=1000))
# f.close()
