import json, pprint, sys, re, time, math, datetime, itertools, statistics, requests, scipy.stats
import matplotlib
from matplotlib import pyplot as plt
from matplotlib.collections import LineCollection
import numpy as np

f = open('_old_steam_app_release_dates_and_reviews', encoding='utf8')
all_review_data = json.loads(f.read())
f.close()


def Wilson_score_confidence_interval_for_a_Bernoulli_parameter(p, n, confidence = 0.9999):
	# z_α/2 is the (1-α/2) quantile of the standard normal distribution, 1.96 value corresponds to confidence 0.95
	z = scipy.stats.norm.ppf([1 - (1 - confidence) / 2])[0]
	return ((p + (z**2) / (2 * n) - z * math.sqrt((p * (1 - p) + (z**2) / (4 * n)) / n)) / (1 + (z**2) / n),
			(p + (z**2) / (2 * n) + z * math.sqrt((p * (1 - p) + (z**2) / (4 * n)) / n)) / (1 + (z**2) / n))

exlude_dates_outside = [0, matplotlib.dates.date2num(datetime.datetime.strptime('Jun 23 18', '%b %d %y'))]
num_excluded_invalid_date = 0
num_excluded_dates_outside_specified_range = 0
data = []
for app_id, review_data in all_review_data.items():
	p = review_data['all_time_review_rating'] / float(100)
	n = review_data['all_time_number_of_reviews']

	if not review_data['release_date']:
		num_excluded_invalid_date += 1
		continue

	# steam api has several different date formats
	try:
		release_date = matplotlib.dates.date2num(datetime.datetime.strptime(review_data['release_date'], '%b %d, %Y'))
	except:
		try:
			release_date = matplotlib.dates.date2num(datetime.datetime.strptime(review_data['release_date'], '%b %Y'))
		except:
			release_date = matplotlib.dates.date2num(datetime.datetime.strptime(review_data['release_date'], '%d %b, %Y'))

	if release_date < exlude_dates_outside[0] or release_date > exlude_dates_outside[1]:
		num_excluded_dates_outside_specified_range += 1
		continue

	min_confident_score, max_confident_score = Wilson_score_confidence_interval_for_a_Bernoulli_parameter(p, n)

	data.append([release_date, p, n, min_confident_score, max_confident_score, app_id])

print('num_excluded_invalid_date', num_excluded_invalid_date)
print('num_excluded_dates_outside_specified_range', num_excluded_dates_outside_specified_range)

data = list(filter(lambda point: point[0] > matplotlib.dates.date2num(datetime.datetime.strptime('Aug 1 16', '%b %d %y')), data))
# data = list(filter(lambda point: point[0] < matplotlib.dates.date2num(datetime.datetime.strptime('Sep 1 16', '%b %d %y')), data))
data = list(filter(lambda point: float('inf') >= point[3] > 0.8, data))
data.sort(key=lambda point: point[3], reverse=True)

zipped_data_bin = list(zip(*data))
representative_date = zipped_data_bin[0][0]
n_data_bin = sum(zipped_data_bin[2])
p_data_bin = sum(map(lambda point: point[1] * point[2], data)) / float(n_data_bin)
binned_min_confident_score, binned_max_confident_score = Wilson_score_confidence_interval_for_a_Bernoulli_parameter(p_data_bin, n_data_bin)

print(len(data), 'Entries')
for point in data:
	app_info_json = json.loads(requests.get('http://store.steampowered.com/api/appdetails?appids=' + str(point[5])).text)
	try:
		name = app_info_json[str(point[5])]['data']['name']
	except:
		continue
	print('{:0.2f}'.format(point[3]), 'http://store.steampowered.com/app/' + str(point[5]), name)
	# if point[5] == '519560':
	# 	for confidence in [0.9, 0.95, 0.99, 0.999, 0.9999]:
	# 		min_confident_score, max_confident_score = Wilson_score_confidence_interval_for_a_Bernoulli_parameter(point[1], point[2], confidence)
	# 		print(min_confident_score)
	# 	sys.exit()