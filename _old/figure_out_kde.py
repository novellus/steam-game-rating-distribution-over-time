import yaml, pprint, sys, re, time, math, datetime, itertools, statistics, scipy.stats, random, functools
import matplotlib
from matplotlib import pyplot as plt
from matplotlib.collections import LineCollection
import numpy as np
from bisect import bisect_left, bisect_right
from sklearn.neighbors import KernelDensity
from mpl_toolkits.mplot3d import Axes3D
# from sklearn.grid_search import GridSearchCV
from sklearn.model_selection import GridSearchCV
from statsmodels.nonparametric.kernel_density import KDEMultivariate

np.set_printoptions(precision=3, linewidth=2000, suppress=True)
def print_3d_array_as_columns(a):
	print('\n'.join(list(map(lambda line_parts: ''.join([line_part.ljust(30) for line_part in line_parts]), list(zip(*(map(lambda submatrix: submatrix.split('\n '), re.findall('\[\[[\s0-9\.].*?\]\]', np.array2string(a), flags=re.DOTALL)))))))))


def univariate():
	f = lambda x: math.cos(x)
	uniform_x = np.linspace(-math.pi/2, math.pi/2, 100000)
	uniform_y = [f(x) for x in uniform_x]
	distribution = random.choices(uniform_x, weights=uniform_y, k=1000)

	# # Histograms suck!
	# H, xedges = np.histogram(distribution, bins=np.linspace(-math.pi/2, math.pi/2, 100))
	# plt.scatter(xedges[:-1], H, marker='x', s=2, c='red')
	# plt.show()

	# kernel density estimate
	resampled_points = np.linspace(-math.pi/2, math.pi/2, 100)
	log_dens = KernelDensity(kernel='epanechnikov').fit(np.array(distribution)[:, None]).score_samples(resampled_points[:, None])
	plt.plot(resampled_points, np.exp(log_dens))
	plt.show()

def multivariate():
	f = lambda x,y: -x**2 + y**2 + 1
	uniform_x = np.linspace(-1, 1, 1000)
	uniform_y = np.linspace(-1, 1, 1000)
	uniform_x_y_pairs = [(x,y) for y in uniform_y for x in uniform_x]
	uniform_z = [f(x, y) for x, y in uniform_x_y_pairs]
	distribution = np.array(random.choices(uniform_x_y_pairs, weights=uniform_z, k=10000))

	# uniform 3D plot
	fig = plt.figure()
	ax = Axes3D(fig)
	X, Y = np.meshgrid(uniform_x, uniform_y)
	Z = np.array(uniform_z).reshape(len(uniform_x), len(uniform_y))
	ax.set_xlabel('x')
	ax.set_ylabel('y')
	ax.set_zlabel('z')
	ax.plot_surface(X, Y, Z)

	# Histograms suck!
	plt.figure()
	zipped_dist = list(zip(*distribution))
	H, xedges, yedges = np.histogram2d(zipped_dist[0], zipped_dist[1], bins=(np.linspace(-1, 1, 100), np.linspace(-1, 1, 100)))
	plt.imshow(np.transpose(H), origin='low', aspect='auto',
		interpolation='catrom',
		extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
		cmap='magma')
	plt.colorbar()

	# kernel density estimate
	resampled_x = np.linspace(-1, 1, 1000)
	resampled_y = np.linspace(-1, 1, 1000)
	resampled_points = np.array([(x,y) for y in resampled_y for x in resampled_x])
	bandwidth_estimator = GridSearchCV(KernelDensity(kernel='epanechnikov'), {'bandwidth': np.linspace(0.01, 0.15, 20)})
	bandwidth_estimator.fit(distribution)
	print(bandwidth_estimator.best_params_)
	log_density = bandwidth_estimator.best_estimator_.score_samples(resampled_points)
	density = np.exp(log_density)
	# 3d plot
	fig = plt.figure()
	ax = Axes3D(fig)
	X, Y = np.meshgrid(resampled_x, resampled_y)
	Z = np.array(density).reshape(len(resampled_x), len(resampled_y))
	ax.set_xlabel('x')
	ax.set_ylabel('y')
	ax.set_zlabel('z')
	ax.plot_surface(X, Y, Z)
	# imshow plot
	plt.figure()
	plt.imshow(Z, origin='low', aspect='auto',
		interpolation='catrom',
		extent=[resampled_x[0], resampled_x[-1], resampled_y[0], resampled_y[-1]],
		cmap='magma')
	plt.colorbar()

	# iteratively search for an optimum bandwidth via 20-fold cross validation
	resampled_x = np.linspace(-1, 1, 1000)
	resampled_y = np.linspace(-1, 1, 1000)
	resampled_points = np.array([(x,y) for y in resampled_y for x in resampled_x])
	search_min, search_max, search_num = 0.001, 2, 9
	assert search_num >= 4
	while abs(search_min - search_max) > 0.001:
		search_space = np.linspace(search_min, search_max, search_num)
		print('searching', search_space)
		# evaluate search space
		bandwidth_estimator = GridSearchCV(KernelDensity(kernel='epanechnikov'), {'bandwidth': search_space}, cv=20)
		bandwidth_estimator.fit(distribution)
		bandwidth = bandwidth_estimator.best_params_['bandwidth']
		# define next search space
		i = int(np.where(search_space == bandwidth)[0])
		if i == 0:
			search_min, search_max = search_space[:2]
		elif i == len(search_space) - 1:
			search_min, search_max = search_space[-2:]
		else:
			search_min, _, search_max = search_space[i-1:i+2]
		print('recursing on', bandwidth)
	print('settled on', bandwidth_estimator.best_params_)
	print('scored', bandwidth_estimator.best_score_)
	log_density = bandwidth_estimator.best_estimator_.score_samples(resampled_points)
	density = np.exp(log_density)
	Z = np.array(density).reshape(len(resampled_x), len(resampled_y))
	plt.figure()
	plt.imshow(Z, origin='low', aspect='auto',
		interpolation='catrom',
		extent=[resampled_x[0], resampled_x[-1], resampled_y[0], resampled_y[-1]],
		cmap='magma')
	plt.colorbar()

	plt.show()

def multivariate_statsmodels():
	f = lambda x,y: -(x/50)**2 + y**2 + 1
	uniform_x = np.linspace(-50, 50, 1000)
	uniform_y = np.linspace(-1, 1, 1000)
	uniform_x_y_pairs = [(x,y) for y in uniform_y for x in uniform_x]
	uniform_z = [f(x, y) for x, y in uniform_x_y_pairs]
	distribution = np.array(random.choices(uniform_x_y_pairs, weights=uniform_z, k=1000))

	# uniform 3D plot
	fig = plt.figure()
	ax = Axes3D(fig)
	X, Y = np.meshgrid(uniform_x, uniform_y)
	Z = np.array(uniform_z).reshape(len(uniform_x), len(uniform_y))
	ax.set_xlabel('x')
	ax.set_ylabel('y')
	ax.set_zlabel('z')
	ax.plot_surface(X, Y, Z)

	# Histograms suck!
	plt.figure()
	zipped_dist = list(zip(*distribution))
	H, xedges, yedges = np.histogram2d(zipped_dist[0], zipped_dist[1], bins=(np.linspace(-50, 50, 100), np.linspace(-1, 1, 100)))
	plt.imshow(np.transpose(H), origin='low', aspect='auto',
		interpolation='catrom',
		extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
		cmap='magma')
	plt.colorbar()

	# kernel density estimate
	resampled_x = np.linspace(-50, 50, 100)
	resampled_y = np.linspace(-1, 1, 100)
	resampled_points = np.array([(x,y) for y in resampled_y for x in resampled_x])

	# log_density = KernelDensity(kernel='epanechnikov', bandwidth=0.1).fit(distribution).score_samples(resampled_points)
	# density = np.exp(log_density)
	# Z = np.array(density).reshape(len(resampled_x), len(resampled_y))

	# kde = KDEMultivariate(distribution, bw='cv_ls', var_type='uu')
	# print('cv_ls')
	# print(kde.bw)
	# print(kde.loo_likelihood(kde.bw))
	# Z = kde.pdf(resampled_points).reshape(len(resampled_x), len(resampled_y))
	# plt.figure()
	# plt.imshow(Z, origin='low', aspect='auto',
	# 	interpolation='catrom',
	# 	extent=[resampled_x[0], resampled_x[-1], resampled_y[0], resampled_y[-1]],
	# 	cmap='magma')
	# plt.title('cv_ls')
	# plt.colorbar()
	# kde = KDEMultivariate(distribution, bw='cv_ml', var_type='uu')
	# print('cv_ml')
	# print(kde.bw)
	# print(kde.loo_likelihood(kde.bw))
	# Z = kde.pdf(resampled_points).reshape(len(resampled_x), len(resampled_y))
	# plt.figure()
	# plt.imshow(Z, origin='low', aspect='auto',
	# 	interpolation='catrom',
	# 	extent=[resampled_x[0], resampled_x[-1], resampled_y[0], resampled_y[-1]],
	# 	cmap='magma')
	# plt.title('cv_ml')
	# plt.colorbar()
	kde = KDEMultivariate(distribution, bw=[5, 0.1], var_type='uu')
	print([5, 0.1])
	print(kde.bw)
	print(kde.loo_likelihood(kde.bw))
	Z = kde.pdf(resampled_points).reshape(len(resampled_x), len(resampled_y))
	plt.figure()
	plt.imshow(Z, origin='low', aspect='auto',
		interpolation='catrom',
		extent=[resampled_x[0], resampled_x[-1], resampled_y[0], resampled_y[-1]],
		cmap='magma')
	plt.title(str([5, 0.1]))
	plt.colorbar()

	search_min_x, search_max_x, search_num_each_axis = 0.001, 50, 7
	search_min_y, search_max_y 						 = 0.001, 1
	assert search_num_each_axis >= 4
	while abs(search_min_x - search_max_x) > 0.01 or abs(search_min_y - search_max_y) > 0.01:
		search_space_x = np.linspace(search_min_x, search_max_x, search_num_each_axis)
		search_space_y = np.linspace(search_min_y, search_max_y, search_num_each_axis)
		search_space = np.array([(x,y) for y in search_space_y for x in search_space_x])
		print_3d_array_as_columns(search_space.reshape(len(search_space_x), len(search_space_y), 2))
		print('searching x', search_space_x)
		print('searching y', search_space_y)
		# evaluate search space
		likelihoods = []
		for i, bandwidth_estimate in enumerate(search_space):
			print('processing', i+1, 'of', len(search_space), end='\r')
			kde = KDEMultivariate(distribution, bw=bandwidth_estimate, var_type='uu')
			likelihood = kde.loo_likelihood(kde.bw)
			likelihoods.append(likelihood)

		print(np.array(list(zip(search_space, np.array(likelihoods)))).reshape(len(search_space_x), len(search_space_y), 3))
		maximum_likelihood = max(likelihoods)
		i = likelihoods.index(maximum_likelihood)
		bandwidth = search_space[i]
		# define next search space
		i_x = int(np.where(search_space_x == bandwidth[0])[0])
		if i_x == 0:
			search_min_x, search_max_x = search_space_x[:2]
		elif i_x == len(search_space_x) - 1:
			search_min_x, search_max_x = search_space_x[-2:]
		else:
			search_min_x, _, search_max_x = search_space_x[i_x-1:i_x+2]
		i_y = int(np.where(search_space_y == bandwidth[1])[0])
		if i_y == 0:
			search_min_y, search_max_y = search_space_y[:2]
		elif i_y == len(search_space_y) - 1:
			search_min_y, search_max_y = search_space_y[-2:]
		else:
			search_min_y, _, search_max_y = search_space_y[i_y-1:i_y+2]
		print('recursing on', bandwidth)
	print('settled on', bandwidth)
	kde = KDEMultivariate(distribution, bw=bandwidth, var_type='uu')
	likelihood = kde.loo_likelihood(kde.bw)
	print('likelihood', likelihood)
	Z = kde.pdf(resampled_points).reshape(len(resampled_x), len(resampled_y))

	# class fake_scikit_learn_kde():
	# 	def __init__(self, var_type, bw):
	# 		self.var_type = var_type
	# 		self.bw = bw
	# 	def fit(distribution):
	# 		return KDEMultivariate(distribution, var_type=self.var_type)
	# bandwidth_estimator = GridSearchCV(KDEMultivariate(distribution, var_type='uu'), {'bw': [[x, 0.1] for x in np.linspace(0.01, 10, 5)]})
	# bandwidth_estimator.fit(distribution)
	# print(bandwidth_estimator.best_params_)
	# Z = bandwidth_estimator.best_estimator_.pdf(resampled_points)

	plt.figure()
	plt.imshow(Z, origin='low', aspect='auto',
		interpolation='catrom',
		extent=[resampled_x[0], resampled_x[-1], resampled_y[0], resampled_y[-1]],
		cmap='magma')
	plt.colorbar()
	plt.show()

# multivariate_statsmodels()
multivariate()
# univariate()