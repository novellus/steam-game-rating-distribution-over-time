# local
import common
import plot_sliders

import dateparser
import math
import matplotlib
import os
import scipy.stats
from collections import defaultdict
from matplotlib import pyplot as plt


def Wilson_score_confidence_interval_for_a_Bernoulli_parameter(p, n, confidence = 0.9999):
    # p = ratio of positive ratings in the sample
    # n = total ratings in the sample
    # computes confidence interval for actual value

    # z_α/2 is the (1-α/2) quantile of the standard normal distribution, 1.96 value corresponds to confidence 0.95
    z = scipy.stats.norm.ppf([1 - (1 - confidence) / 2])[0]
    interval_min = (p + (z**2) / (2 * n) - z * math.sqrt((p * (1 - p) + (z**2) / (4 * n)) / n)) / (1 + (z**2) / n)
    interval_max = (p + (z**2) / (2 * n) + z * math.sqrt((p * (1 - p) + (z**2) / (4 * n)) / n)) / (1 + (z**2) / n)

    return interval_min, interval_max


def wilson_score_scatter_plot(dates, ps, ns, confidence = 0.9999):
    print('creating wilson score scatter plot')

    # compute scores
    scores = []
    for i, p in enumerate(ps):
        n = ns[i]
        interval_min, interval_max = Wilson_score_confidence_interval_for_a_Bernoulli_parameter(p, n, confidence=confidence)
        scores.append(interval_min)

    # create plot
    plt.figure()
    plt.plot_date(dates, scores, xdate=True, marker='x', markersize=2, c='blue')
    plt.ylim(0, 1)
    plt.xlabel('App Release Date')
    plt.ylabel('Lower bound of Wilson score confidence interval for Steam apps')
    plt.title(f'All time Steam app ratings\nWilson score at {confidence} confidence')
    ax = plt.gca()
    ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%b %y"))
    ax.xaxis.set_minor_locator(matplotlib.dates.MonthLocator())
    ax.xaxis.set_major_locator(matplotlib.dates.YearLocator())


def wilson_score_density_plot(dates, ps, ns):
    # creates interactive density plot of wilson score vs date
    print('creating wilson score density plot')

    # init value for interactive plot elements
    init_confidence = 0.9999
    init_x_bandwidth = 45
    init_y_bandwidth = 0.0141

    # bounds for interactive sliders
    min_x_bandwidth = 1*10**-10
    max_x_bandwidth = 200
    min_y_bandwidth = 1*10**-10
    max_y_bandwidth = 0.2

    # precompute for repeated density calcs
    date_min = min(dates)
    date_max = max(dates)

    x_y_plot_size_ratio = 1.905  # plot, not figure

    fig = plt.figure()
    plt.suptitle(f'Density of Wilson scores for Steam app ratings over time')

    # updates the plot when any interactive parameter changes
    def update_plot(_=0):
        # pull in new parameters
        x_bandwidth = slider_x_bandwidth.val
        y_bandwidth = slider_y_bandwidth.val
        # TODO implement confidence input field
        confidence = init_confidence

        # compute wilson scores for given bandwidth
        scores = []
        for i, p in enumerate(ps):
            n = ns[i]
            interval_min, interval_max = Wilson_score_confidence_interval_for_a_Bernoulli_parameter(p, n, confidence=confidence)
            scores.append(interval_min)

        # KernelDensity only takes a 1D bandwidth argument (which it uses for all directions), so we will rescale x values to spec bandwidth
        x_rescale = y_bandwidth / x_bandwidth
        y_rescale = 1.0
        bandwidth = y_bandwidth

        density_plot_height_weight = 8
        gs = matplotlib.gridspec.GridSpec(2, 2, width_ratios=[1, density_plot_height_weight], height_ratios=[density_plot_height_weight / x_y_plot_size_ratio, 1])
        gs.update(hspace=0.0, wspace=0.0, left=0.075, right=0.94, top=0.95, bottom=0.125)

        distribution = np.array(list(zip(np.array(dates) * x_rescale, np.array(scores) * y_rescale)))
        resampled_x = np.linspace(date_min * x_rescale, date_max * x_rescale, 1000)
        resampled_y = np.linspace(0 * y_rescale, 1 * y_rescale, 1000)
        resampled_points = np.array([(x,y) for y in resampled_y for x in resampled_x])
        log_density = KernelDensity(kernel='epanechnikov', bandwidth=bandwidth).fit(distribution).score_samples(resampled_points)
        density = np.exp(log_density) * y_rescale * x_rescale
        Z = np.array(density).reshape(len(resampled_x), len(resampled_y))
        plt.subplot(gs[1])
        im = plt.imshow(Z, origin='low', aspect='auto', interpolation='catrom', cmap='magma',
            extent=[resampled_x[0] / x_rescale, resampled_x[-1] / x_rescale, resampled_y[0] / y_rescale, resampled_y[-1] / y_rescale])
        xlim = plt.xlim()
        ylim = plt.ylim()
        ax = plt.gca()
        ax.set_xticklabels([])
        ax.set_xticks([])
        ax.set_yticklabels([])
        ax.set_yticks([])
        fig.colorbar(im, cax=fig.add_axes([0.95, 0.01, 0.015, 0.98]))  # x,y,w,h

        distribution = np.array(scores) * y_rescale
        resampled_points = np.linspace(0 * y_rescale, 1 * y_rescale, 1000)
        log_density = KernelDensity(kernel='epanechnikov', bandwidth=bandwidth).fit(distribution[:, None]).score_samples(resampled_points[:, None])
        density = np.exp(log_density) * y_rescale
        plt.subplot(gs[0])
        plt.plot(density, resampled_points / y_rescale)
        plt.xlabel('Density, Wilson score' + ' ' * 27)
        plt.ylabel('Lower bound of Wilson score confidence interval for individual apps')
        plt.ylim(ylim)  # match main plot
        plt.xlim(left=0.0)  # set minimum plot value to 0 for visual clarity

        distribution = np.array(dates) * x_rescale
        resampled_points = np.linspace(date_min * x_rescale, date_max * x_rescale, 1000)
        log_density = KernelDensity(kernel='epanechnikov', bandwidth=bandwidth).fit(distribution[:, None]).score_samples(resampled_points[:, None])
        density = np.exp(log_density) * x_rescale
        plt.subplot(gs[3])
        plt.plot(resampled_points / x_rescale, density)
        plt.xlabel('App Release Date')
        plt.ylabel('Density, Release Date' + ' ' * 17)
        ax = plt.gca()
        ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%b %y"))
        ax.xaxis.set_minor_locator(matplotlib.dates.MonthLocator())
        ax.xaxis.set_major_locator(matplotlib.dates.YearLocator())
        plt.xlim(xlim)  # match main plot
        plt.ylim(bottom=0.0)  # set minimum plot value to 0 for visual clarity

    # create interactive density sliders to specify bandwidth
    ax_x_bandwidth = fig.add_axes([0.055, 0.045, 0.87, 0.015])
    slider_x_bandwidth = HorSlider(ax_x_bandwidth, 'Bandwidth,\nRelease Date', min_x_bandwidth, max_x_bandwidth, valinit=init_x_bandwidth, valfmt='%1.0f', dragging=False)
    slider_x_bandwidth.on_changed(update_plot)
    x_bandwidth_ticks = np.arange(min_x_bandwidth, max_x_bandwidth, 10)
    ax_x_bandwidth.set_xticks(x_bandwidth_ticks)
    ax_x_bandwidth.set_xticklabels(['{0:.0f}'.format(x) for x in x_bandwidth_ticks])
    ax_x_bandwidth.set_xlabel('Days')

    ax_y_bandwidth = fig.add_axes([0.025, 0.17, 0.015 / x_y_plot_size_ratio, 0.76])
    slider_y_bandwidth = VertSlider(ax_y_bandwidth, 'Bandwidth,\nWilson Score', min_y_bandwidth, max_y_bandwidth, valinit=init_y_bandwidth, valfmt='%1.5f', dragging=False)
    slider_y_bandwidth.on_changed(update_plot)
    y_bandwidth_ticks = np.arange(min_y_bandwidth, max_y_bandwidth, 0.01)
    ax_y_bandwidth.set_yticks(y_bandwidth_ticks)
    ax_y_bandwidth.set_yticklabels(['{0:.2f}'.format(x) for x in y_bandwidth_ticks])
    # ax_y_bandwidth.set_ylabel('') # unitless

    # TODO
    # create confidence input text box
    # not a slider since values of interest are arranged non-linearly near 1, and also takes up less room in the plot window
    # ax_2 = plt.subplot(gs[2])
    # matplotlib.widgets.TextBox(ax_2, label_2, initial='inf')


def load_data():
    print('loading manifest')

    f = open('01_steam_manifest.yaml')
    manifest = common.warningless_yaml_load(f.read())
    f.close()

    print('iterating manifest')

    data_folder = '02_app_info'

    # load and collate data
    data = defaultdict(list)
    num_manifest_entries = len(manifest['applist']['apps'])
    for i_app, app in enumerate(manifest['applist']['apps']):
        appid = app['appid']
        name = app['name']

        # report loading status
        if not i_app % 100:
            print(f'loading {appid}, {i_app} of {num_manifest_entries}, {100.0 * i_app / num_manifest_entries:.4}%')

        # collate loaded data
        yaml_path = os.path.join(data_folder, f'{appid}.yaml')  # stores parsed server responses
        if os.path.exists(yaml_path):
            f = open(yaml_path, encoding='utf-8')
            app_data = common.warningless_yaml_load(f.read())
            f.close()

            data['appids'].append(appid)
            data['names'].append(name)
            data['date_strings'].append(app_data['details']['data']['release_date']['date'])
            data['positive_reviews'].append(app_data['reviews']['query_summary']['total_positive'])
            data['total_reviews'].append(app_data['reviews']['query_summary']['total_reviews'])

    return data


def filter_data(data):
    data2 = defaultdict(list)

    for i in range(len(data['appids'])):
        # make assertions on data
        try:
            assert data['total_reviews'][i] > 0
            # TODO select only games
        except:
            continue
        else:
            # include data
            for key in data:
                data2[key].append(data[key][i])

    return data2


def static_computations(data):
    print('computing data')

    # convert date string to computer representation
    for i, date_string in enumerate(data['date_strings']):
        dt = common.parse_date(date_string)
        data['datetimes'].append(dt)
        data['matplotlib_dates'].append(matplotlib.dates.date2num(dt))

    # compute ratio of positive reviews
    for i, positive_reviews in enumerate(data['positive_reviews']):
        data['p'].append(float(positive_reviews) / data['total_reviews'][i])


if __name__ == '__main__':
    data = load_data()
    data = filter_data(data)
    static_computations(data)

    wilson_score_scatter_plot(data['matplotlib_dates'], data['p'], data['total_reviews'])
    wilson_score_density_plot(data['matplotlib_dates'], data['p'], data['total_reviews'])

    plt.show()
