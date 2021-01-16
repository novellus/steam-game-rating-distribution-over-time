# local
import common
import plot_sliders

import dateparser
import math
import matplotlib
import multiprocessing
import numpy as np
import os
import scipy.stats
import sys
import time
import traceback
from collections import defaultdict
from matplotlib import pyplot as plt
from sklearn.neighbors import KernelDensity


persistent_references = []  # interactive matplotlib elements require persistent direct references to continue functioning


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

    # init value for score confidence
    init_confidence = 0.9999

    # interactive slider configuration
    init_score_bandwidth = 0.0141
    min_score_bandwidth = 1*10**-10
    max_score_bandwidth = 0.2
    tick_period_score_bandwidth = 0.01
    tick_format_score_bandwidth = '%1.2f'
    fine_format_score_bandwidth = '%1.3f'

    init_date_bandwidth = 45
    min_date_bandwidth = 1*10**-10
    max_date_bandwidth = 200
    tick_period_date_bandwidth = 10
    tick_format_date_bandwidth = '%1.0f'
    fine_format_date_bandwidth = '%1.0f'  # intentionally same

    # overall size and aspect ratioing
    figure_padding = {'left': 0.03, 'right': 0.97, 'top': 0.95, 'bottom': 0.04}  # leave just enough room for labels
    gridspec_padding = 0.08  # all sides
    ratio_xy = 1.618
    ratio_plots_to_configuration = 40  # does not include padding (at matplotlib default setting)
    ratio_2d_to_1d_density = 8  # no padding within this gridspec
    figure_init_width = 30
    figure_init_size = [figure_init_width, figure_init_width / ratio_xy]

    # density distribution discrete sampling paramaters
    num_points_score = 1000
    num_points_date = int(num_points_score * ratio_xy)
    min_plot_score = 0.0
    max_plot_score = 1.0
    min_plot_date = min(dates)
    max_plot_date = max(dates)

    # main plot tick configuration
    tick_format_date = '%b %y'
    # tick_format_score = matplotlib default

    title = 'Density of Wilson score for Steam app ratings over time'
    fig = plt.figure(title, figsize = figure_init_size)
    plt.suptitle(title)

    # divvy up grid space
    gs = matplotlib.gridspec.GridSpec(2, 3,
                                      width_ratios=[1, ratio_plots_to_configuration, 1],
                                      height_ratios=[ratio_plots_to_configuration / ratio_xy, 1],
                                      hspace=gridspec_padding, wspace=gridspec_padding,
                                      **figure_padding)

    gs_score_bandwidth = gs[0]
    gs_color_bar = gs[2]
    gs_date_bandwidth = gs[4]
    gs_confidence = gs[5]

    gs_main_plots = gs[1].subgridspec(2, 2,
                                      width_ratios=[1, ratio_2d_to_1d_density],
                                      height_ratios=[ratio_2d_to_1d_density / ratio_xy, 1],
                                      hspace=0.0, wspace=0.0)  # no space between these graphs, they will share axes. This drives the use of a subgridspec

    gs_score_density = gs_main_plots[0]
    gs_2d_density = gs_main_plots[1]
    gs_date_density = gs_main_plots[3]

    # interactive bandwidth selection sliders for score and date
    # disable dragging, because computation is not fast enough to keep up
    ax_score_bandwidth = plt.subplot(gs_score_bandwidth)
    slider_score_bandwidth = plot_sliders.VertSlider(ax_score_bandwidth, 'Bandwidth,\nWilson Score\n\n', min_score_bandwidth, max_score_bandwidth, valinit=init_score_bandwidth, valfmt=fine_format_score_bandwidth, dragging=False)
    ticks_score_bandwidth = np.arange(min_score_bandwidth, max_score_bandwidth, tick_period_score_bandwidth)
    ax_score_bandwidth.set_yticks(ticks_score_bandwidth)
    ax_score_bandwidth.set_yticklabels([tick_format_score_bandwidth % x for x in ticks_score_bandwidth])
    ax_score_bandwidth.set_ylabel('Unitless, ratio')
    persistent_references.append(slider_score_bandwidth)

    ax_date_bandwidth = plt.subplot(gs_date_bandwidth)
    slider_date_bandwidth = plot_sliders.HorSlider(ax_date_bandwidth, '\n\nBandwidth,\nRelease Date', min_date_bandwidth, max_date_bandwidth, valinit=init_date_bandwidth, valfmt=fine_format_date_bandwidth, dragging=False)
    ticks_date_bandwidth = np.arange(min_date_bandwidth, max_date_bandwidth, tick_period_date_bandwidth)
    ax_date_bandwidth.set_xticks(ticks_date_bandwidth)
    ax_date_bandwidth.set_xticklabels([tick_format_date_bandwidth % x for x in ticks_date_bandwidth])
    ax_date_bandwidth.set_xlabel('Days')
    persistent_references.append(slider_date_bandwidth)

    # interactive confidence input
    ax_confidence = plt.subplot(gs_confidence)
    text_box_confidence = matplotlib.widgets.TextBox(ax_confidence, initial=str(init_confidence), label=None)
    plt.title('Wilson Score\nConfidence')  # TextBox label position is not configurable
    persistent_references.append(text_box_confidence)

    # configure main density plots: 2D density, and 1D density for each axis
    # for now fill in blank datasets to minimize code duplication. We'll compute a real dataset at the end with init configuration values via update_plot defined below

    # score density
    ax_score_density = plt.subplot(gs_score_density)
    blank_score_dataset = [[1]*num_points_score,  # arbitrary density
                           np.linspace(min_plot_score, max_plot_score, num_points_score),
                          ]
    plot_score_density, = plt.plot(*blank_score_dataset)
    plt.xlabel('Density, Wilson score' + ' ' * 25)  # avoid overlapping labels from adjacent date density plot
    plt.ylabel('Lower bound of Wilson score confidence interval')
    plt.ylim(min_plot_score, max_plot_score)
    plt.xlim(left=0.0)  # set minimum plot density value to 0 for visual clarity

    # date density
    ax_date_density = plt.subplot(gs_date_density)
    blank_date_dataset = [np.linspace(min_plot_date, max_plot_date, num_points_date),
                          [1]*num_points_date,  # arbitrary density
                         ]
    plot_date_density, = plt.plot(*blank_date_dataset)
    plt.xlabel('App Release Date')
    plt.ylabel('Density, Release Date' + ' ' * 15)  # avoid overlapping labels from adjacent score density plot
    ax_date_density.xaxis.set_major_formatter(matplotlib.dates.DateFormatter(tick_format_date))
    ax_date_density.xaxis.set_minor_locator(matplotlib.dates.MonthLocator())
    ax_date_density.xaxis.set_major_locator(matplotlib.dates.YearLocator())
    plt.xlim(min_plot_date, max_plot_date)
    plt.ylim(bottom=0.0)  # set minimum plot density value to 0 for visual clarity

    # 2D density
    ax_2d_density = plt.subplot(gs_2d_density)
    blank_image_dataset = [1] * (num_points_score * num_points_date)
    blank_image_dataset = np.array(blank_image_dataset).reshape(num_points_date, num_points_score)
    im_2d_density = plt.imshow(blank_image_dataset, origin='low', aspect='auto', interpolation='catrom', cmap='magma',
                               extent=[min_plot_date, max_plot_date, min_plot_score, max_plot_score])
    # Disable all axis labels. We label the shared axes in the 1D density plots
    ax_2d_density.set_xticklabels([])
    ax_2d_density.set_xticks([])
    ax_2d_density.set_yticklabels([])
    ax_2d_density.set_yticks([])

    # color bar for 2D density plot
    ax_color_bar = plt.subplot(gs_color_bar)
    fig.colorbar(im_2d_density, cax=ax_color_bar)
    plt.title('2D Density Scale')

    # callback for all interactive elements
    # function input ignored due to multiple triggers all resulting in the same recalculation
    # also used to populate initial data with init configuration values
    def update_plot(_=None):
        # pull in new parameters
        text_score_bandwidth = slider_score_bandwidth.val
        text_date_bandwidth = slider_date_bandwidth.val
        text_confidence = text_box_confidence.text
        try:
            score_bandwidth = float(text_score_bandwidth)
            assert score_bandwidth > 0, 'score_bandwidth must be strictly greater than 0 ' + str(score_bandwidth)

            date_bandwidth = float(text_date_bandwidth)
            assert date_bandwidth > 0, 'date_bandwidth must be strictly greater than 0 ' + str(date_bandwidth)

            confidence = float(text_confidence)
            assert 0 < confidence < 1, 'confidence must be strictly between 0 and 1 ' + str(confidence)

        except:
            print(', '.join([str(x) for x in sys.exc_info()[0:2]]))
            return

        # compute wilson scores for given bandwidth
        scores = []
        for i, p in enumerate(ps):
            n = ns[i]
            interval_min, interval_max = Wilson_score_confidence_interval_for_a_Bernoulli_parameter(p, n, confidence=confidence)
            scores.append(interval_min)

        # KernelDensity only takes a 1D bandwidth argument (which it uses for all directions), so we will rescale date values to obtain an independant bandwidth per axis
        # this results in linearly mis-scaled density values, but still correct relative scale
        date_rescale = score_bandwidth / date_bandwidth
        kde_bandwidth = score_bandwidth  # used for all 1D and 2D calcs

        # turn data into np arrays, required by KernelDensity
        np_scores = np.array(scores)
        np_dates = np.array(dates)
        np_rescaled_dates = np_dates * date_rescale
        np_2d_dist = np.array(list(zip(np_rescaled_dates, np_scores)))

        # compute density estimates
        sample_space_scores = np.linspace(min_plot_score, max_plot_score, num_points_score)
        sample_space_dates = np.linspace(min_plot_date, max_plot_date, num_points_date)
        sample_space_rescaled_dates = sample_space_dates * date_rescale
        sample_space_2d_dist = np.array([(date, score) for score in sample_space_scores for date in sample_space_rescaled_dates])  # array order is linearized image pixel order

        log_density_scores = KernelDensity(kernel='epanechnikov', bandwidth=kde_bandwidth).fit(np_scores[:, None]).score_samples(sample_space_scores[:, None])
        density_scores = np.exp(log_density_scores)

        log_density_dates = KernelDensity(kernel='epanechnikov', bandwidth=kde_bandwidth).fit(np_rescaled_dates[:, None]).score_samples(sample_space_rescaled_dates[:, None])
        density_dates = np.exp(log_density_dates) * date_rescale

        log_density_2d_dist = KernelDensity(kernel='epanechnikov', bandwidth=kde_bandwidth).fit(np_2d_dist).score_samples(sample_space_2d_dist)
        density_2d_dist = np.exp(log_density_2d_dist) * date_rescale
        min_density_2d_dist, max_density_2d_dist = min(density_2d_dist), max(density_2d_dist)  # record min/max for normalization parameters
        density_2d_dist = np.array(density_2d_dist).reshape(num_points_score, num_points_date)  # reshape for image plot

        # update plot data
        plot_score_density.set_data(density_scores, sample_space_scores)
        plot_date_density.set_data(sample_space_dates, density_dates)
        im_2d_density.set_data(density_2d_dist)

        # update color normalization parameters for the 2D image
        im_2d_density.set_clim(vmin=min_density_2d_dist, vmax=max_density_2d_dist)

        # rescale only the upper density axis for each 1D density plot
        # date and score ranges are unaffected by plot interaction
        ax_score_density.relim()
        ax_score_density.autoscale(axis='x')
        ax_score_density.set_xlim(left=0.0)  # maintain lower limit at 0

        ax_date_density.relim()
        ax_date_density.autoscale(axis='y')
        ax_date_density.set_ylim(bottom=0.0)  # maintain lower limit at 0

        # force figure to redraw
        fig.canvas.draw()
        fig.canvas.draw()  # intentionally repeated
        fig.canvas.flush_events()

    # register callbacks
    slider_score_bandwidth.on_changed(update_plot)
    slider_date_bandwidth.on_changed(update_plot)
    text_box_confidence.on_submit(update_plot)

    # finally, call the update function to populate the dataset with init configuration values
    update_plot()


def threaded_data_loader(pipe, manifest):
    data = defaultdict(list)
    num_entries = len(manifest)
    
    data_folder = '02_app_info'
    
    for i_app, app in enumerate(manifest):
        appid = app['appid']
        name = app['name']

        # report loading status
        if not i_app % 100:
            pipe.send(('status', float(i_app)/num_entries))

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

            # # dev, remove
            # if len(data['appids']) > 10:
            #     pipe.send(('data', data))
            #     return

    pipe.send(('data', data))


def load_data():
    # loads with many threads

    print('loading manifest')
    f = open('01_steam_manifest.yaml')
    manifest = common.warningless_yaml_load(f.read())
    f.close()

    num_threads = 60
    # num_threads = 1  # dev, remove
    num_manifest_entries = len(manifest['applist']['apps'])
    num_entries_per_thread = int(num_manifest_entries / num_threads)
    previous_index = 0

    # create threads
    pipes = []
    for i_thread in range(num_threads):
        print(f'creating {i_thread+1}/{num_threads} threads to load data', end='\r')
        parent_conn, child_conn = multiprocessing.Pipe()
        pipes.append(parent_conn)


        if i_thread == num_threads - 1:
            manifest_entries = manifest['applist']['apps'][previous_index:]
        else:
            upper_index = previous_index + num_entries_per_thread
            manifest_entries = manifest['applist']['apps'][previous_index: upper_index]
            previous_index = upper_index

        p = multiprocessing.Process(target=threaded_data_loader, args=(child_conn, manifest_entries))
        p.start()
    print()

    # load data
    data = defaultdict(list)
    finished = [False] * num_threads
    progress = [0] * num_threads
    print_progress = False
    while not all(finished):
        for i_pipe, pipe in enumerate(pipes):
            if not finished[i_pipe] and pipe.poll():
                pipe_code, message = pipe.recv()
                print_progress = True

                if pipe_code == 'status':
                    progress[i_pipe] = message
                elif pipe_code == 'data':
                    for key, threaded_data in message.items():
                        data[key].extend(threaded_data)
                    finished[i_pipe] = True
                    progress[i_pipe] = 1.0

        if print_progress:
            # print(f'loading {[f"{100.0*x:.4}%" for x in progress]}%')
            # print(f'loading {[f"{100.0*x:#g}"[:5]+'%' for x in progress]}')  # no way to specify fixed width decimal notation in format-specification-mini-language
            print('loading data', [f'{100.0*x:#g}'[:5]+'%' for x in progress], end='\n\n')  # no way to specify fixed width decimal notation in format-specification-mini-language
            print_progress = False
        time.sleep(0.1)

    return data


def filter_data(data):
    data2 = defaultdict(list)

    for i in range(len(data['appids'])):
        # make assertions on data
        try:
            assert data['total_reviews'][i] > 0
            # TODO select only games
            # TODO select only apps released after June 2013
            # TODO select only apps released after June 2013
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
