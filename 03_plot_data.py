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
import time
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
    slider_x_bandwidth = plot_sliders.HorSlider(ax_x_bandwidth, 'Bandwidth,\nRelease Date', min_x_bandwidth, max_x_bandwidth, valinit=init_x_bandwidth, valfmt='%1.0f', dragging=False)
    slider_x_bandwidth.on_changed(update_plot)
    x_bandwidth_ticks = np.arange(min_x_bandwidth, max_x_bandwidth, 10)
    ax_x_bandwidth.set_xticks(x_bandwidth_ticks)
    ax_x_bandwidth.set_xticklabels(['{0:.0f}'.format(x) for x in x_bandwidth_ticks])
    ax_x_bandwidth.set_xlabel('Days')

    ax_y_bandwidth = fig.add_axes([0.025, 0.17, 0.015 / x_y_plot_size_ratio, 0.76])
    slider_y_bandwidth = plot_sliders.VertSlider(ax_y_bandwidth, 'Bandwidth,\nWilson Score', min_y_bandwidth, max_y_bandwidth, valinit=init_y_bandwidth, valfmt='%1.5f', dragging=False)
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

    update_plot()


def wilson_score_density_plot2(dates, ps, ns):
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
    plot_score_density = plt.plot(*blank_score_dataset)
    plt.xlabel('Density, Wilson score' + ' ' * 25)  # avoid overlapping labels from adjacent date density plot
    plt.ylabel('Lower bound of Wilson score confidence interval')
    plt.ylim(min_plot_score, max_plot_score)
    plt.xlim(left=0.0)  # set minimum plot density value to 0 for visual clarity

    # date density
    ax_date_density = plt.subplot(gs_date_density)
    blank_date_dataset = [np.linspace(min_plot_date, max_plot_date, num_points_date),
                          [1]*num_points_date,  # arbitrary density
                         ]
    plot_date_density = plt.plot(*blank_date_dataset)
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
    ax = plt.gca()
    ax.set_xticklabels([])
    ax.set_xticks([])
    ax.set_yticklabels([])
    ax.set_yticks([])

    # color bar for 2D density plot
    ax_color_bar = plt.subplot(gs_color_bar)
    fig.colorbar(im_2d_density, cax=ax_color_bar)
    plt.title('2D Density Scale')

    # callback for all interactive elements
    # also used to populate initial data with init configuration values
    def update_plot(_=None):
        # TODO
        pass

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

            # dev, remove
            if len(data['appids']) > 10:
                pipe.send(('data', data))
                return

    pipe.send(('data', data))


def load_data():
    # loads with many threads

    print('loading manifest')
    f = open('01_steam_manifest.yaml')
    manifest = common.warningless_yaml_load(f.read())
    f.close()

    # num_threads = 60
    num_threads = 1  # dev, remove
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

    # wilson_score_scatter_plot(data['matplotlib_dates'], data['p'], data['total_reviews'])
    # wilson_score_density_plot(data['matplotlib_dates'], data['p'], data['total_reviews'])
    # wilson_score_density_plot2(data['matplotlib_dates'], data['p'], data['total_reviews'])
    # persistent_references.append( wilson_score_density_plot3(data['matplotlib_dates'], data['p'], data['total_reviews']) )
    wilson_score_density_plot2(data['matplotlib_dates'], data['p'], data['total_reviews'])

    plt.show()
