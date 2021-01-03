import yaml, math, datetime, scipy.stats
import matplotlib
from matplotlib import pyplot as plt
import numpy as np
from sklearn.neighbors import KernelDensity
from matplotlib.widgets import AxesWidget
import six

f = open('steam_app_release_dates_and_reviews', encoding='utf8')
input_data = yaml.load(f.read())
f.close()

dataset_date = input_data['current_date']


def Wilson_score_confidence_interval_for_a_Bernoulli_parameter(p, n, confidence = 0.9999):
	# z_α/2 is the (1-α/2) quantile of the standard normal distribution, 1.96 value corresponds to confidence 0.95
	z = scipy.stats.norm.ppf([1 - (1 - confidence) / 2])[0]
	return ((p + (z**2) / (2 * n) - z * math.sqrt((p * (1 - p) + (z**2) / (4 * n)) / n)) / (1 + (z**2) / n),
			(p + (z**2) / (2 * n) + z * math.sqrt((p * (1 - p) + (z**2) / (4 * n)) / n)) / (1 + (z**2) / n))

num_rejected_full_game_not_real = 0
collated_data = {}  # collate DLC based on base game
for app_id, review_data in input_data['app_data'].items():
	# grab metrics on this game, even if DLC
	p = review_data['computed_data']['all_time_review_rating'] / float(100)
	n = review_data['computed_data']['all_time_number_of_reviews']
	n_pos = p * n

	# resolve if DLC
	real_app_id = app_id
    # fullgame: {appid: '494900', name: Complete Figure Drawing Course HD}
	if 'fullgame' in review_data['api_data']['data']:
		real_app_id = review_data['api_data']['data']['fullgame']['appid']
		if real_app_id not in input_data['app_data']:
			num_rejected_full_game_not_real += 1
			continue
	
	# use release date of base game
	release_date = input_data['app_data'][real_app_id]['computed_data']['release_date']

	# Combine datasets of DLC and base game
	if real_app_id in collated_data:
		collated_n = n + collated_data[real_app_id][2]
		collated_n_pos = n_pos + collated_data[real_app_id][5]
		collated_p = float(collated_n_pos) / collated_n
	else:
		collated_p = p
		collated_n = n
		collated_n_pos = n_pos

	# compute overall score given all thus far known data
	min_confident_score, max_confident_score = Wilson_score_confidence_interval_for_a_Bernoulli_parameter(collated_p, collated_n)
	collated_data[real_app_id] = [release_date, collated_p, collated_n, min_confident_score, max_confident_score, collated_n_pos]

print('num_rejected_full_game_not_real', num_rejected_full_game_not_real)

# flatten data
data = []
for app_id, point in collated_data.items():
	data.append(point)
data.sort()  # release date is first key

plt.figure()
zipped_data = list(zip(*data))
plt.plot_date(zipped_data[0], zipped_data[3], xdate=True, marker='x', markersize=2, c='red')
plt.ylim(0, 1)
plt.xlabel('App Release Date')
plt.ylabel('Lower bound of Wilson score confidence interval for all apps individually')
plt.title('All time Steam app ratings with >5 reviews, as of ' + dataset_date)
ax = plt.gca()
ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%b %y"))
ax.xaxis.set_minor_locator(matplotlib.dates.MonthLocator())
ax.xaxis.set_major_locator(matplotlib.dates.YearLocator())


### adjustable kernel density plot, with bandwidth selectable per axis
# shamelessly copied from https://stackoverflow.com/questions/25934279/add-a-vertical-slider-with-matplotlib/25940123
# modified text position/orientation
class VertSlider(AxesWidget):
    """
    A slider representing a floating point range.

    For the slider to remain responsive you must maintain a
    reference to it.

    The following attributes are defined
      *ax*        : the slider :class:`matplotlib.axes.Axes` instance

      *val*       : the current slider value

      *hline*     : a :class:`matplotlib.lines.Line2D` instance
                     representing the initial value of the slider

      *poly*      : A :class:`matplotlib.patches.Polygon` instance
                     which is the slider knob

      *valfmt*    : the format string for formatting the slider text

      *label*     : a :class:`matplotlib.text.Text` instance
                     for the slider label

      *closedmin* : whether the slider is closed on the minimum

      *closedmax* : whether the slider is closed on the maximum

      *slidermin* : another slider - if not *None*, this slider must be
                     greater than *slidermin*

      *slidermax* : another slider - if not *None*, this slider must be
                     less than *slidermax*

      *dragging*  : allow for mouse dragging on slider

    Call :meth:`on_changed` to connect to the slider event
    """
    def __init__(self, ax, label, valmin, valmax, valinit=0.5, valfmt='%1.2f',
                 closedmin=True, closedmax=True, slidermin=None,
                 slidermax=None, dragging=True, 
                 label_pos=(0.5, -0.005), valtext_pos=(0.5, 1.005), **kwargs):
        """
        Create a slider from *valmin* to *valmax* in axes *ax*.

        Additional kwargs are passed on to ``self.poly`` which is the
        :class:`matplotlib.patches.Rectangle` which draws the slider
        knob.  See the :class:`matplotlib.patches.Rectangle` documentation
        valid property names (e.g., *facecolor*, *edgecolor*, *alpha*, ...).

        Parameters
        ----------
        ax : Axes
            The Axes to put the slider in

        label : str
            Slider label

        valmin : float
            The minimum value of the slider

        valmax : float
            The maximum value of the slider

        valinit : float
            The slider initial position

        valfmt : str
            Used to format the slider value, fprint format string

        closedmin : bool
            Indicate whether the slider interval is closed on the bottom

        closedmax : bool
            Indicate whether the slider interval is closed on the top

        slidermin : Slider or None
            Do not allow the current slider to have a value less than
            `slidermin`

        slidermax : Slider or None
            Do not allow the current slider to have a value greater than
            `slidermax`


        dragging : bool
            if the slider can be dragged by the mouse

        """
        AxesWidget.__init__(self, ax)

        self.valmin = valmin
        self.valmax = valmax
        self.val = valinit
        self.valinit = valinit
        self.poly = ax.axhspan(valmin, valinit, 0, 1, **kwargs)

        self.hline = ax.axhline(valinit, 0, 1, color='r', lw=1)

        self.valfmt = valfmt
        ax.set_xticks([])
        ax.set_ylim((valmin, valmax))
        ax.set_yticks([])
        ax.set_navigate(False)

        self.connect_event('button_press_event', self._update)
        self.connect_event('button_release_event', self._update)
        if dragging:
            self.connect_event('motion_notify_event', self._update)
        self.label = ax.text(*label_pos, label, transform=ax.transAxes,
                             verticalalignment='top',
                             horizontalalignment='center', rotation=90)

        self.valtext = ax.text(*valtext_pos, valfmt % valinit,
                               transform=ax.transAxes,
                               verticalalignment='bottom',
                               horizontalalignment='center', rotation=90)

        self.cnt = 0
        self.observers = {}

        self.closedmin = closedmin
        self.closedmax = closedmax
        self.slidermin = slidermin
        self.slidermax = slidermax
        self.drag_active = False

    def _update(self, event):
        """update the slider position"""
        if self.ignore(event):
            return

        if event.button != 1:
            return

        if event.name == 'button_press_event' and event.inaxes == self.ax:
            self.drag_active = True
            event.canvas.grab_mouse(self.ax)

        if not self.drag_active:
            return

        elif ((event.name == 'button_release_event') or
              (event.name == 'button_press_event' and
               event.inaxes != self.ax)):
            self.drag_active = False
            event.canvas.release_mouse(self.ax)
            return

        val = event.ydata
        if val <= self.valmin:
            if not self.closedmin:
                return
            val = self.valmin
        elif val >= self.valmax:
            if not self.closedmax:
                return
            val = self.valmax

        if self.slidermin is not None and val <= self.slidermin.val:
            if not self.closedmin:
                return
            val = self.slidermin.val

        if self.slidermax is not None and val >= self.slidermax.val:
            if not self.closedmax:
                return
            val = self.slidermax.val

        self.set_val(val)

    def set_val(self, val):
        xy = self.poly.xy
        xy[1] = 0, val
        xy[2] = 1, val
        self.poly.xy = xy
        self.valtext.set_text(self.valfmt % val)
        if self.drawon:
            self.ax.figure.canvas.draw_idle()
        self.val = val
        if not self.eventson:
            return
        for cid, func in six.iteritems(self.observers):
            func(val)

    def on_changed(self, func):
        """
        When the slider value is changed, call *func* with the new
        slider position

        A connection id is returned which can be used to disconnect
        """
        cid = self.cnt
        self.observers[cid] = func
        self.cnt += 1
        return cid

    def disconnect(self, cid):
        """remove the observer with connection id *cid*"""
        try:
            del self.observers[cid]
        except KeyError:
            pass

    def reset(self):
        """reset the slider to the initial value if needed"""
        if (self.val != self.valinit):
            self.set_val(self.valinit)

# modified for text spacing
class HorSlider(AxesWidget):
    """
    A slider representing a floating point range.

    Create a slider from *valmin* to *valmax* in axes *ax*. For the slider to
    remain responsive you must maintain a reference to it. Call
    :meth:`on_changed` to connect to the slider event.

    Attributes
    ----------
    val : float
        Slider value.
    """
    def __init__(self, ax, label, valmin, valmax, valinit=0.5, valfmt='%1.2f',
                 closedmin=True, closedmax=True, slidermin=None,
                 slidermax=None, dragging=True, valstep=None,
                 label_pos=(-0.005, 0.5), valtext_pos=(1.005, 0.5), **kwargs):
        """
        Parameters
        ----------
        ax : Axes
            The Axes to put the slider in.

        label : str
            Slider label.

        valmin : float
            The minimum value of the slider.

        valmax : float
            The maximum value of the slider.

        valinit : float, optional, default: 0.5
            The slider initial position.

        valfmt : str, optional, default: "%1.2f"
            Used to format the slider value, fprint format string.

        closedmin : bool, optional, default: True
            Indicate whether the slider interval is closed on the bottom.

        closedmax : bool, optional, default: True
            Indicate whether the slider interval is closed on the top.

        slidermin : Slider, optional, default: None
            Do not allow the current slider to have a value less than
            the value of the Slider `slidermin`.

        slidermax : Slider, optional, default: None
            Do not allow the current slider to have a value greater than
            the value of the Slider `slidermax`.

        dragging : bool, optional, default: True
            If True the slider can be dragged by the mouse.

        valstep : float, optional, default: None
            If given, the slider will snap to multiples of `valstep`.

        Notes
        -----
        Additional kwargs are passed on to ``self.poly`` which is the
        :class:`~matplotlib.patches.Rectangle` that draws the slider
        knob.  See the :class:`~matplotlib.patches.Rectangle` documentation for
        valid property names (e.g., `facecolor`, `edgecolor`, `alpha`).
        """
        AxesWidget.__init__(self, ax)

        if slidermin is not None and not hasattr(slidermin, 'val'):
            raise ValueError("Argument slidermin ({}) has no 'val'"
                             .format(type(slidermin)))
        if slidermax is not None and not hasattr(slidermax, 'val'):
            raise ValueError("Argument slidermax ({}) has no 'val'"
                             .format(type(slidermax)))
        self.closedmin = closedmin
        self.closedmax = closedmax
        self.slidermin = slidermin
        self.slidermax = slidermax
        self.drag_active = False
        self.valmin = valmin
        self.valmax = valmax
        self.valstep = valstep
        valinit = self._value_in_bounds(valinit)
        if valinit is None:
            valinit = valmin
        self.val = valinit
        self.valinit = valinit
        self.poly = ax.axvspan(valmin, valinit, 0, 1, **kwargs)
        self.vline = ax.axvline(valinit, 0, 1, color='r', lw=1)

        self.valfmt = valfmt
        ax.set_yticks([])
        ax.set_xlim((valmin, valmax))
        ax.set_xticks([])
        ax.set_navigate(False)

        self.connect_event('button_press_event', self._update)
        self.connect_event('button_release_event', self._update)
        if dragging:
            self.connect_event('motion_notify_event', self._update)
        self.label = ax.text(*label_pos, label, transform=ax.transAxes,
                             verticalalignment='center',
                             horizontalalignment='right')

        self.valtext = ax.text(*valtext_pos, valfmt % valinit,
                               transform=ax.transAxes,
                               verticalalignment='center',
                               horizontalalignment='left')

        self.cnt = 0
        self.observers = {}

        self.set_val(valinit)

    def _value_in_bounds(self, val):
        """ Makes sure self.val is with given bounds."""
        if self.valstep:
            val = np.round((val - self.valmin)/self.valstep)*self.valstep
            val += self.valmin

        if val <= self.valmin:
            if not self.closedmin:
                return
            val = self.valmin
        elif val >= self.valmax:
            if not self.closedmax:
                return
            val = self.valmax

        if self.slidermin is not None and val <= self.slidermin.val:
            if not self.closedmin:
                return
            val = self.slidermin.val

        if self.slidermax is not None and val >= self.slidermax.val:
            if not self.closedmax:
                return
            val = self.slidermax.val
        return val

    def _update(self, event):
        """update the slider position"""
        if self.ignore(event):
            return

        if event.button != 1:
            return

        if event.name == 'button_press_event' and event.inaxes == self.ax:
            self.drag_active = True
            event.canvas.grab_mouse(self.ax)

        if not self.drag_active:
            return

        elif ((event.name == 'button_release_event') or
              (event.name == 'button_press_event' and
               event.inaxes != self.ax)):
            self.drag_active = False
            event.canvas.release_mouse(self.ax)
            return
        val = self._value_in_bounds(event.xdata)
        if (val is not None) and (val != self.val):
            self.set_val(val)

    def set_val(self, val):
        """
        Set slider value to *val*

        Parameters
        ----------
        val : float
        """
        xy = self.poly.xy
        xy[2] = val, 1
        xy[3] = val, 0
        self.poly.xy = xy
        self.valtext.set_text(self.valfmt % val)
        if self.drawon:
            self.ax.figure.canvas.draw_idle()
        self.val = val
        if not self.eventson:
            return
        for cid, func in six.iteritems(self.observers):
            func(val)

    def on_changed(self, func):
        """
        When the slider value is changed call *func* with the new
        slider value

        Parameters
        ----------
        func : callable
            Function to call when slider is changed.
            The function must accept a single float as its arguments.

        Returns
        -------
        cid : int
            Connection id (which can be used to disconnect *func*)
        """
        cid = self.cnt
        self.observers[cid] = func
        self.cnt += 1
        return cid

    def disconnect(self, cid):
        """
        Remove the observer with connection id *cid*

        Parameters
        ----------
        cid : int
            Connection id of the observer to be removed
        """
        try:
            del self.observers[cid]
        except KeyError:
            pass

    def reset(self):
        """Reset the slider to the initial value"""
        if (self.val != self.valinit):
            self.set_val(self.valinit)

# release_date, collated_p, collated_n, min_confident_score, max_confident_score, collated_n_pos
filtered_dates, _, _, filtered_scores, _, _ = list(zip(*list(filter(lambda point: point[0] >= matplotlib.dates.date2num(datetime.datetime.strptime('Jun 2013', '%b %Y')), data))))
date_min = min(filtered_dates)
date_max = max(filtered_dates)
x_y_plot_size_ratio = 1.905
fig = plt.figure()
plt.suptitle('Density, lower bound of 99.99% confidence interval for individual Steam App Ratings over Time. Apps with >5 reviews, as of ' + dataset_date)
def update_plot(_=0):
	x_bandwidth = slider_x_bandwidth.val
	y_bandwidth = slider_y_bandwidth.val
	# rescale x to spec bandwidth since KernelDensity only takes a 1D bandwidth argument
	x_rescale = y_bandwidth / x_bandwidth
	y_rescale = 1.0
	bandwidth = y_bandwidth

	density_plot_height_weight = 8
	gs = matplotlib.gridspec.GridSpec(2, 2, width_ratios=[1, density_plot_height_weight], height_ratios=[density_plot_height_weight / x_y_plot_size_ratio, 1])
	gs.update(hspace=0.0, wspace=0.0, left=0.075, right=0.94, top=0.95, bottom=0.125)

	distribution = np.array(list(zip(np.array(filtered_dates) * x_rescale, np.array(filtered_scores) * y_rescale)))
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

	distribution = np.array(filtered_scores) * y_rescale
	resampled_points = np.linspace(0 * y_rescale, 1 * y_rescale, 1000)
	log_density = KernelDensity(kernel='epanechnikov', bandwidth=bandwidth).fit(distribution[:, None]).score_samples(resampled_points[:, None])
	density = np.exp(log_density) * y_rescale
	plt.subplot(gs[0])
	plt.plot(density, resampled_points / y_rescale)
	plt.xlabel('Density, Wilson score' + ' ' * 27)
	plt.ylabel('Lower bound of Wilson score confidence interval for individual apps')
	plt.ylim(ylim)

	distribution = np.array(filtered_dates) * x_rescale
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
	plt.xlim(xlim)

min_x_bandwidth = 1*10**-10
max_x_bandwidth = 200
ax_x_bandwidth = fig.add_axes([0.055, 0.045, 0.87, 0.015])
slider_x_bandwidth = HorSlider(ax_x_bandwidth, 'Bandwidth,\nRelease Date', min_x_bandwidth, max_x_bandwidth, valinit=45, valfmt='%1.0f', dragging=False)
slider_x_bandwidth.on_changed(update_plot)
x_bandwidth_ticks = np.arange(min_x_bandwidth, max_x_bandwidth, 10)
ax_x_bandwidth.set_xticks(x_bandwidth_ticks)
ax_x_bandwidth.set_xticklabels(['{0:.0f}'.format(x) for x in x_bandwidth_ticks])
ax_x_bandwidth.set_xlabel('Days')

min_y_bandwidth = 1*10**-10
max_y_bandwidth = 0.2
ax_y_bandwidth = fig.add_axes([0.025, 0.17, 0.015 / x_y_plot_size_ratio, 0.76])
slider_y_bandwidth = VertSlider(ax_y_bandwidth, 'Bandwidth,\nWilson Score', min_y_bandwidth, max_y_bandwidth, valinit=0.0141, valfmt='%1.5f', dragging=False)
slider_y_bandwidth.on_changed(update_plot)
y_bandwidth_ticks = np.arange(min_y_bandwidth, max_y_bandwidth, 0.01)
ax_y_bandwidth.set_yticks(y_bandwidth_ticks)
ax_y_bandwidth.set_yticklabels(['{0:.2f}'.format(x) for x in y_bandwidth_ticks])
# ax_x_bandwidth.set_ylabel('') # unitless

update_plot()




plt.show()