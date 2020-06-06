from collections import defaultdict

import numpy as np

from ._common_plot import CommonPlot


def line(x=None, y=None, data=None, aggfunc=None, split=None, row=None, col=None, 
         x_order=None, y_order=None, split_order=None, row_order=None, col_order=None,
         orientation='v', sort='lex_asc', wrap=None, figsize=None, title=None, sharex=True, 
         sharey=True, xlabel=None, ylabel=None, xlim=None, ylim=None, xscale='linear', 
         yscale='linear', cmap=None, x_textwrap=10, y_textwrap=None):

        self = CommonPlot(x, y, data, aggfunc, split, row, col, 
                          x_order, y_order, split_order, row_order, col_order,
                          orientation, sort, wrap, figsize, title, sharex, 
                          sharey, xlabel, ylabel, xlim, ylim, xscale, yscale, cmap,
                          x_textwrap, y_textwrap)

        for ax, info in self.final_data.items():
            for x, y, label, col_name, row_label, col_label in info:
                x_plot, y_plot = self.get_x_y_plot(x, y)
                ax.plot(x_plot, y_plot, label=label)
            self.add_ticklabels(x, y, ax)
        
        self.add_legend()
        if x.dtype == 'O' or y.dtype == 'O':
            self.update_fig_size(len(x), 1)
        return self.clean_up()
        
def scatter(x=None, y=None, data=None, aggfunc=None, split=None, row=None, col=None, 
            x_order=None, y_order=None, split_order=None, row_order=None, col_order=None,
            orientation='v', sort='lex_asc', wrap=None, figsize=None, title=None, sharex=True, 
            sharey=True, xlabel=None, ylabel=None, xlim=None, ylim=None, xscale='linear', 
            yscale='linear', cmap=None, x_textwrap=10, y_textwrap=None, regression=False):

        self = CommonPlot(x, y, data, aggfunc, split, row, col, 
                          x_order, y_order, split_order, row_order, col_order,
                          orientation, sort, wrap, figsize, title, sharex, 
                          sharey, xlabel, ylabel, xlim, ylim, xscale, yscale, cmap,
                          x_textwrap, y_textwrap)

        for ax, info in self.final_data.items():
            for x, y, label, col_name, row_label, col_label in info:
                x_plot, y_plot = self.get_x_y_plot(x, y)
                ax.scatter(x_plot, y_plot, label=label, alpha=.7)
                if regression:
                    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
                    x_line = np.array([x.min(), x.max()])
                    y_line = x_line * slope + intercept
                    ax.plot(x_line, y_line)
            self.add_ticklabels(x, y, ax)

        self.add_legend()
        if x.dtype == 'O' or y.dtype == 'O':
            self.update_fig_size(len(x), 1)
        return self.clean_up()


def bar(x=None, y=None, data=None, aggfunc=None, split=None, row=None, col=None, 
        x_order=None, y_order=None, split_order=None, row_order=None, col_order=None,
        orientation='v', stacked=False, sort='lex_asc', wrap=None, figsize=None, 
        title=None, sharex=True, sharey=True, xlabel=None, ylabel=None, xlim=None, 
        ylim=None, xscale='linear', yscale='linear', cmap=None, size=.92, 
        x_textwrap=10, y_textwrap=None, bar_kwargs=None):

        self = CommonPlot(x, y, data, aggfunc, split, row, col, 
                          x_order, y_order, split_order, row_order, col_order,
                          orientation, sort, wrap, figsize, title, sharex, 
                          sharey, xlabel, ylabel, xlim, ylim, xscale, yscale, cmap,
                          x_textwrap, y_textwrap)

        default_bar_kwargs = {'ec': 'white', 'alpha': .9}
        if bar_kwargs is None:
            bar_kwargs = default_bar_kwargs
        else:
            try:
                bar_kwargs = {**default_bar_kwargs, **bar_kwargs}
            except:
                raise TypeError('`bar_kwargs` must be a dictionary')

        for ax, info in self.final_data.items():
            cur_size = size
            if not stacked:
                cur_size /= len(info)
            base = None
            for i, (x, y, label, col_name, row_label, col_label) in enumerate(info):
                x_plot, y_plot = self.get_x_y_plot(x, y)
                if base is None:
                    base = np.zeros(len(x_plot))
                if len(x) > 200:
                    warnings.warn('You are plotting more than 200 bars. '
                                  'Did you forget to privde an `aggfunc`?')

                if self.orientation == 'v':
                    x_plot = x_plot + cur_size * i * (1 - stacked)
                    ax.bar(x_plot, y_plot, label=label, width=cur_size, 
                           bottom=base, align='edge', **bar_kwargs)
                    if stacked:
                        base += y_plot
                else:
                    y_plot = y_plot + cur_size * i * (1 - stacked)
                    ax.barh(y_plot, x_plot, label=label, height=cur_size, 
                            left=base, align='edge', **bar_kwargs)
                    if stacked:
                        base += x_plot
            self.add_ticklabels(x, y, ax, delta=size / 2)

        self.add_legend()
        self.update_fig_size(len(info), len(x))
        return self.clean_up()

def count(val, data=None, normalize=None, split=None, row=None, col=None, 
          x_order=None, y_order=None, split_order=None, row_order=None, col_order=None,
          orientation='v', stacked=False, sort='lex_asc', wrap=None, figsize=None, title=None, 
          sharex=True, sharey=True, xlabel=None, ylabel=None, xlim=None, ylim=None, 
          xscale='linear', yscale='linear', cmap=None, size=.92, x_textwrap=10, y_textwrap=None):

        x, y = (val, None) if orientation == 'v' else (None, val)
        aggfunc = 'count'
        self = CommonPlot(x, y, data, aggfunc, split, row, col, 
                          x_order, y_order, split_order, row_order, col_order,
                          orientation, sort, wrap, figsize, title, sharex, 
                          sharey, xlabel, ylabel, xlim, ylim, xscale, yscale, cmap,
                          x_textwrap, y_textwrap)

        for ax, info in self.final_data.items():
            cur_size = size / len(info)
            for i, (x, y, label, col_name, row_label, col_label) in enumerate(info):
                x_plot, y_plot = self.get_x_y_plot(x, y)
                if self.orientation == 'v':
                    x_plot = x_plot + cur_size * i
                    ax.bar(x_plot, y_plot, label=label, width=cur_size, align='edge')
                else:
                    y_plot = y_plot + cur_size * i
                    ax.barh(y_plot, x_plot, label=label, height=cur_size, align='edge')
            self.add_ticklabels(x, y, ax, delta=size / 2)

        self.add_legend()
        self.update_fig_size(len(info), len(x))
        return self.clean_up()

def _common_dist(x=None, y=None, data=None, split=None, row=None, col=None, x_order=None, 
        y_order=None, split_order=None, row_order=None, col_order=None, orientation='h', 
        sort='lex_asc', wrap=None, figsize=None, title=None, sharex=True, sharey=True, xlabel=None, 
        ylabel=None, xlim=None, ylim=None, xscale='linear', yscale='linear', cmap=None, 
        x_textwrap=10, y_textwrap=None, kind=None, **kwargs):
        
        aggfunc = '__distribution__'
        self = CommonPlot(x, y, data, aggfunc, split, row, col, 
                          x_order, y_order, split_order, row_order, col_order,
                          orientation, sort, wrap, figsize, title, sharex, 
                          sharey, xlabel, ylabel, xlim, ylim, xscale, yscale, cmap,
                          x_textwrap, y_textwrap)

        key = 'bodies' if kind == 'violinplot' else 'boxes'
        vert = self.orientation == 'v'
        for ax, info in self.final_data.items():
            plot_func = getattr(ax, kind)
            cur_data = defaultdict(list)
            cur_ticklabels = defaultdict(list)
            for x, y, split_label, col_name, row_label, col_label in info:
                x_plot, y_plot = self.get_x_y_plot(x, y)
                if vert:
                    cur_data[split_label].append(y_plot)
                else:
                    cur_data[split_label].append(x_plot)
                cur_ticklabels[split_label].append(col_name)

            handles = []
            split_labels = []
            ticklabels = []
            other_info = {}
            n_splits = len(cur_data)
            widths = min(.5 + .1 * n_splits, .9) / n_splits
            n_boxes = len(info)
            n = len(next(iter(cur_data.values())))  # number of groups
            for i, (split_label, data) in enumerate(cur_data.items()):
                markersize = max(6 - n_boxes // 5, 2)
                positions = np.arange(n) + i * widths
                filt = [len(arr) > 0 for arr in data]
                positions = [p for (p, f) in zip(positions, filt) if f]
                data = [d for (d, f) in zip(data, filt) if f]
                if kind == 'boxplot':
                    kwargs['boxprops'] = {'facecolor': self.colors[i % len(self.colors)] ,
                                          'edgecolor': 'black'}
                    kwargs['flierprops'] = {'markersize': markersize}
                ret = plot_func(data, vert=vert, positions=positions, widths=widths, **kwargs)
                if kind == 'violinplot':
                    for k in ['cmeans', 'cmins', 'cmaxes', 'cbars', 'cmedians', 'cquantiles']:
                        if k in ret:
                            ret[k].set_linewidth(1)
                handles.append(ret[key][0])
                split_labels.append(split_label)
            delta = (n_splits / 2 - .5) * widths
            ticklabels = cur_ticklabels[split_label]
            if vert:
                ax.set_xticks(np.arange(n) + delta)
                ax.set_xticklabels(ticklabels)
            else:
                ax.set_yticks(np.arange(n) + delta)
                ax.set_yticklabels(ticklabels)

        self.add_legend(handles, split_labels)
        self.update_fig_size(n_splits, n)
        return self.clean_up()

# could add groupby to box
def box(x=None, y=None, data=None, split=None, row=None, col=None, x_order=None, 
        y_order=None, split_order=None, row_order=None, col_order=None, orientation='h', 
        sort='lex_asc', wrap=None, figsize=None, title=None, sharex=True, sharey=True, xlabel=None, 
        ylabel=None, xlim=None, ylim=None, xscale='linear', yscale='linear', cmap=None, 
        x_textwrap=10, y_textwrap=None, notch=None, sym=None, whis=None, patch_artist=True, 
        bootstrap=None, usermedians=None, conf_intervals=None, meanline=None, showmeans=None, 
        showcaps=None, showbox=None, showfliers=None, boxprops=None, labels=None, flierprops=None, 
        medianprops=None, meanprops=None, capprops=None, whiskerprops=None, manage_ticks=True,
        autorange=False, zorder=None):

    if medianprops is None:
        medianprops = {'color': '.2'}
    
    kwargs = dict(notch=notch, sym=sym, whis=whis, patch_artist=patch_artist,
                  bootstrap=bootstrap, usermedians=usermedians, conf_intervals=conf_intervals, 
                  meanline=meanline, showmeans=showmeans, showcaps=showcaps, showbox=showbox, 
                  showfliers=showfliers, boxprops=boxprops, labels=labels, flierprops=flierprops,
                  medianprops=medianprops, meanprops=meanprops, capprops=capprops, 
                  whiskerprops=whiskerprops, manage_ticks=manage_ticks, 
                  autorange=autorange, zorder=zorder)
    
    return _common_dist(x, y, data, split, row, col, x_order, y_order, split_order, 
                        row_order, col_order, orientation, sort, wrap, figsize, title, sharex, 
                        sharey, xlabel, ylabel, xlim, ylim, xscale, yscale, cmap, 
                        x_textwrap, y_textwrap, kind='boxplot', **kwargs)


def violin(x=None, y=None, data=None, split=None, row=None, col=None, x_order=None, y_order=None, 
           split_order=None, row_order=None, col_order=None, orientation='h', sort='lex_asc', 
           wrap=None, figsize=None, title=None, sharex=True, sharey=True, xlabel=None, 
           ylabel=None, xlim=None, ylim=None, xscale='linear', yscale='linear', cmap=None, 
           x_textwrap=10, y_textwrap=None, showmeans=False, showextrema=True, showmedians=True, 
           quantiles=None, points=100, bw_method=None):

    kwargs = dict(showmeans=showmeans, showextrema=showextrema, showmedians=showmedians, 
                  quantiles=quantiles, points=points, bw_method=bw_method)

    return _common_dist(x, y, data, split, row, col, 
                        x_order, y_order, split_order, row_order, col_order,
                        orientation, sort, wrap, figsize, title, sharex, 
                        sharey, xlabel, ylabel, xlim, ylim, xscale, yscale, cmap, 
                        x_textwrap, y_textwrap, kind='violinplot', **kwargs)


def hist(val, data=None, split=None, row=None, col=None, x_order=None, y_order=None, 
         split_order=None, row_order=None, col_order=None, orientation='v', sort='lex_asc', 
         wrap=None, figsize=None, title=None, sharex=True, sharey=True, xlabel=None, 
         ylabel=None, xlim=None, ylim=None, xscale='linear', yscale='linear', cmap=None, 
         x_textwrap=10, y_textwrap=None, bins=None, range=None, density=False, weights=None, 
         cumulative=False, bottom=None, histtype='bar', align='mid', rwidth=None, log=False,
         stacked=False):

        x, y = (val, None) if orientation == 'v' else (None, val)
        kwargs = dict(bins=bins, range=range, density=density, weights=weights, 
                      cumulative=cumulative, bottom=bottom, histtype=histtype, align=align, 
                      rwidth=rwidth, log=log, stacked=stacked)

        self = CommonPlot(x, y, data, aggfunc, split, row, col, 
                          x_order, y_order, split_order, row_order, col_order,
                          orientation, sort, wrap, figsize, title, sharex, 
                          sharey, xlabel, ylabel, xlim, ylim, xscale, yscale, cmap,
                          x_textwrap, y_textwrap)

        orientation = 'vertical' if self.orientation == 'v' else 'horizontal'
        for ax, info in self.final_data.items():
            handles = []
            labels = []
            n_splits = len(info)
            for x, y, label, col_name, row_label, col_label in info:
                val = y if orientation == 'vertical' else x
                ax.hist(val, orientation=orientation, label=label, **kwargs, 
                        ec='white', lw=1, alpha=.8)

        self.add_legend()
        self.update_fig_size(n_splits, len(x))
        return self.clean_up()

# """
# The `aggplot` function aggregates a single column of data. To begin,
# choose the column you would like to aggregate and set it as the `agg`
# parameter. The behavior of `aggplot` changes based on the type of
# variable used for `agg`.

# For numeric columns, the average of the values are calculated by default.
# Use the `aggfunc` parameter to choose the type of aggregation. You may
# use strings such as 'min', 'max', 'median', etc...

# For string and categorical columns, the counts of the unique values are
# calculated by default. Use the `normalize` parameter to return the
# percentages instead of the counts. Choose how you would like to
# `normalize` by setting it to one of the strings 'agg', 'split', 'row',
# 'col', or 'all'.

# Use the `groupby` parameter to select a column to group by. This column
# is passed to the pandas DataFrame `groupby` method. Choose the aggregation
# method with `aggfunc`. Note, that you cannot use set `groupby` if the `agg`
# variable is string/categorical.

# Parameters
# ----------
# agg: str
#     Column name of DataFrame you would like to aggregate. By default, the
#     mean of numeric columns and the counts of unique values of
#     string/categorical columns are returned.

# groupby: str
#     Column name of the grouping variable. Only available when the `agg`
#     variable is numeric.

# data: DataFrame
#     A Pandas DataFrame that typically has non-aggregated data.
#     This type of data is often referred to as "tidy" or "long" data.

# split: str
#     Column name to further group the `agg` variable within a single plot.
#     Each unique value in the `split` column forms a new group.

# row: str
#     Column name used to group data into separate plots. Each unique value
#     in the `row` column forms a new row.

# col: str
#     Column name used to group data into separate plots. Each unique value
#     in the `col` column forms a new row.

# kind: str
#     Type of plot to use. Possible choices for all `agg` variables:
#     * 'bar'
#     * 'line'

#     Additional choices for numeric `agg` variables
#     * 'hist'
#     * 'kde'
#     * 'box'

# orientation: str {'v', 'h'}
#     Choose the orientation of the plots. By default, they are vertical
#     ('v'). Use 'h' for horizontal

# sort: bool - default is False
#     Whether to sort the `groupby` variables

# aggfunc: str or function
#     Used to aggregate `agg` variable. Use any of the strings that Pandas
#     can understand. You can also use a custom function as long as it
#     aggregates, i.e. returns a single value.

# normalize: str, tuple
#     When aggregating a string/categorical column, return the percentage
#     instead of the counts. Choose what columns you would like to
#     normalize over with the strings 'agg', 'split', 'row', 'col', or 'all'.
#     Use a tuple of these strings to normalize over two or more of the
#     above. For example, use ('split', 'row') to normalize over the
#     `split`, `row` combination.

# wrap: int
#     When using either `row` or either `col` and not both, determines the
#     maximum number of rows/cols before a new row/col is used.

# stacked: bool
#     Controls whether bars will be stacked on top of each other

# figsize: tuple
#     Use a tuple of integers. Passed directly to Matplotlib to set the
#     size of the figure in inches.

# rot: int
#     Long labels will be automatically wrapped, but you can still use
#     this parameter to rotate x-tick labels. Only applied to strings.

# title: str
#     Sets the figure title NOT the Axes title

# sharex: bool
#     Whether all plots should share the x-axis or not. Default is True

# sharey: bool
#     Whether all plots should share the y-axis or not. Default is True

# xlabel: str
#     Label used for x-axis on figures with a single plot

# ylabel: str
#     Label used for y-axis on figures with a single plot

# xlim: 2-item tuple of numerics
#     Determines x-axis limits for figures with a single plot

# ylim: 2-item tuple of numerics
#     Determines y-axis limits for figures with a single plot

# xscale: {'linear', 'log', 'symlog', 'logit'}
#     Sets the scale of the x-axis.

# yscale: {'linear', 'log', 'symlog', 'logit'}
#     Sets the scale of the y-axis

# kwargs: dict
#     Extra arguments used to control the plot used as the `kind`
#     parameter.

# Returns
# -------
# A Matplotlib Axes whenever both `row` and `col` are not defined and a
# Matplotlib Figure when one or both are.

# """