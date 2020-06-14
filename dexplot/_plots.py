import warnings
from collections import defaultdict

import numpy as np
import pandas as pd
from scipy import stats

from ._common_plot import CommonPlot


def get_bar_kwargs(bar_kwargs):
    default_bar_kwargs = {'ec': 'white', 'alpha': .9}
    if bar_kwargs is None:
        bar_kwargs = default_bar_kwargs
    else:
        try:
            bar_kwargs = {**default_bar_kwargs, **bar_kwargs}
        except:
            raise TypeError('`bar_kwargs` must be a dictionary')
    return bar_kwargs

def line(x=None, y=None, data=None, aggfunc=None, split=None, row=None, col=None, 
         x_order=None, y_order=None, split_order=None, row_order=None, col_order=None,
         orientation='v', sort_values=None, wrap=None, figsize=None, title=None, sharex=True, 
         sharey=True, xlabel=None, ylabel=None, xlim=None, ylim=None, xscale='linear', 
         yscale='linear', cmap=None, x_textwrap=10, y_textwrap=None, x_rot=None, y_rot=None):
        
        self = CommonPlot(x, y, data, aggfunc, split, row, col, 
                          x_order, y_order, split_order, row_order, col_order,
                          orientation, sort_values, wrap, figsize, title, sharex, 
                          sharey, xlabel, ylabel, xlim, ylim, xscale, yscale, cmap,
                          x_textwrap, y_textwrap, x_rot, y_rot)
        
        marker = 'o' if self.groupby else None

        for ax, info in self.final_data.items():
            for x, y, label, col_name, row_label, col_label in info:
                x_plot, y_plot = self.get_x_y_plot(x, y)
                ax.plot(x_plot, y_plot, label=label, marker=marker)
            
            if self.groupby:
                ticklabels = x if self.orientation == 'v' else y
                self.add_ticklabels(ticklabels, ax)
        
        self.add_legend(label)
        if x.dtype == 'O' or y.dtype == 'O':
            self.update_fig_size(len(x), 1)
        return self.clean_up()
        
def scatter(x=None, y=None, data=None, aggfunc=None, split=None, row=None, col=None, 
            x_order=None, y_order=None, split_order=None, row_order=None, col_order=None,
            orientation='v', sort_values=None, wrap=None, figsize=None, title=None, sharex=True, 
            sharey=True, xlabel=None, ylabel=None, xlim=None, ylim=None, xscale='linear', 
            yscale='linear', cmap=None, x_textwrap=10, y_textwrap=None, x_rot=None, y_rot=None, 
            regression=False):

        self = CommonPlot(x, y, data, aggfunc, split, row, col, 
                          x_order, y_order, split_order, row_order, col_order,
                          orientation, sort_values, wrap, figsize, title, sharex, 
                          sharey, xlabel, ylabel, xlim, ylim, xscale, yscale, cmap,
                          x_textwrap, y_textwrap, x_rot, y_rot)

        alpha = 1 if self.groupby else .7

        for ax, info in self.final_data.items():
            for x, y, label, col_name, row_label, col_label in info:
                x_plot, y_plot = self.get_x_y_plot(x, y)
                ax.scatter(x_plot, y_plot, label=label, alpha=alpha)
                if regression:
                    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
                    x_line = np.array([x.min(), x.max()])
                    y_line = x_line * slope + intercept
                    ax.plot(x_line, y_line)
            if self.groupby:
                ticklabels = x if self.orientation == 'v' else y
                self.add_ticklabels(ticklabels, ax)

        self.add_legend(label)
        if x.dtype == 'O' or y.dtype == 'O':
            self.update_fig_size(len(x), 1)
        return self.clean_up()

def bar(x=None, y=None, data=None, aggfunc=None, split=None, row=None, col=None, 
        x_order=None, y_order=None, split_order=None, row_order=None, col_order=None,
        orientation='v', sort_values=None, wrap=None, figsize=None, title=None, 
        sharex=True, sharey=True, xlabel=None, ylabel=None, xlim=None, 
        ylim=None, xscale='linear', yscale='linear', cmap=None, x_textwrap=10, 
        y_textwrap=None, x_rot=None, y_rot=None, size=.92, stacked=False, bar_kwargs=None):

        self = CommonPlot(x, y, data, aggfunc, split, row, col, 
                          x_order, y_order, split_order, row_order, col_order,
                          orientation, sort_values, wrap, figsize, title, sharex, 
                          sharey, xlabel, ylabel, xlim, ylim, xscale, yscale, cmap,
                          x_textwrap, y_textwrap, x_rot, y_rot)

        bar_kwargs = get_bar_kwargs(bar_kwargs)
        for ax, info in self.final_data.items():
            cur_size = size if stacked else size / len(info)
            for i, (x, y, label, col_name, row_label, col_label) in enumerate(info):
                x_plot, y_plot = self.get_x_y_plot(x, y)
                if i == 0:
                    base = np.zeros(len(x_plot))
                if len(x) > 200:
                    warnings.warn('You are plotting more than 200 bars. '
                                  'Did you forget to provide an `aggfunc`?')

                if self.orientation == 'v':
                    x_plot = x_plot + cur_size * i * (1 - stacked)
                    ax.bar(x_plot, y_plot, label=label, width=cur_size, 
                           bottom=base, align='edge', **bar_kwargs)
                    if stacked:
                        base += y_plot
                else:
                    y_plot = y_plot - cur_size * (i + 1) * (1 - stacked)
                    ax.barh(y_plot, x_plot, label=label, height=cur_size, 
                            left=base, align='edge', **bar_kwargs)
                    if stacked:
                        base += x_plot
            ticklabels = x if self.orientation == 'v' else y
            delta = size / 2
            self.add_ticklabels(ticklabels, ax, delta=delta)

        self.add_legend(label)
        self.update_fig_size(len(info), len(x))
        return self.clean_up()

def count(val, data=None, normalize=False, split=None, row=None, col=None, 
          x_order=None, y_order=None, split_order=None, row_order=None, col_order=None,
          orientation='v', sort_values='desc', wrap=None, figsize=None, title=None, 
          sharex=True, sharey=True, xlabel=None, ylabel=None, xlim=None, ylim=None, 
          xscale='linear', yscale='linear', cmap=None, x_textwrap=10, y_textwrap=None, 
          x_rot=None, y_rot=None, size=.92, stacked=False, bar_kwargs=None):
       
        bar_kwargs = get_bar_kwargs(bar_kwargs)
        x, y = (val, None) if orientation == 'v' else (None, val)
        aggfunc = '__distribution__'
        self = CommonPlot(x, y, data, aggfunc, split, row, col, 
                          x_order, y_order, split_order, row_order, col_order,
                          orientation, None, wrap, figsize, title, sharex, 
                          sharey, xlabel, ylabel, xlim, ylim, xscale, yscale, cmap,
                          x_textwrap, y_textwrap, x_rot, y_rot, kind='count')

        count_dict = {}

        if isinstance(normalize, str):
            if normalize in (val, self.split, self.row, self.col):
                normalize = [normalize]
        
        if isinstance(normalize, tuple):
            normalize = list(normalize)
        elif hasattr(normalize, 'tolist'):
            normalize = normalize.tolist()
        elif not isinstance(normalize, (bool, list)):
            raise ValueError('`normalize` must either be `True`/`False`, one of the columns passed '
                                 'to `val`, `split`, `row` or `col`, or a list of '
                                 'those columns')
        normalize_kind = None
        if isinstance(normalize, list):
            row_col = []
            val_split = []
            for col in normalize:
                if col in (self.row, self.col):
                    row_col.append(col)
                elif col in (val, self.split):
                    val_split.append(col)
                else:
                    raise ValueError('Columns passed to `normalize` must be the same as '
                                     ' `val`, `split`, `row` or `col`.')

            if row_col:
                all_counts = {}
                for grp, data in self.data.groupby(row_col):
                    if len(row_col) == 1:
                        grp = str(grp)
                    else:
                        grp = tuple(str(g) for g in grp)

                    if val_split:
                        normalize_kind = 'all'
                        all_counts[grp] = data.groupby(val_split).size()
                    else:
                        normalize_kind = 'grid'
                        all_counts[grp] = len(data)
            else:
                normalize_kind = 'single'
                all_counts = self.data.groupby(val_split).size()

        n = 0
        for ax, info in self.final_data.items():
            columns = []
            vcs = []
            for vals, split_label, col_name, row_label, col_label in info:
                vcs.append(vals.value_counts())
                columns.append(split_label)

            df = pd.concat(vcs, axis=1)
            df.columns = columns
            df.index.name = vals.name
            if normalize_kind == 'single':
                if len(val_split) == 2:
                    df = df / all_counts.unstack(self.split)
                elif df.index.name == all_counts.index.name:
                    df = df.div(all_counts, axis=0)
                else:
                    df = df / all_counts
            elif normalize_kind in ('grid', 'all'):
                grp = []
                for col in normalize:
                    if col == self.row:
                        grp.append(row_label)
                    if col == self.col:
                        grp.append(col_label)
                
                if len(grp) == 1:
                    grp = grp[0]
                else:
                    grp = tuple(grp)
                grp_val = all_counts[grp]
                
                if normalize_kind == 'grid':
                    df = df / grp_val
                elif len(val_split) == 2:
                    df = df / grp_val.unstack(self.split)
                elif df.index.name == grp_val.index.name:
                    df = df.div(grp_val, axis=0)
                else:
                    df = df / grp_val

            else:
                n += df.sum().sum()
            count_dict[ax] = df
            
        if normalize is True:
            count_dict = {ax: df / n for ax, df in count_dict.items()}
        

        for ax, df in count_dict.items():
            base = np.zeros(len(df))
            cur_size = size if stacked else size / len(df.columns)
            position = np.arange(len(df))

            if sort_values == 'asc' and not (self.split or self.row or self.col):
                df = df.iloc[::-1]

            ticklabels = df.index.values
                
            for col in df.columns:
                values = df[col].values
                
                if self.orientation == 'v':
                    ax.bar(position, values, label=col, width=cur_size, 
                            bottom=base, align='edge', **bar_kwargs)
                    position = position + cur_size * (1 - stacked)
                else:
                    ax.barh(position - cur_size, values, label=col, height=cur_size, 
                            left=base, align='edge', **bar_kwargs)
                    position = position - cur_size * (1 - stacked)

                if stacked:
                    base += values
                
            self.add_ticklabels(ticklabels, ax, delta=size / 2)
        if self.split or len(df.columns) > 1:
            self.add_legend(col)
        self.update_fig_size(len(info), len(df))
        return self.clean_up()


def _common_dist(x=None, y=None, data=None, split=None, row=None, col=None, x_order=None, 
                 y_order=None, split_order=None, row_order=None, col_order=None, orientation='h', 
                 wrap=None, figsize=None, title=None, sharex=True, sharey=True, xlabel=None, 
                 ylabel=None, xlim=None, ylim=None, xscale='linear', yscale='linear', cmap=None, 
                 x_textwrap=10, y_textwrap=None, x_rot=None, y_rot=None, kind=None, **kwargs):

        aggfunc = '__distribution__'
        sort_values = None
        self = CommonPlot(x, y, data, aggfunc, split, row, col, 
                          x_order, y_order, split_order, row_order, col_order,
                          orientation, sort_values, wrap, figsize, title, sharex, 
                          sharey, xlabel, ylabel, xlim, ylim, xscale, yscale, cmap,
                          x_textwrap, y_textwrap, x_rot, y_rot)

        key = 'bodies' if kind == 'violinplot' else 'boxes'
        vert = self.orientation == 'v'
        for ax, info in self.final_data.items():
            plot_func = getattr(ax, kind)
            cur_data, cur_ticklabels = self.get_distribution_data(info)

            handles = []
            split_labels = []
            n_splits = len(cur_data)
            widths = min(.5 + .15 * n_splits, .9) / n_splits
            n_boxes = len(info)
            n = len(next(iter(cur_data.values())))  # number of groups
            markersize = max(6 - n_boxes // 5, 2)
            for i, (split_label, data) in enumerate(cur_data.items()):
                filt = [len(arr) > 0 for arr in data]
                positions = np.array([i for (i, f) in enumerate(filt) if f])
                data = [np.array(d) for (d, f) in zip(data, filt) if f]
                if self.orientation == 'h':
                    positions = positions - i * widths
                else:
                    positions = positions + i * widths
                
                if kind == 'boxplot':
                    kwargs['boxprops'] = {'facecolor': self.colors[i % len(self.colors)] ,
                                          'edgecolor': 'black'}
                    kwargs['flierprops'] = {'markersize': markersize}
                
                ret = plot_func(data, vert=vert, positions=positions, widths=widths, **kwargs)

                if kind == 'violinplot':
                    for k in ['cmeans', 'cmins', 'cmaxes', 'cbars', 'cmedians', 'cquantiles']:
                        if k in ret:
                            ret[k].set_linewidth(1)
                    for body in ret['bodies']:
                        body.set_alpha(.8)

                handles.append(ret[key][0])
                split_labels.append(split_label)
            
            delta = (n_splits / 2 - .5) * widths
            ticklabels = cur_ticklabels[split_label]
            self.add_ticklabels(ticklabels, ax, delta=delta)

        self.add_legend(self.split, handles, split_labels)
        self.update_fig_size(n_splits, n)
        return self.clean_up()

# could add groupby to box
def box(x=None, y=None, data=None, split=None, row=None, col=None, x_order=None, 
        y_order=None, split_order=None, row_order=None, col_order=None, orientation='h', 
        wrap=None, figsize=None, title=None, sharex=True, sharey=True, xlabel=None, 
        ylabel=None, xlim=None, ylim=None, xscale='linear', yscale='linear', cmap=None, 
        x_textwrap=10, y_textwrap=None, x_rot=None, y_rot=None, notch=None, sym=None, whis=None, 
        patch_artist=True, bootstrap=None, usermedians=None, conf_intervals=None, meanline=None, 
        showmeans=None, showcaps=None, showbox=None, showfliers=None, boxprops=None, labels=None, 
        flierprops=None, medianprops=None, meanprops=None, capprops=None, whiskerprops=None, 
        manage_ticks=True, autorange=False, zorder=None):

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
                        row_order, col_order, orientation, wrap, figsize, title, 
                        sharex, sharey, xlabel, ylabel, xlim, ylim, xscale, yscale, cmap, 
                        x_textwrap, y_textwrap, x_rot, y_rot, kind='boxplot', **kwargs)


def violin(x=None, y=None, data=None, split=None, row=None, col=None, x_order=None, 
           y_order=None, split_order=None, row_order=None, col_order=None, orientation='h', 
           wrap=None, figsize=None, title=None, sharex=True, sharey=True, xlabel=None, 
           ylabel=None, xlim=None, ylim=None, xscale='linear', yscale='linear', cmap=None, 
           x_textwrap=10, y_textwrap=None, x_rot=None, y_rot=None, showmeans=False, 
           showextrema=True, showmedians=True, quantiles=None, points=100, bw_method=None):

    kwargs = dict(showmeans=showmeans, showextrema=showextrema, showmedians=showmedians, 
                  quantiles=quantiles, points=points, bw_method=bw_method)

    return _common_dist(x, y, data, split, row, col, 
                        x_order, y_order, split_order, row_order, col_order,
                        orientation, wrap, figsize, title, sharex, 
                        sharey, xlabel, ylabel, xlim, ylim, xscale, yscale, cmap, 
                        x_textwrap, y_textwrap, x_rot, y_rot, kind='violinplot', **kwargs)


def hist(val, data=None, split=None, row=None, col=None, split_order=None, row_order=None, 
         col_order=None, orientation='v', wrap=None, figsize=None, title=None, 
         sharex=True, sharey=True, xlabel=None, ylabel=None, xlim=None, ylim=None, xscale='linear', 
         yscale='linear', cmap=None, x_textwrap=10, y_textwrap=None, x_rot=None, y_rot=None, 
         bins=None, range=None, density=False, weights=None, cumulative=False, bottom=None, 
         histtype='bar', align='mid', rwidth=None, log=False):
        
        x_order = y_order = None
        x, y = (val, None) if orientation == 'v' else (None, val)
        bins = bins if bins else 20
        kwargs = dict(bins=bins, range=range, density=density, weights=weights, 
                      cumulative=cumulative, bottom=bottom, histtype=histtype, align=align, 
                      rwidth=rwidth, log=log)

        aggfunc = '__distribution__'
        sort_values = None
        self = CommonPlot(x, y, data, aggfunc, split, row, col, 
                          x_order, y_order, split_order, row_order, col_order,
                          orientation, sort_values, wrap, figsize, title, sharex, 
                          sharey, xlabel, ylabel, xlim, ylim, xscale, yscale, cmap,
                          x_textwrap, y_textwrap, x_rot, y_rot)

        orientation = 'vertical' if self.orientation == 'v' else 'horizontal'
        for ax, info in self.final_data.items():
            cur_data, cur_ticklabels = self.get_distribution_data(info)

            handles = []
            split_labels = []
            n_splits = len(cur_data)
            n = len(next(iter(cur_data.values())))  # number of groups
            for split_label, data in cur_data.items():
                filt = [len(arr) > 0 for arr in data]
                vals = [d for (d, f) in zip(data, filt) if f]    
                ret = ax.hist(vals, orientation=orientation, alpha=.8, **kwargs)
                handles.append(ret[-1][0])
                split_labels.append(split_label)

        self.add_legend(self.split, handles, split_labels)
        # self.update_fig_size(n_splits, n)
        return self.clean_up()


def kde(x=None, y=None, data=None, split=None, row=None, col=None, split_order=None, 
        row_order=None, col_order=None, orientation='v', wrap=None, figsize=None, 
        title=None, sharex=True, sharey=True, xlabel=None, ylabel=None, xlim=None, 
        ylim=None, xscale='linear', yscale='linear', cmap=None, x_textwrap=10, y_textwrap=None, 
        x_rot=None, y_rot=None, range=None, cumulative=False):
        
        from ._utils import calculate_density_1d, calculate_density_2d

        x_order = y_order = None
        # x, y = (x, None) if orientation == 'v' else (None, x)
        kwargs = dict(range=range, cumulative=cumulative)

        if x is not None and y is not None and split is not None:
            raise ValueError('Cannot use `split` for 2-dimensional KDE plots')

        aggfunc = '__distribution__' if y is None else None
        sort_values = None
        self = CommonPlot(x, y, data, aggfunc, split, row, col, 
                          x_order, y_order, split_order, row_order, col_order,
                          orientation, sort_values, wrap, figsize, title, sharex, 
                          sharey, xlabel, ylabel, xlim, ylim, xscale, yscale, cmap,
                          x_textwrap, y_textwrap, x_rot, y_rot, check_numeric=True)

        for ax, info in self.final_data.items():
            for vals in info:
                if aggfunc == '__distribution__':
                    x, split_label = vals[:2]
                    x, y = calculate_density_1d(x, cumulative=cumulative)
                    x, y = (x, y) if self.orientation == 'v' else (y, x)
                    ax.plot(x, y, label=split_label)
                else:
                    x, y, split_label = vals[:3]
                    xmin, xmax, ymin, ymax, Z = calculate_density_2d(x, y)
                    ax.imshow(Z, extent=[xmin, xmax, ymin, ymax], aspect='auto')
                
        self.add_legend(self.split)
        # self.update_fig_size(n_splits, n)
        return self.clean_up()

xy_doc = """\
x : str, default None
    Column name of DataFrame whose values will go along the x-axis

y : str, default None
    Column name of DataFrame whose values will go along the y-axis
"""

val_doc = """\
val : str, default None
    Column name of DataFrame whose values will be used for distribution
"""

aggfunc_doc = """\
aggfunc : str or function, default None
    Kind of aggregation to perform. Use a string that the DataFrame `agg` 
    method understands. If providing a function, it will also be passed to
    the `agg` method.

    The strings 'countna' and 'percna' are also available to find the
    number and percentage of missing values.
"""

xy_order = """\
x_order : str or list, default None
    Used as both a way to order and filter the x-values. Use the strings
    'asc'/'desc' to order ascending or descending.

    Set a specific order with a list, i.e. `['House', 'Apartment', 'Townhouse']`

    Use the strings `'top n'` or `'bottom n'` where `n` is an integer. This will
    filter for the most/least frequent groups.

    By default, sorting happens in ascending order.

y_order : str or list, default None
    See x_order

split_order : str or list, default None
    See x_order

row_order : str or list, default None
    See x_order
    
col_order : str or list, default None
    See x_order
"""

split_order = """\
split_order : str or list, default None
    Used as both a way to order and filter the x-values. Use the strings
    'asc'/'desc' to order ascending or descending.

    Set a specific order with a list, i.e. `['House', 'Apartment', 'Townhouse']`

    Use the strings `'top n'` or `'bottom n'` where `n` is an integer. This will
    filter for the most/least frequent groups.

    By default, sorting happens in ascending order.

row_order : str or list, default None
    See split_order
    
col_order : str or list, default None
    See split_order
"""

sort_values_doc = """
sort_values : str - 'asc' or 'desc', default None
    Sort the values ascending or descending. If this is given, then
    x/y_order is ignored.
"""

doc = \
"""
{plot_doc}

Parameters
----------
{xy}

data : DataFrame or Series, default None
    A pandas DataFrame with long or wide data. If provided a Series, do not
    supply x or y.

{aggfunc}

split : str, default None
    Column name that will be used in the DataFrame `groupby` method to 
    split the data into independent groups within a single plot

row : str
    Column name that will be used in the DataFrame `groupby` method to 
    split the data into independent groups to form new plots. Each unique value
    in the `row` column forms a new row of plots.

col : str
    Column name that will be used in the DataFrame `groupby` method to 
    split the data into independent groups to form new plots. Each unique value
    in the `row` column forms a new row of plots.

{order}

orientation : str 'v' or 'h'
    Choose the orientation of the plots. By default, they are vertical
    ('v'), except for box and violin plots, which are horizontal.

{sort_values}

wrap : int, default None
    When using either `row` or either `col`, but not both, determines the
    maximum number of rows/cols before a new row/col is used.

figsize : tuple, default None
    A tuple of numbers used passed to the `figsize` matplotlib parameter. 
    By default, the figure size will be determined based on the kind of
    plot produced.

title : str
    Sets the figure title NOT the Axes title

sharex : bool
    Whether all plots should share the x-axis or not. Default is True

sharey : bool
    Whether all plots should share the y-axis or not. Default is True

xlabel : str
    Label used for x-axis on figures with a single plot

ylabel : str
    Label used for y-axis on figures with a single plot

xlim : 2-item tuple of numbers
    Determines x-axis limits for figures with a single plot

ylim : 2-item tuple of numbers
    Determines y-axis limits for figures with a single plot

xscale : 'linear', 'log', 'symlog', 'logit'
    Sets the scale of the x-axis.

yscale : 'linear', 'log', 'symlog', 'logit'
    Sets the scale of the y-axis

cmap : str or matplotlib colormap instance, default None

x_textwrap : int, default 10
    Number of characters before wrapping text for x-labels

y_textwrap : int, default None
    Number of characters before wrapping text for y-labels

x_rot : int or float, default None
    Degree of rotation of x-tick labels. If between 0 and 180
    horizontal_alignment is set to 'right', otherwise 'left'

y_rot : int or float, default None
    Degree of rotation of y-tick labels. If between 0 and 180
    vertical_alignment is set to 'top', otherwise 'bottom'

Returns
-------
A Matplotlib Figure instance
"""


# line doc
line_doc = """\
Create line plots
"""

scatter_doc = """\
Create scatter plots
"""

bar_doc = """\
Create bar plots
"""

count_doc = """\
Create count plots
"""

box_doc = """\
Create box plots
"""

violin_doc = """\
Create violin plots
"""

hist_doc = """\
Create histograms
"""

kde_doc = """\
Create kernel density estimate plots
"""

line.__doc__ = doc.format(plot_doc=line_doc, xy=xy_doc, aggfunc=aggfunc_doc, 
                          order=xy_order, sort_values=sort_values_doc)

scatter.__doc__ = doc.format(plot_doc=scatter_doc, xy=xy_doc, aggfunc=aggfunc_doc, 
                          order=xy_order, sort_values=sort_values_doc)

bar.__doc__ = doc.format(plot_doc=bar_doc, xy=xy_doc, aggfunc=aggfunc_doc, 
                          order=xy_order, sort_values=sort_values_doc)

count.__doc__ = doc.format(plot_doc=count_doc, xy=val_doc, aggfunc='', 
                          order=split_order, sort_values=sort_values_doc)

box.__doc__ = doc.format(plot_doc=box_doc, xy=xy_doc, aggfunc='', 
                          order=xy_order, sort_values='')

violin.__doc__ = doc.format(plot_doc=violin_doc, xy=xy_doc, aggfunc='', 
                          order=xy_order, sort_values='')

hist.__doc__ = doc.format(plot_doc=hist_doc, xy=val_doc, aggfunc='', 
                          order=split_order, sort_values='')

kde.__doc__ = doc.format(plot_doc=kde_doc, xy=val_doc, aggfunc='', 
                          order=split_order, sort_values='')
