import textwrap
import warnings
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import ticker, category
from matplotlib.colors import Colormap
from scipy import stats


NONETYPE = type(None)

class CommonPlot:


    def __init__(self, x, y, data, groupby, aggfunc, split, row, col, 
                 x_order, y_order, split_order, row_order, col_order,
                 orientation, sort, wrap, figsize, title, sharex, sharey, 
                 xlabel, ylabel, xlim, ylim, xscale, yscale, cmap):

        self.orig_fontsize = plt.rcParams['font.size']
        plt.rcParams['font.size'] = 7
        self.groups = []
        self.data = self.get_data(data)
        self.x = self.get_col(x)
        self.y = self.get_col(y)
        self.validate_x_y()
        self.groupby = self.get_col(groupby, True)
        self.aggfunc = aggfunc
        self.split = self.get_col(split, True)
        self.row = self.get_col(row, True)
        self.col = self.get_col(col, True)
        self.orientation = orientation
        self.agg = self.set_agg()
        self.make_groups_categorical()
        
        self.x_order = x_order
        self.y_order = y_order
        self.split_order = split_order
        self.row_order = row_order
        self.col_order = col_order
        
        self.sort = sort
        self.wrap = wrap
        self.title = title
        self.sharex = sharex
        self.sharey = sharey
        self.xlabel = xlabel
        self.ylabel = ylabel
        self.xlim = xlim
        self.ylim = ylim
        self.xscale = xscale
        self.yscale = yscale
        self.colors = self.get_colors(cmap)

        self.validate_args(figsize)
        self.plot_type = self.get_plot_type()
        self.agg_kind = self.get_agg_kind()
        self.data = self.set_index()
        self.unique_rows, self.unique_cols = self.get_uniques()
        self.fig_shape = self.get_fig_shape()
        self.figsize = self.get_figsize(figsize)
        self.fig, self.axs = self.create_figure()
        self.set_color_cycle()
        self.data_for_plots = self.get_data_for_every_plot()
        self.final_data = self.get_final_data()
        self.style_fig()
        self.add_x_y_labels()
        self.add_ax_titles()

    def get_data(self, data):
        if not isinstance(data, pd.DataFrame):
            raise TypeError('`data` must be a pandas DataFrame')
        elif len(data) == 0:
            raise ValueError('DataFrame contains no data')
        return data

    def get_col(self, col, group=False):
        if col:
            try:
                col in self.data.columns
            except KeyError:
                raise KeyError(f'{col} is not a column in the DataFrame')

            if group:
                if col in self.groups:
                    raise ValueError(f'Column {col} has already been selected for another grouping column')
                else:
                    self.groups.append(col)
        
            return col

    def validate_x_y(self):
        if self.x == self.y:
            raise ValueError('`x` and `y` cannot be the same column name')

    def set_agg(self):
        if self.groupby is None:
            if self.aggfunc is not None:
                raise ValueError('You can only set `aggfunc` when `groupby` is given')
            return
        elif self.aggfunc is None:
            raise ValueError('If you provide `groupby`, then you must also provide `aggfunc`')
        
        if self.orientation == 'v':
            if self.x != self.groupby:
                raise ValueError('When grouping with vertical orientation, `x` and `groupby` must be the same columns')
            return self.y
        elif self.y != self.groupby:
            raise ValueError('When grouping with horizontal orientation, `y` and `groupby` must be the same columns')
        else:
            return self.x

    def make_groups_categorical(self):
        category_cols = [self.groupby, self.split, self.row, self.col]
        copied = False
        for col in category_cols:
            if col:
                if self.data[col].dtype.name != 'category':
                    if not copied:
                        self.data = self.data.copy()
                        copied = True
                    self.data[col] = self.data[col].astype('category')

    def get_colors(self, cmap):
        if cmap is None:
            cmap = 'dark12'
            
        if isinstance(cmap, str):
            from .colors._colormaps import colormaps
            try:
                return colormaps[cmap.lower()]
            except KeyError:
                raise KeyError(f'Colormap {cmap} does not exist. Here are the '
                               f'possible colormaps: {colormaps.keys()}')
        elif isinstance(cmap, Colormap):
            return cmap(range(cmap.N)).tolist()
        elif isinstance(cmap, list):
            return cmap
        elif isinstance(cmap, tuple):
            return list(cmap)
        elif hasattr(cmap, 'tolist'):
            return cmap.tolist()
        else:
            raise TypeError('`cmap` must be a string name of a colormap, a matplotlib colormap '
                            'instance, list, or tuple of colors')
        
    def validate_args(self, figsize):
        self.validate_figsize(figsize)
        self.validate_plot_args()
        self.validate_mpl_args()

    def validate_figsize(self, figsize):
        if isinstance(figsize, (list, tuple)):
            if len(figsize) != 2:
                raise ValueError('figsize must be a two-item tuple/list')
            for val in figsize:
                if not isinstance(val, (int, float)):
                    raise ValueError('Each item in figsize must be an integer or a float')
        elif figsize is not None:
            raise TypeError('figsize must be a two-item tuple')

    def validate_plot_args(self):
        if self.orientation not in ('v', 'h'):
            raise ValueError('`orientation` must be either "v" or "h".')

        if not isinstance(self.wrap, (np.integer, int, NONETYPE)):
            raise TypeError(f'`wrap` must either be None or an integer, not {type(wrap)}')

        if self.row and self.col and self.wrap is not None:
            raise ValueError('You cannot provide a value for `wrap` if `row` '
                             'and `col` are also provided')

        if not isinstance(self.sort, bool):
            raise TypeError('`sort` must be a bool')

    def validate_mpl_args(self):
        if not isinstance(self.title, (NONETYPE, str)):
            raise TypeError('`title` must be either None or a str')
        if self.sharex not in (False, True, None, 'row', 'col'):
            raise ValueError('`sharex` must be one of `False`, `True`, `None`, "row", or "col"')
        if self.sharey not in (False, True, None, 'row', 'col'):
            raise ValueError('`sharex` must be one of `False`, `True`, `None`, "row", or "col"')

        if not isinstance(self.xlabel, (NONETYPE, str)):
            raise TypeError('`xlabel` must be either None or a str')
        if not isinstance(self.ylabel, (NONETYPE, str)):
            raise TypeError('`ylabel` must be either None or a str')

        if not isinstance(self.xlim, (NONETYPE, tuple)):
            raise TypeError('`xlim` must be a two-item tuple of numerics or `None`')
        if not isinstance(self.ylim, (NONETYPE, tuple)):
            raise TypeError('`xlim` must be a two-item tuple of numerics or `None`')
        if self.xscale not in ('linear', 'log', 'symlog', 'logit'):
            raise ValueError("`xscale must be one of 'linear', 'log', 'symlog', 'logit'")
        if self.yscale not in ('linear', 'log', 'symlog', 'logit'):
            raise ValueError("`xscale must be one of 'linear', 'log', 'symlog', 'logit'")

    def get_plot_type(self):
        if self.row and self.col:
            return 'square'
        if self.row:
            return 'row_only'
        if self.col:
            return 'col_only'
        return 'single'

    def get_agg_kind(self):
        if self.agg:
            # string and category use 'O'
            agg_kind = self.data[self.agg].dtype.kind

            if agg_kind not in ['i', 'f', 'b', 'O']:
                raise TypeError('The data type for the `agg` column must either be boolean, integer, '
                                f'float, or categorical/object/string and not {agg_data.dtype}')
            return agg_kind

    def set_index(self):
        data = self.data
        rc = []
        if self.row:
            rc.append(self.row)
        if self.col:
            rc.append(self.col)
        if rc:
            data = data.set_index(rc)
        return data

    def get_uniques(self):
        if self.plot_type == 'single':
            return None, None
        elif self.plot_type == 'row_only':
            return self.data.index.unique(), None
        elif self.plot_type == 'col_only':
            return None, self.data.index.unique()
        else:
            return self.data.index.levels
        
    def get_fig_shape(self):
        if self.plot_type == 'single':
            return 1, 1

        nrows = ncols = 1
        if self.unique_rows is not None:
            nrows = len(self.unique_rows)
        if self.unique_cols is not None:
            ncols = len(self.unique_cols) 

        if self.wrap:
            if self.plot_type == 'row_only':
                ncols = (nrows - 1) // self.wrap + 1
                nrows = min(nrows, self.wrap)
            elif self.plot_type == 'col_only':
                nrows = (ncols - 1) // self.wrap + 1
                ncols = min(ncols, self.wrap)
        return nrows, ncols

    def get_data_for_every_plot(self):
        rows, cols = self.unique_rows, self.unique_cols
        if self.plot_type == 'row_only':
            return [(row, self.data.loc[row]) for row in rows]
        if self.plot_type in ('row_only', 'col_only'):
            return [(col, self.data.loc[col]) for col in cols]
        elif self.plot_type == 'square':
            return [((row, col), self.data.loc[(row, col)]) for row in rows for col in cols]
        else:
            return [(None, self.data)]

    def get_labels(self, labels):
        if isinstance(labels, tuple):
            return labels
        elif labels is None:
            return None, None
        elif self.plot_type == 'row_only':
            return labels, None
        else:
            return None, labels

    def get_figsize(self, figsize):
        if figsize:
            return figsize
        else:
            return self.fig_shape[1] * 4, self.fig_shape[0] * 3

    def create_figure(self):
        fig, axs = plt.subplots(*self.fig_shape, tight_layout=True, dpi=144, 
                                figsize=self.figsize, sharex=self.sharex, sharey=self.sharey)
        if self.fig_shape != (1, 1):
            axs = axs.flatten(order='F')
        else:
            axs = [axs]
        return fig, axs

    def set_color_cycle(self):
        for ax in self.axs:
            ax.set_prop_cycle(color=self.colors)

    def get_correct_data_order(self, x, y=None):
        if y is None:
            x, y = x.index.values, x.values
        else:
            x, y = x.values, y.values
        if self.orientation == 'h':
            x, y = y, x
        return x, y

    def get_final_data(self):
        final_data = defaultdict(list)
        for (labels, data), ax in zip(self.data_for_plots, self.axs):
            row_label, col_label = self.get_labels(labels)
            if self.split:
                for grp, data_grp in data.groupby(self.split):
                    if self.aggfunc == '__ignore__':
                        # no aggregation - splitting data into groups (for distribution plots)
                        column_data = []
                        labels = []
                        for grp_grp, data_grp_grp in data_grp.groupby(self.groupby):
                            column_data.append(data_grp_grp[self.agg])
                            labels.append(grp_grp)
                        x, y = column_data, labels
                    elif self.agg:
                        s = data_grp.groupby(self.groupby)[self.agg].agg(self.aggfunc)
                        x, y = self.get_correct_data_order(s)
                    else:
                        x, y = self.get_correct_data_order(data_grp[self.x], data_grp[self.y])
                    final_data[ax].append((x, y, grp, row_label, col_label))
            elif self.aggfunc == '__ignore__':
                # no aggregation - splitting data into groups (for distribution plots)
                column_data = []
                labels = []
                for grp, data_grp in data.groupby(self.groupby):
                    column_data.append(data_grp[self.agg])
                    labels.append(grp)
                x, y = column_data, labels
                final_data[ax].append((x, y, None, row_label, col_label))
            elif self.agg:
                s = data.groupby(self.groupby)[self.agg].agg(self.aggfunc)
                x, y = self.get_correct_data_order(s)
                final_data[ax].append((x, y, None, row_label, col_label))
            else:
                x, y = self.get_correct_data_order(data[self.x], data[self.y])
                final_data[ax].append((x, y, ax, None, row_label, col_label))
        return final_data

    def style_fig(self):
        for ax in self.axs:
            ax.tick_params(length=0)
            ax.set_facecolor('.9')
            ax.grid(True)
            ax.set_axisbelow(True)
            for spine in ax.spines.values():
                spine.set_visible(False)

    def add_x_y_labels(self):
        self.fig.text(0, .5, self.y, rotation=90, ha='center', va='center', size='larger')
        self.fig.text(.5, 0, self.x, ha='center', va='center', size='larger')

    def add_ax_titles(self):
        for ax, info in self.final_data.items():
            x, y, label, row_label, col_label = info[0]
            row_label = row_label or ''
            col_label = col_label or ''
            if row_label and col_label:
                title = row_label + ' - ' + col_label
            else:
                title = row_label or col_label
            title = textwrap.fill(title, 30)
            ax.set_title(title)


def line(x, y, data, groupby=None, aggfunc=None, split=None, row=None, col=None, 
         x_order=None, y_order=None, split_order=None, row_order=None, col_order=None,
         orientation='v', sort=False, wrap=None, figsize=None, title=None, sharex=True, 
         sharey=True, xlabel=None, ylabel=None, xlim=None, ylim=None, xscale='linear', 
         yscale='linear', cmap=None):

        self = CommonPlot(x, y, data, groupby, aggfunc, split, row, col, 
                          x_order, y_order, split_order, row_order, col_order,
                          orientation, sort, wrap, figsize, title, sharex, 
                          sharey, xlabel, ylabel, xlim, ylim, xscale, yscale, cmap)

        if self.agg_kind == 'O':
            raise ValueError('Cannot do line plot when the aggregating '
                             'variable is string/categorical')

        for ax, info in self.final_data.items():
            for x, y, label, row_label, col_label in info:
                x_plot, y_plot = x, y
                if x_plot.dtype.kind == 'O':
                    x_plot = np.arange(len(x_plot))
                if y_plot.dtype.kind == 'O':
                    y_plot = np.arange(len(y_plot))
                
                ax.plot(x_plot, y_plot, label=label)
                    
            if x.dtype.kind == 'O':
                x_num = np.arange(len(x))
                categories = [textwrap.fill(cat, 10) for cat in x]
                ax.set_xticks(x_num)
                ax.set_xticklabels(categories)
            if y.dtype.kind == 'O':
                y_num = np.arange(len(y))
                categories = [textwrap.fill(cat, 10) for cat in y]
                ax.set_yticks(y_num)
                ax.set_yticklabels(categories)
                
        if self.split:
            handles, labels = self.axs[0].get_legend_handles_labels()
            self.fig.legend(handles, labels, loc='center left', bbox_to_anchor=(1, .6))
        return self.fig


def scatter(x, y, data, groupby=None, aggfunc=None, split=None, row=None, col=None, 
            x_order=None, y_order=None, split_order=None, row_order=None, col_order=None,
            orientation='v', sort=False, wrap=None, figsize=None, title=None, sharex=True, 
            sharey=True, xlabel=None, ylabel=None, xlim=None, ylim=None, xscale='linear', 
            yscale='linear', cmap=None, regression=False):

        self = CommonPlot(x, y, data, groupby, aggfunc, split, row, col, 
                          x_order, y_order, split_order, row_order, col_order,
                          orientation, sort, wrap, figsize, title, sharex, 
                          sharey, xlabel, ylabel, xlim, ylim, xscale, yscale, cmap)

        if self.agg_kind == 'O':
            raise ValueError('Cannot do line plot when the aggregating '
                             'variable is string/categorical')

        for ax, info in self.final_data.items():
            for x, y, label, row_label, col_label in info:
                if x.dtype.kind == 'O':
                    categories = [textwrap.fill(cat, 10) for cat in x]
                    x = np.arange(len(x))
                    d = dict(zip(categories, x))
                    formatter = category.StrCategoryFormatter(d)
                    ax.xaxis.set_major_formatter(formatter)
                    
                ax.scatter(x, y, label=label, alpha=.7)
                if regression:
                    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
                    x_line = np.array([x.min(), x.max()])
                    y_line = x_line * slope + intercept
                    ax.plot(x_line, y_line)

        if self.split:
            handles, labels = self.axs[0].get_legend_handles_labels()
            self.fig.legend(handles, labels, loc='center left', bbox_to_anchor=(1, .6))
        return self.fig


def bar(x, y, data, groupby=None, aggfunc=None, split=None, row=None, col=None, 
        x_order=None, y_order=None, split_order=None, row_order=None, col_order=None,
        orientation='v', sort=False, wrap=None, figsize=None, title=None, sharex=True, 
        sharey=True, xlabel=None, ylabel=None, xlim=None, ylim=None, xscale='linear', 
        yscale='linear', cmap=None, size=.92):

        self = CommonPlot(x, y, data, groupby, aggfunc, split, row, col, 
                          x_order, y_order, split_order, row_order, col_order,
                          orientation, sort, wrap, figsize, title, sharex, 
                          sharey, xlabel, ylabel, xlim, ylim, xscale, yscale, cmap)

        if self.agg_kind == 'O':
            raise ValueError('Cannot do line plot when the aggregating '
                             'variable is string/categorical')

        
        for ax, info in self.final_data.items():
            cur_size = size / len(info)
            for i, (x, y, label, row_label, col_label) in enumerate(info):
                if self.orientation == 'v' and x.dtype.kind == 'O':
                    x = np.arange(len(x)) + cur_size * i
                elif self.orientation == 'h' and y.dtype.kind == 'O':
                    y = np.arange(len(y)) + cur_size * i

                if len(x) > 200:
                    warnings.warn('You are plotting more than 200 bars. Did you forget to use groupby and aggfunc?')

                if self.orientation == 'v':
                    ax.bar(x, y, label=label, width=cur_size, align='edge')
                else:
                    ax.barh(y, x, label=label, height=cur_size, align='edge')

        x, y = self.final_data[self.fig.axes[0]][0][:2]
        ncols = self.fig_shape[1]
        if self.orientation == 'v':
            categories = [textwrap.fill(cat, 10) for cat in x]
            x = np.arange(len(x)) + size / 2
            d = dict(zip(categories, x))
            for ax in self.fig.axes[-ncols:]:
                ax.set_xticks(x)
                ax.set_xticklabels(categories)
        else:
            for ax in self.fig.axes[::ncols]:
                ax.set_yticks(np.arange(len(y)) + size / 2)
                ax.set_yticklabels(y)

        if self.split:
            handles, labels = self.axs[0].get_legend_handles_labels()
            self.fig.legend(handles, labels, loc='center left', bbox_to_anchor=(1, .6))
        plt.rcParams['font.size'] = self.orig_fontsize
        return self.fig


def box(x, y, data, split=None, row=None, col=None, x_order=None, y_order=None, 
        split_order=None, row_order=None, col_order=None, orientation='h', sort=False, 
        wrap=None, figsize=None, title=None, sharex=True, sharey=True, xlabel=None, 
        ylabel=None, xlim=None, ylim=None, xscale='linear', yscale='linear', cmap=None, 
        notch=None, sym=None, whis=None, positions=None, widths=None, patch_artist=None,
        bootstrap=None, usermedians=None, conf_intervals=None, meanline=None, showmeans=None, 
        showcaps=None, showbox=None, showfliers=None, boxprops=None, labels=None, flierprops=None,
        medianprops=None, meanprops=None, capprops=None, whiskerprops=None, manage_ticks=True,
        autorange=False, zorder=None):

        groupby = y if orientation == 'h' else x
        aggfunc = '__ignore__'

        self = CommonPlot(x, y, data, groupby, aggfunc, split, row, col, 
                          x_order, y_order, split_order, row_order, col_order,
                          orientation, sort, wrap, figsize, title, sharex, 
                          sharey, xlabel, ylabel, xlim, ylim, xscale, yscale, cmap)

        vert = self.orientation == 'v'
        for ax, info in self.final_data.items():
            handles = []
            labels = []
            widths = .9 / len(info)
            for i, (x, y, label, row_label, col_label) in enumerate(info):
                positions = np.arange(len(x)) + i * widths
                box = ax.boxplot(x, vert=vert, positions=positions, widths=widths, 
                                 patch_artist=True, 
                                 boxprops={'facecolor': self.colors[i] ,'edgecolor': 'black'},
                                 medianprops={'color': '.2'}, 
                                 flierprops={'markersize': np.sqrt(widths) * 6})
                handles.append(box['boxes'][0])
                labels.append(label)
            if vert:
                ax.set_xticks(positions // 1 + .45)
                ax.set_xticklabels(y)
            else:
                ax.set_yticks(positions // 1 + .45)
                ax.set_yticklabels(y)

        if self.split:
            self.fig.legend(handles, labels, loc='upper left', bbox_to_anchor=(1, .8))
        return self.fig

        
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