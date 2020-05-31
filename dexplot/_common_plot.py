import textwrap
from collections import defaultdict

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from . import _utils

NONETYPE = type(None)
_NORMALIZE_ERROR_MSG = '`normalize` can only be None, "all", one of the values passed to ' \
                      ' the parameter names "agg", "split", "row", "col", or a combination' \
                      ' of those parameter names in a tuple, if they are defined.'

class CommonPlot:

    def __init__(self, x, y, data, groupby, aggfunc, split, row, col, 
                 orientation, sort, wrap, figsize, title, sharex, sharey, 
                 xlabel, ylabel, xlim, ylim, xscale, yscale):

        self.groups = []
        self.data = self.get_data(data)
        self.x = self.get_col(x)
        self.y = self.get_col(y)
        self.validate_x_y()
        self.groupby = self.get_col_data(groupby, True)
        self.aggfunc = aggfunc
        self.split = self.get_col_data(split, True)
        self.row = self.get_col_data(row, True)
        self.col = self.get_col_data(col, True)
        self.orientation = orientation
        self.agg = self.set_agg()
        self.has_agg = self.agg is not None
        self.has_split = self.split is not None
        self.has_row = self.row is not None
        self.has_col = self.col is not None
        
        self.sort = sort
        self.wrap = wrap
        self.figsize = figsize or plt.rcParams['figure.figsize']
        self.title = title
        self.sharex = sharex
        self.sharey = sharey
        self.xlabel = xlabel
        self.ylabel = ylabel
        self.xlim = xlim
        self.ylim = ylim
        self.xscale = xscale
        self.yscale = yscale

        self.validate_args()
        self.plot_type = self.get_plot_type()
        self.agg_kind = self.get_agg_kind()
        self.data = self.set_index()
        self.fig_shape = self.get_fig_shape()
        self.data_for_plots = self.get_data_for_every_plot()
        self.grouped = self.get_grouped()
        self.fig, self.axs = self.create_figure()
        self.no_legend = True

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
        if self.x = self.y
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
        
    def validate_args(self):
        self.validate_figsize()
        self.validate_plot_args()
        self.validate_mpl_args()

    def validate_figsize(self):
        if isinstance(self.figsize, (list, tuple)):
            if len(self.figsize) != 2:
                raise ValueError('figsize must be a two-item tuple/list')
            for val in self.figsize:
                if not isinstance(val, (int, float)):
                    raise ValueError('Each item in figsize must be an integer or a float')
        else:
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
        if self.has_agg:
            # string and category use 'O'
            agg_kind = self.agg.dtype.kind

            if agg_kind not in ['i', 'f', 'b', 'O']:
                raise TypeError('The data type for the `agg` column must either be boolean, integer, '
                                f'float, or categorical/object/string and not {agg_data.dtype}')
            return agg_kind

    def set_index(self):
        data = self.data
        rc = []
        if self.has_row:
            rc.append(self.row)
        if self.has_col:
            rc.append(self.col)
        if rc:
            data = data.set_index(rc)
        return data

    def get_fig_shape(self):
        if self.plot_type == 'single':
            return 1, 1

        nrows = len(self.data.index.levels[0])
        ncols = len(self.data.index.levels[-1])
        if self.plot_type == 'row_only':
            ncols = 1
            if self.wrap:
                nrows = min(nrows, self.wrap)
                ncols = (nrows - 1) // self.wrap + 1
        elif self.plot_type == 'col_only':
            nrows = 1
            if self.wrap:
                ncols = min(ncols, self.wrap)
                nrows = (ncols - 1) // self.wrap + 1
        return nrows, ncols

    def get_data_for_every_plot(self):
        vals = self.data.index.levels[0]
        if self.plot_type in ('row_only', 'col_only'):
            return [(val, self.data.loc[val]) for val in vals]
        elif self.plot_type == 'square':
            rows, cols = vals, self.data.index.levels[1]
            return [((row, col), self.data.loc[(row, col)]) for row in rows for col in cols]
        else:
            return [(None, self.data)]

    def create_figure(self):
        fig, ax_array = plt.subplots(*self.fig_shape, tight_layout=True, dpi=144)
        axs = ax_array.flatten(order='F')
        return fig, axs


def line(x, y, data, groupby=None, aggfunc=None, split=None, row=None, col=None, 
         orientation='v', sort=False, wrap=None, figsize=None, title=None, sharex=True, 
         sharey=True, xlabel=None, ylabel=None, xlim=None, ylim=None, xscale='linear', 
         yscale='linear'):

        self = CommonPlot(x, y, data, groupby, aggfunc, split, row, col, 
                          orientation, sort, wrap, figsize, title, sharex, 
                          sharey, xlabel, ylabel, xlim, ylim, xscale, yscale)

        if self.agg_kind == 'O':
            raise ValueError('Cannot do line plot when the aggregating '
                             'variable is string/categorical')

        
        for (labels, data), ax in zip(self.data_for_plots, axs):
            if self.has_split:
                for grp, data_grp in data.groupby(self.split):
                    if self.has_agg:
                        s = data_grp.groupby(self.groupby).agg(self.aggfunc)
                        x, y = s.index, s.values
                    else:
                        x, y = data_grp[self.x], data_grp[self.y]
                    ax.plot(x, y)
            elif self.has_agg:
                if self.has_agg:
                    s = data.groupby(self.groupby).agg(self.aggfunc)
                    x, y = s.index, s.values
                else:
                    x, y = data[self.x], data[self.y]
                ax.plot(x, y)


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