import textwrap
import warnings
from collections import defaultdict
import io

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import ticker
from matplotlib.colors import Colormap
from scipy import stats


NONETYPE = type(None)

class CommonPlot:


    def __init__(self, x, y, data, aggfunc, split, row, col, 
                 x_order, y_order, split_order, row_order, col_order,
                 orientation, sort, wrap, figsize, title, sharex, sharey, 
                 xlabel, ylabel, xlim, ylim, xscale, yscale, cmap, 
                 x_textwrap, y_textwrap):

        self.groups = []
        self.data = self.get_data(data)
        self.x = self.get_col(x)
        self.y = self.get_col(y)
        self.validate_x_y()
        self.orientation = orientation
        self.aggfunc = aggfunc
        self.groupby = self.get_col(self.get_groupby(), True)
        self.split = self.get_col(split, True)
        self.row = self.get_col(row, True)
        self.col = self.get_col(col, True)
        
        self.agg = self.set_agg()
        self.make_groups_categorical()
        
        self.x_order = x_order
        self.y_order = y_order
        self.split_order = split_order
        self.row_order = row_order
        self.col_order = col_order
        
        self.sort = sort
        self.groupby_sort = self.sort is not None
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
        self.x_textwrap = x_textwrap
        self.y_textwrap = y_textwrap

        self.validate_args(figsize)
        self.plot_type = self.get_plot_type()
        self.agg_kind = self.get_agg_kind()
        self.data = self.set_index()
        self.rows, self.cols = self.get_uniques()
        self.rows, self.cols = self.get_row_col_order()
        self.fig_shape = self.get_fig_shape()
        self.user_figsize = figsize is not None
        self.figsize = self.get_figsize(figsize)
        self.original_rcParams = plt.rcParams.copy()
        self.set_rcParams()
        self.fig, self.axs = self.create_figure()
        self.set_color_cycle()
        self.data_for_plots = self.get_data_for_every_plot()
        self.final_data = self.get_final_data()
        self.style_fig()
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
        if self.x == self.y and self.x is not None and self.y is not None:
            raise ValueError('`x` and `y` cannot be the same column name')

    def get_groupby(self):
        if self.x is None or self.y is None or self.aggfunc is None:
            return
        return self.x if self.orientation == 'v' else self.y

    def set_agg(self):
        return self.y if self.orientation == 'v' else self.x

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
        self.validate_sort()

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

    def validate_sort(self):
        if self.sort not in ['lex_asc', 'lex_desc', 'asc', 'desc', None]:
            raise ValueError('`sort` must be one of "lex_asc", "lex_desc", "asc", "desc", or `None`')

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

    def get_row_col_order(self):
        rows, cols = self.rows, self.cols
        if rows is not None:
            rows = sorted(rows)
        if cols is not None:
            cols = sorted(cols)

        if self.row_order:
            new_rows = []
            for row in self.row_order:
                if row not in rows:
                    raise ValueError(f'Row value {row} does not exist')
                new_rows.append(row)
            rows = new_rows
        if self.col_order:
            new_cols = []
            for col in self.col_order:
                if col not in cols:
                    raise ValueError(f'Column value {col} does not exist')
                new_cols.append(col)
            cols = new_cols
        return rows, cols
        
    def get_fig_shape(self):
        if self.plot_type == 'single':
            return 1, 1

        nrows = ncols = 1
        if self.rows is not None:
            nrows = len(self.rows)
        if self.cols is not None:
            ncols = len(self.cols) 

        if self.wrap:
            if self.plot_type == 'row_only':
                ncols = (nrows - 1) // self.wrap + 1
                nrows = min(nrows, self.wrap)
            elif self.plot_type == 'col_only':
                nrows = (ncols - 1) // self.wrap + 1
                ncols = min(ncols, self.wrap)
        return nrows, ncols

    def get_data_for_every_plot(self):
        rows, cols = self.get_row_col_order()
        if self.plot_type == 'row_only':
            return [(row, self.data.loc[row]) for row in rows]
        if self.plot_type in ('row_only', 'col_only'):
            return [(col, self.data.loc[col]) for col in cols]
        elif self.plot_type == 'square':
            groups = []
            for col in cols:
                for row in rows:
                    group = row, col
                    try:
                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore")
                            data = self.data.loc[group]
                    except KeyError:
                        data = self.data.iloc[:0]
                    groups.append((group, data))
            return groups
        else:
            return [(None, self.data)]

    def get_labels(self, labels):
        # this won't work for wrapping
        if self.plot_type == 'square':
            return str(labels[0]), str(labels[1])
        elif self.plot_type == 'row_only':
            return str(labels), None
        elif self.plot_type == 'col_only':
            return None, str(labels)
        return None, None

    def get_figsize(self, figsize):
        if figsize:
            return figsize
        else:
            return self.fig_shape[1] * 4, self.fig_shape[0] * 3

    def create_figure(self):
        fig = plt.Figure(tight_layout=True, dpi=144, figsize=self.figsize)
        axs = fig.subplots(*self.fig_shape, sharex=self.sharex, sharey=self.sharey)
        if self.fig_shape != (1, 1):
            axs = axs.flatten(order='F')
        else:
            axs = [axs]
        return fig, axs

    def set_color_cycle(self):
        for ax in self.axs:
            ax.set_prop_cycle(color=self.colors)

    def sort_xy(self, x, y):
        if self.sort == 'lex_asc' or self.sort is None:
            return x, y
        elif self.sort == 'lex_desc':
            order = np.lexsort([y, x])[::-1]
        elif self.sort == 'asc':
            order = np.lexsort([x, y])
        else:
            order = np.lexsort([x, -y])
        return x[order], y[order]

    def get_order(self, arr, vals):
        arr = arr.tolist()
        order = []
        for val in vals:
            try:
                idx = arr.index(val)
            except ValueError:
                raise ValueError(f'{val} is not a valid column value')
            order.append(idx)
        return order

    def order_xy(self, x, y):
        if self.x_order:
            order = self.get_order(x, self.x_order)
            x = x[order]
            y = y[order]
        elif self.y_order:
            order = self.get_order(y, self.y_order)
            x = x[order]
            y = y[order]
        return x, y

    def get_correct_data_order(self, x, y=None):
        if y is None:
            x, y = x.index.values, x.values
        else:
            x, y = x.values, y.values

        x, y = self.sort_xy(x, y)
        x, y = self.order_xy(x, y)
        if self.orientation == 'h':
            x, y = y, x
        return x, y

    def get_wide_data(self, data):
        x = data.index.values
        y = {col: data[col].values for col in data.columns}
        if self.orientation == 'h':
            x, y = y, x
        return x, y

    def get_wide_columns(self, data):
        cols = []
        used_cols = [self.groupby, self.split, self.row, self.col]
        for col in data.columns:
            if col not in used_cols:
                cols.append(col)
        return cols

    def split_groups(self, data):
        order = []
        groups = []
        for grp, data_grp in data.groupby(self.split, sort=self.groupby_sort):
            order.append((grp, data_grp))
            groups.append(grp)

        if self.split_order:
            new_order = []
            for split in self.split_order:
                try:
                    idx = groups.index(split)
                except ValueError:
                    raise ValueError(f'Value {split} from `split_order` is '
                                     'not in column {self.split}')
        
                new_order.append(idx)
            order = [order[i] for i in new_order]
        return order


    def get_final_groups(self, data, split_label, row_label, col_label):
        groups = []
        if self.groupby is not None:
            if self.aggfunc == '__distribution__':
                for grp, data_grp in data.groupby(self.groupby, sort=self.groupby_sort):
                    x, y = self.get_correct_data_order(data_grp[self.agg])
                    groups.append((x, y, split_label, grp, row_label, col_label))
            else:
                s = data.groupby(self.groupby, sort=self.groupby_sort)[self.agg].agg(self.aggfunc)
                x, y = self.get_correct_data_order(s)
                groups.append((x, y, split_label, None, row_label, col_label))
        elif self.x is None or self.y is None:
            if self.x:
                x, y = self.get_correct_data_order(data[self.x])
                groups.append((x, y, split_label, None, row_label, col_label))
            elif self.y:
                x, y = self.get_correct_data_order(data[self.y])
                groups.append((x, y, split_label, None, row_label, col_label))
            else:
                # wide data
                for col in self.get_wide_columns(data):
                    x, y = self.get_correct_data_order(data[col])
                    groups.append((x, y, split_label, col, row_label, col_label))
        else:
            # simple raw plot - make sure to warn when lots of data for bar/box/hist
            # one graph per row - OK for scatterplots and line plots
            x, y = self.get_correct_data_order(data[self.x], data[self.y])
            groups.append((x, y, None, None, row_label, col_label))
        return groups

    def get_final_data(self):
        # create list of data for each call to plotting method
        final_data = defaultdict(list)
        for (labels, data), ax in zip(self.data_for_plots, self.axs):
            row_label, col_label = self.get_labels(labels)
            if self.split:
                for grp, data_grp in self.split_groups(data):
                    final_data[ax].extend(self.get_final_groups(data_grp, grp, row_label, col_label))
            else:
                final_data[ax].extend(self.get_final_groups(data, None, row_label, col_label))
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
        if self.plot_type == 'single':
            self.axs[0].set_xlabel(self.x)
            self.axs[0].set_ylabel(self.y)
            return
        
        # need to eliminate next line to save lots of time
        self.fig.canvas.print_figure(io.BytesIO())
        rows, cols = self.fig_shape
        top_left_ax, bottom_right_ax = self.axs[0], self.axs[rows * cols - 1]
        top_left_points = top_left_ax.get_position().get_points()
        bottom_right_points = bottom_right_ax.get_position().get_points()

        left = top_left_points[0][0]
        right = bottom_right_points[1][0]
        x = (right + left) / 2

        top = top_left_points[1][1]
        bottom = bottom_right_points[0][1]
        y = (top + bottom) / 2
        self.fig.text(0, y, self.y, rotation=90, ha='center', va='center', size='larger')
        self.fig.text(x, 0, self.x, ha='center', va='center', size='larger')

    def add_ax_titles(self):
        for ax, info in self.final_data.items():
            row_label, col_label = info[0][-2:]
            if row_label is not None:
                row_label = str(row_label)
            if col_label is not None:
                col_label = str(col_label)
            row_label = row_label or ''
            col_label = col_label or ''
            if row_label and col_label:
                title = row_label + ' - ' + col_label
            else:
                title = row_label or col_label
            title = textwrap.fill(str(title), 30)
            ax.set_title(title)

    def set_rcParams(self):
        plt.rcParams['font.size'] = 6
        plt.rcParams['font.family'] = 'Helvetica'

    def get_x_y_plot(self, x, y):
        x_plot, y_plot = x, y
        if x_plot.dtype.kind == 'O':
            x_plot = np.arange(len(x_plot))
        if y_plot.dtype.kind == 'O':
            y_plot = np.arange(len(y_plot))
        return x_plot, y_plot

    def add_ticklabels(self, x, y, ax, delta=0):
        if x.dtype.kind == 'O':
            x_num = np.arange(len(x)) + delta
            categories = [textwrap.fill(str(cat), self.x_textwrap) for cat in x]
            ax.set_xticks(x_num)
            ax.set_xticklabels(categories)

        if y.dtype.kind == 'O':
            y_num = np.arange(len(y)) + delta
            ax.set_yticks(y_num)
            categories = y
            if self.y_textwrap:
                categories = [textwrap.fill(str(cat), self.y_textwrap) for cat in y]
            ax.set_yticklabels(categories)

    def add_legend(self, handles=None, labels=None):
        if self.split:
            if handles is None:
                handles, labels = self.axs[0].get_legend_handles_labels()
            ncol = len(labels) // 8 + 1
            self.fig.legend(handles, labels, loc='upper left', bbox_to_anchor=(1.01, .8), 
                            title=self.split, ncol=ncol)

    def clean_up(self):
        self.add_x_y_labels()
        plt.rcParams = self.original_rcParams
        return self.fig

    def update_fig_size(self, n_splits, n_groups_per_split):
        if self.user_figsize:
            return
        new_size = 1.5 + (.3 + .06 * n_splits) * n_groups_per_split
        if self.orientation == 'v':
            height = max(2.5 - .3 * self.fig_shape[0], 1.2)
            shrink = max(.9 - .1 * self.fig_shape[1], .5)
            width = new_size * shrink * self.fig_shape[1]
            height = height * self.fig_shape[0]
        else:
            width = max(3 - .3 * self.fig_shape[1], 1.5)
            height = new_size * .8 * self.fig_shape[0]
            width = width * self.fig_shape[1]
        width, height = min(width, 25), min(height, 25)
        self.fig.set_size_inches(width, height)