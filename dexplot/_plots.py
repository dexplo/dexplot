import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import summary_table
import textwrap
from scipy.stats import gaussian_kde

NORMALIZE_ERROR_MSG = '`normalize` can only be None, "all", one of the parameter names "agg", ' \
                      '"hue", "row", "col", or a combination of those parameter names ' \
                      'in a tuple, if they are defined.'


def _get_fig_shape(n, wrap, is_row):
    if is_row:
        if wrap is None:
            return n, 1
        else:
            return wrap, int(np.ceil(n / wrap))
    else:
        if wrap is None:
            return 1, n
        else:
            return int(np.ceil(n / wrap)), wrap


def _map_val_to_color(vals):
    n = len(vals)
    if n <= 10:
        colors = plt.cm.tab10(range(n))
    elif n <= 20:
        colors = plt.cm.tab20(range(n))
    else:
        colors = plt.cm.viridis(n)
    return dict(zip(vals, colors))


def _calculate_density(data):
    density_func = gaussian_kde(data)
    min_x, max_x = data.min(), data.max()
    range_x = max_x - min_x
    min_x = min_x - 2 * range_x
    max_x = max_x + 2 * range_x
    x = np.linspace(min_x, max_x, 400)
    density = density_func(x)
    max_density = density.max()
    filt = density > max_density / 1000
    x = x[filt]
    density = density[filt]
    return x, density

# TODO: Do normalization for numeric variables
# TODO: Add counts (number of obs) in bars
# TODO: Automatically wrap ticklabels instead of rotating them

class CommonPlotter:

    def create_figure(self):
        if not (self.row or self.col):
            if self.orig_figsize is None:
                self.figsize = (12, 6)
                self.nrows, self.ncols = 1, 1
            return plt.subplots(figsize=self.figsize)
        if bool(self.row) != bool(self.col):
            split_by = self.row or self.col
            num_unique = len(self.data[split_by].unique())
            if self.wrap is None:
                dim1 = num_unique
                dim2 = 1
            else:
                dim1 = min(num_unique, self.wrap)
                dim2 = (num_unique - 1) // dim1 + 1

            nrows, ncols = (dim1, dim2) if self.row else (dim2, dim1)
        else:
            nrows = len(self.data[self.row].unique())
            ncols = len(self.data[self.col].unique())

        if self.orig_figsize is None:
            self.figsize = _calculate_figsize(nrows, ncols)

        self.nrows, self.ncols = nrows, ncols
        return plt.subplots(nrows, ncols, figsize=self.figsize)


class AggPlot:

    def __init__(self, agg, groupby, data, hue, row, col, kind, orient, sort, aggfunc, normalize,
                 wrap, figsize, rot, title, sharex, sharey, xlabel, ylabel, xlim, ylim,
                 xscale, yscale, kwargs):
        self.col_params = ['agg', 'groupby', 'hue', 'row', 'col']
        self.validate_figsize(figsize)
        self.validate_data(data)
        self.validate_column_names(agg, groupby, hue, row, col)
        self.validate_orient_wrap_kind_sort(orient, wrap, kind, sort)
        self.validate_agg_kind()
        self.init_column_data()
        self.validate_groupby_agg()
        self.validate_diff_col_names()
        self.validate_kde_hist()
        self.validate_normalize(normalize)
        self.validate_kwargs(kwargs)
        self.validate_rot(rot)
        self.validate_mpl_args(title, sharex, sharey, xlabel, ylabel, xlim, ylim, xscale, yscale)
        self.get_unique_hues()
        self.get_unique_groupby()
        self.get_unique_agg()
        self.get_unique_row()
        self.get_unique_col()
        self.normalize_counts = self.get_normalize_counts()
        self.is_single_plot()
        self.plot_func = self.get_plotting_function()
        self.aggfunc = aggfunc
        self.width = .8
        self.no_legend = True
        self.vc_dict = {}

    def validate_figsize(self, figsize):
        self.orig_figsize = figsize
        if figsize is None:
            self.figsize = plt.rcParams['figure.figsize']
        elif isinstance(figsize, tuple):
            if len(figsize) != 2:
                raise ValueError('figsize must be a two-item tuple')
            for val in figsize:
                if not isinstance(val, (int, float)):
                    raise ValueError('Each item in figsize must be an integer or a float')
            self.figsize = figsize
        else:
            raise TypeError('figsize must be a two-item tuple')

    def validate_data(self, data):
        if not isinstance(data, pd.DataFrame):
            raise TypeError('`data` must be a DataFrame')
        elif len(data) == 0:
            raise ValueError('DataFrame contains no  data')
        else:
            self.data = data

    def validate_column_names(self, agg, groupby, hue, row, col):
        param_vals = [agg, groupby, hue, row, col]
        for arg_name, col_name in zip(self.col_params, param_vals):
            if col_name and col_name not in self.data.columns:
                raise KeyError(f'You passed {col_name} to parameter {arg_name} which is not a '
                                 'column name')
            self.__dict__[arg_name] = col_name

    def validate_orient_wrap_kind_sort(self, orient, wrap, kind, sort):

        if orient not in ['v', 'h']:
            raise ValueError('`orient` must be either "v" or "h".')

        if wrap is not None:
            if not isinstance(wrap, int):
                raise TypeError('`wrap` must either be None or an integer. '
                                f'You passed {type(wrap)}')

        if kind not in {'bar', 'line', 'box', 'hist', 'kde'}:
            raise ValueError('`kind` must be either "bar", "line", "box", "hist", or "kde"')

        if not isinstance(sort, bool):
            raise TypeError('`sort` must be a bool')

        self.orient, self.wrap, self.kind, self.sort = orient, wrap, kind, sort

    def validate_agg_kind(self):
        self.agg_data = self.data[self.agg]
        self.agg_kind = self.agg_data.dtype.kind

        if self.agg_kind not in ['i', 'f', 'b', 'O']:
            raise TypeError(f'The data type for the `agg` column must either be boolean, integer, '
                            'float, or categorical/object and not {agg_data.dtype}')

        if self.agg_kind == 'O':
            if self.kind not in ('bar', 'line'):
                raise ValueError('When the `agg` variable is object/categorical, `kind` can '
                                 'only be "bar" or "line"')

    def init_column_data(self):
        if self.groupby is not None:
            self.groupby_data = self.data[self.groupby]
            self.groupby_kind = self.groupby_data.dtype.kind

        if self.hue is not None:
            self.hue_data = self.data[self.hue]
            self.hue_kind = self.hue_data.dtype.kind

        if self.row is not None:
            self.row_data = self.data[self.row]
            self.row_kind = self.row_data.dtype.kind

        if self.col is not None:
            self.col_data = self.data[self.col]
            self.col_kind = self.col_data.dtype.kind

    def validate_groupby_agg(self):
        if self.groupby:
            if self.agg_kind == 'O' and self.groupby_kind == 'O':
                raise TypeError('When the `agg` column is categorical, you cannot use `groupby`. '
                                'Instead, place the groupby column as either '
                                ' `hue`, `row`, or `col`.')

    def validate_diff_col_names(self):
        param_names = ['agg', 'groupby', 'hue', 'row', 'col']
        seen = set()
        for name in param_names:
            val = self.__dict__[name]
            if val is not None:
                if val in seen:
                    raise ValueError(f'Duplicate column found in parameter `{name}` with "{val}". '
                                     'All column names supplied to `agg`, `groupby`, `hue`'
                                     ', `row` and `col` must be unique')
                seen.add(val)

    def validate_kde_hist(self):
        if self.kind in ('hist', 'kde'):
            if self.groupby and self.hue:
                raise NotImplementedError('When plotting a "hist" or "kde", you can set at most '
                                          'one of `groupby` or `hue` but not both')

    def validate_normalize(self, normalize):
        if self.agg_kind == 'O':
            valid_normalize = ['all']
            for cp in ['agg', 'hue', 'row', 'col']:
                if self.__dict__[cp] is not None:
                    valid_normalize.append(cp)
            if isinstance(normalize, str):
                if normalize not in valid_normalize:
                    raise ValueError(NORMALIZE_ERROR_MSG)
            elif isinstance(normalize, tuple):
                if len(normalize) == 1:
                    return self.validate_normalize(normalize[0])
                for val in normalize:
                    if val not in valid_normalize[1:]:
                        raise ValueError(NORMALIZE_ERROR_MSG)
                for val in normalize:
                    if normalize.count(val) > 1:
                        raise ValueError(f'{val} is duplicated in your `normalize` tuple')
            elif normalize is not None:
                raise TypeError(NORMALIZE_ERROR_MSG)

            self.normalize = normalize
        else:
            # TODO: force normalziation for numerics
            self.normalize = False

    def validate_kwargs(self, kwargs):
        if kwargs is None:
            self.kwargs = {}
        elif not isinstance(kwargs, dict):
            raise TypeError('`kwargs` must be `None` or a dict')
        else:
            self.kwargs = kwargs

        if self.kind == 'line':
            if 'lw' not in self.kwargs:
                self.kwargs['lw'] = 3
            if 'marker' not in self.kwargs:
                self.kwargs['marker'] = 'o'

    def validate_rot(self, rot):
        if not isinstance(rot, (int, float)):
            raise ValueError('`rot` must be an int or float')
        else:
            self.rot = rot

    def validate_mpl_args(self, title, sharex, sharey, xlabel, ylabel, xlim, ylim, xscale, yscale):
        NoneType = type(None)
        if not isinstance(title, (NoneType, str)):
            raise TypeError('`title` must be either None or a str')
        if sharex not in [False, True, None, 'row', 'col']:
            raise ValueError('`sharex` must be one of `False`, `True`, `None`, "row", or "col"')
        if sharey not in [False, True, None, 'row', 'col']:
            raise ValueError('`sharex` must be one of `False`, `True`, `None`, "row", or "col"')

        if not isinstance(xlabel, (NoneType, str)):
            raise TypeError('`xlabel` must be either None or a str')
        elif xlabel is None:
            xlabel = ''
        if not isinstance(ylabel, (NoneType, str)):
            raise TypeError('`ylabel` must be either None or a str')
        elif ylabel is None:
            ylabel = ''

        if not isinstance(xlim, (NoneType, tuple)):
            raise TypeError('`xlim` must be a two-item tuple of numerics or `None`')
        if not isinstance(ylim, (NoneType, tuple)):
            raise TypeError('`xlim` must be a two-item tuple of numerics or `None`')
        if xscale not in {'linear', 'log', 'symlog', 'logit'}:
            raise ValueError("`xscale must be one of 'linear', 'log', 'symlog', 'logit'")
        if yscale not in {'linear', 'log', 'symlog', 'logit'}:
            raise ValueError("`xscale must be one of 'linear', 'log', 'symlog', 'logit'")
        self.title = title
        self.sharex = sharex
        self.sharey = sharey
        self.xlabel = xlabel
        self.ylabel = ylabel
        self.xlim = xlim
        self.ylim = ylim
        self.xscale = xscale
        self.yscale = yscale

    def get_unique_hues(self):
        if self.hue:
            self.all_hues = np.sort(self.data[self.hue].unique())

    def get_unique_groupby(self):
        if self.groupby:
            self.all_groups = np.sort(self.data[self.groupby].unique())

    def get_unique_agg(self):
        if self.agg_kind == 'O':
            self.all_aggs = np.sort(self.data[self.agg].unique())

    def get_unique_row(self):
        if self.row:
            self.all_rows = np.sort(self.data[self.row].unique())

    def get_unique_col(self):
        if self.col:
            self.all_cols = np.sort(self.data[self.col].unique())


    def get_normalize_counts(self):
        if self.agg_kind != 'O' or not self.normalize:
            return None
        if self.normalize == 'all':
            return self.data[self.agg].count()
        if self.normalize == 'agg':
            return self.data[self.agg].value_counts() \
                       .rename_axis(self.agg).rename(None).reset_index()
        if self.normalize == 'hue':
            return self.data[self.hue].value_counts()\
                       .rename_axis(self.hue).rename(None).reset_index()
        if self.normalize == 'row':
            return self.data[self.row].value_counts() \
                       .rename_axis(self.row).rename(None).reset_index()
        if self.normalize == 'col':
            return self.data[self.col].value_counts() \
                       .rename_axis(self.col).rename(None).reset_index()
        if isinstance(self.normalize, tuple):
            group_cols = [self.__dict__[col] for col in self.normalize]
            uniques, names = [], []
            for val in self.normalize:
                uniques.append(getattr(self, f'all_{val}s'))
                names.append(self.__dict__[val])

            df = self.data.groupby(group_cols).size()
            mi = pd.MultiIndex.from_product(uniques, names=names)
            return df.reindex(mi).reset_index()

    def is_single_plot(self):
        if not (self.row or self.col):
            self.single_plot = True
        else:
            self.single_plot = False

    def get_plotting_function(self):
        plot_dict = {'bar': self.barplot,
                     'line': self.lineplot,
                     'box': self.boxplot,
                     'hist': self.histplot,
                     'kde': self.kdeplot}
        return plot_dict[self.kind]

    def create_figure(self):
        if not (self.row or self.col):
            if self.orig_figsize is None:
                self.figsize = (12, 6)
                self.nrows, self.ncols = 1, 1
            return plt.subplots(figsize=self.figsize)
        if bool(self.row) != bool(self.col):
            split_by = self.row or self.col
            num_unique = len(self.data[split_by].unique())
            if self.wrap is None:
                dim1 = num_unique
                dim2 = 1
            else:
                dim1 = min(num_unique, self.wrap)
                dim2 = (num_unique - 1) // dim1 + 1

            nrows, ncols = (dim1, dim2) if self.row else (dim2, dim1)
        else:
            nrows = len(self.data[self.row].unique())
            ncols = len(self.data[self.col].unique())

        if self.orig_figsize is None:
            self.figsize = _calculate_figsize(nrows, ncols)

        self.nrows, self.ncols = nrows, ncols
        return plt.subplots(nrows, ncols, figsize=self.figsize)

    def set_single_plot_labels(self, ax):
        if self.kind in ('bar', 'line', 'box'):
            if self.orient == 'v' and self.agg_kind != 'O':
                ax.set_xlabel(self.xlabel)
                ax.set_ylabel(self.ylabel or self.agg)
                if not (self.groupby or self.hue):
                    ax.set_xticklabels([])
                    ax.set_xticks([])
                else:
                    ax.tick_params(axis='x', labelrotation=self.rot)
            elif self.orient == 'h' and self.agg_kind != 'O':
                if not (self.groupby or self.hue or self.normalize):
                    ax.set_yticklabels([])
                    ax.set_yticks([])
                    ax.set_xlabel(self.xlabel or self.agg)
                else:
                    ax.set_xlabel(self.xlabel or self.agg)
            elif self.orient == 'v' and self.agg_kind == 'O':
                pass
                # ax.tick_params(axis='x', labelrotation=self.rot)

            if self.groupby and self.hue and self.no_legend:
                ax.legend()
            if self.agg_kind == 'O' and self.hue and self.no_legend:
                ax.legend()
        else:
            if self.orient == 'v':
                ax.set_xlabel(self.xlabel or self.agg)
            else:
                ax.set_ylabel(self.ylabel or self.agg)
            if (self.groupby or self.hue) and self.no_legend:
                ax.legend()

    def set_figure_plot_labels(self, fig):
        label_fontsize = plt.rcParams['font.size'] * 1.5
        only_agg = not (self.groupby or self.hue)
        is_vert = self.orient == 'v'
        is_blb = self.kind in ('bar', 'line', 'box')
        is_hk = self.kind in ('hist', 'kde')
        is_numeric = self.agg_kind != 'O'
        is_numeric_split = self.hue and self.groupby and is_numeric
        is_numeric_split_hue = (self.hue or self.groupby) and is_numeric and is_hk
        is_cat_split = self.hue and not is_numeric
        for ax in fig.axes:
            if is_vert and only_agg and is_blb and is_numeric:
                ax.set_xticks([])
            if not is_vert and only_agg and is_blb and is_numeric:
                ax.set_yticks([])

        if is_numeric:
            if is_blb and is_vert:
                fig.text(-.02, .5, self.agg, rotation=90, fontsize=label_fontsize,
                         ha='center', va='center')
            elif is_blb and not is_vert:
                fig.text(.5, -0.01, self.agg, fontsize=label_fontsize,
                         ha='center', va='center')
            elif not is_blb and is_vert:
                fig.text(.5, -0.01, self.agg, fontsize=label_fontsize,
                         ha='center', va='center')
            else:
                fig.text(-.02, .5, self.agg, rotation=90, fontsize=label_fontsize,
                         ha='center', va='center')
        if (is_numeric_split or is_cat_split or is_numeric_split_hue) and not fig.legends:
            handles, labels = fig.axes[-1].get_legend_handles_labels()
            fig.legend(handles, labels, bbox_to_anchor=(1.04, .5), loc='center left')

    def apply_single_plot_changes(self, ax):
        if self.title:
            ax.figure.suptitle(self.title, y=1.02)

        self.set_single_plot_labels(ax)

        if self.xlim:
            ax.set_xlim(self.xlim)
        if self.ylim:
            ax.set_ylim(self.ylim)
        if self.xscale != 'linear':
            ax.set_xscale(self.xscale)
        if self.yscale != 'linear':
            ax.set_yscale(self.yscale)

    def barplot(self, ax, data, **kwargs):
        n_rows, n_cols = data.shape
        width = self.width / n_cols
        bar_start = (n_cols - 1) / 2 * width
        x_range = np.arange(n_rows)
        for i, (height, col) in enumerate(zip(data.values.T, data.columns)):
            x_data = x_range - bar_start + i * width
            if self.orient == 'v':
                ax.bar(x_data, height, width, label=col, tick_label=data.index)
            else:
                ax.barh(x_data, height, width, label=col, tick_label=data.index)
        if self.orient == 'v':
            ax.set_xticks(x_range)
        else:
            ax.set_yticks(x_range)

    def lineplot(self, ax, data, **kwargs):
        index = data.index
        for i, (height, col) in enumerate(zip(data.values.T, data.columns)):
            if self.orient == 'v':
                ax.plot(index, height, label=col, **self.kwargs)
            else:
                ax.plot(height, index, label=col, **self.kwargs)

    def boxplot(self, ax, data, **kwargs):
        vert = self.orient == 'v'
        if 'boxprops' not in kwargs:
            kwargs['boxprops'] = {'facecolor': plt.cm.tab10(0)}
        if 'medianprops' not in kwargs:
            kwargs['medianprops'] = {'color': 'black'}
        if 'patch_artist' not in kwargs:
            kwargs['patch_artist'] = True
        return ax.boxplot(data, vert=vert, **kwargs)

    def histplot(self, ax, data, **kwargs):
        orientation = 'vertical' if self.orient == 'v' else 'horizontal'
        labels = kwargs['labels']
        return ax.hist(data, orientation=orientation, label=labels, **self.kwargs)

    def kdeplot(self, ax, data, **kwargs):
        labels = kwargs['labels']
        if not isinstance(data, list):
            data = [data]
        for label, cur_data in zip(labels, data):
            if len(cur_data) > 1:
                x, density = _calculate_density(cur_data)
            else:
                x, density = [], []
            if self.orient == 'h':
                x, density = density, x
            ax.plot(x, density, label=label, **self.kwargs)

    def wrap_labels(self, fig):
        renderer = fig.canvas.get_renderer()
        nrows, ncols, *_ = fig.axes[0].get_subplotspec().get_rows_columns()
        ax_width = fig.get_window_extent(renderer).width / ncols
        fig.canvas.draw()
        needs_wrap = False
        for ax in fig.axes:
            xlabels = ax.get_xticklabels()
            num_labels = max(len(xlabels), 1)
            max_width = (ax_width * .8) / num_labels
            new_labels = []
            for label in xlabels:
                text = label.get_text()
                try:
                    float(text)
                    new_labels.append(text)
                    continue
                except ValueError:
                    pass
                len_text = len(text)
                width = label.get_window_extent(renderer).width
                if width > max_width:
                    needs_wrap = True
                    ratio = max_width / width
                    text_width = int(ratio * len_text)
                    wrapper = textwrap.TextWrapper(text_width)
                    new_labels.append('\n'.join(wrapper.wrap(text)))
                else:
                    new_labels.append(text)
            if needs_wrap:
                ax.set_xticklabels(new_labels)

            ylabels = ax.get_yticklabels()
            new_labels = []
            needs_wrap = False
            for label in ylabels:
                text = label.get_text()
                if len(text) > 30:
                    needs_wrap = True
                    wrapper = textwrap.TextWrapper(30)
                    new_labels.append('\n'.join(wrapper.wrap(text)))
                else:
                    new_labels.append(text)
            if needs_wrap:
                ax.set_yticklabels(new_labels)

            # wrap title
            title = ax.title
            text = title.get_text()
            len_title = len(text)
            width = title.get_window_extent(renderer).width
            max_width = ax_width * .8
            if width > max_width:
                ratio = max_width / width
                text_width = int(ratio * len_title)
                wrapper = textwrap.TextWrapper(text_width)
                ax.set_title('\n'.join(wrapper.wrap(text)))

    def align_axes(self, axes):
        def set_lim(cur_axes, axis):
            axes_flat = cur_axes.flatten()
            lims = []
            if axis == 'x':
                get_func, set_func = 'get_xlim', 'set_xlim'
            else:
                get_func, set_func = 'get_ylim', 'set_ylim'
            for ax in axes_flat:
                if ax.lines or ax.patches:
                    lims.append(getattr(ax, get_func)())

            max_lim = max(lim[1] for lim in lims)
            min_lim = min(lim[0] for lim in lims)

            for ax in axes_flat:
                getattr(ax, set_func)(min_lim, max_lim)

        if self.sharex is True:
            set_lim(axes, 'x')
        elif self.sharex == 'row':
            if axes.ndim == 1:
                axes = axes[:, np.newaxis]
            for row in axes:
                set_lim(row, 'x')
        elif self.sharex == 'col':
            if axes.ndim == 1:
                axes = axes[:, np.newaxis]
            for col in axes.T:
                set_lim(col, 'x')

        if self.sharey is True:
            set_lim(axes, 'y')
        elif self.sharey == 'row':
            if axes.ndim == 1:
                axes = axes[:, np.newaxis]
            for row in axes:
                set_lim(row, 'y')
        elif self.sharey == 'col':
            if axes.ndim == 1:
                axes = axes[:, np.newaxis]
            for col in axes.T:
                set_lim(col, 'y')

    def remove_yticklabels(self, axes):
        if self.sharey in ('col', False):
            return
        if axes.ndim == 1:
            axes = axes[:, np.newaxis]
        for ax in axes[:, 1:].flatten():
            if ax is not None:
                n = len(ax.get_yticklabels())
                ax.set_yticklabels([''] * n)

    def remove_xticklabels(self, axes):
        if self.sharex in ('row', False) or self.nrows == 1:
            return
        if axes.ndim == 1:
            axes = axes[:, np.newaxis]
        for ax in axes[:-1].flatten():
            if ax is not None:
                n = len(ax.get_xticklabels())
                ax.set_xticklabels([''] * n)

    def remove_ax(self, axes):
        if self.row and not self.col:
            if self.ncols > 1:
                num_plots = len(self.all_rows)
                left_over = self.nrows * self.ncols - num_plots
                if left_over > 0:
                    ax_flat = axes.flatten('F')
                    good_labels = ax_flat[self.wrap - 1].get_xticklabels()
                    labels = [label.get_text() for label in good_labels]
                    ax_flat[-left_over - 1].set_xticklabels(labels)
                    for ax in ax_flat[-left_over:]:
                        ax.remove()
        elif self.col and not self.row:
            if self.nrows > 1:
                num_plots = len(self.all_cols)
                left_over = self.nrows * self.ncols - num_plots
                if left_over > 0:
                    ax_flat = axes.flatten('C')
                    good_labels = ax_flat[-self.wrap].get_xticklabels()
                    labels = [label.get_text() for label in good_labels]
                    for ax in ax_flat[-self.ncols - left_over: -self.ncols]:
                        ax.set_xticklabels(labels)
                    for ax in ax_flat[-left_over:]:
                        ax.remove()

    def plot(self):
        fig, ax = self.create_figure()
        if not (self.groupby or self.hue or self.row or self.col):
            ax = self.plot_only_agg(ax, self.data)
        elif self.hue and not (self.groupby or self.row or self.col):
            ax = self.plot_hue_agg(ax, self.data)
        elif self.groupby and not (self.hue or self.row or self.col):
            ax = self.plot_groupby_agg(ax, self.data)
        elif self.groupby and self.hue and not (self.row or self.col):
            ax = self.plot_groupby_hue_agg(ax, self.data)
        elif bool(self.row) != bool(self.col):
            self.plot_row_or_col(fig, ax)
        elif self.row and self.col:
            self.plot_row_and_col(fig, ax)

        if self.single_plot:
            self.apply_single_plot_changes(ax)
        else:
            self.set_figure_plot_labels(fig)
            self.align_axes(ax)
            self.remove_yticklabels(ax)
            self.remove_xticklabels(ax)

        self.wrap_labels(fig)
        self.remove_ax(ax)
        fig.tight_layout()

        if self.single_plot:
            return ax
        return fig,

    def do_normalization(self, vc, data=None):
        if not self.normalize:
            return vc
        elif self.normalize == 'all':
            vc.iloc[:, -1] = vc.iloc[:, -1] / self.normalize_counts
            vc = vc.set_index(vc.columns[:-1].tolist())
            return vc

        if isinstance(self.normalize, tuple):
            join_key = [self.__dict__[col] for col in self.normalize]
        else:
            join_key = self.__dict__[self.normalize]

        unique_col_name = "@@@@@count"

        if self.normalize in ('row', 'col'):
            col_name = self.__dict__[self.normalize]
            cur_group = data.iloc[0].loc[col_name]
            df = self.normalize_counts
            cur_count = df[df[col_name] == cur_group].iloc[0, -1]
            vc.iloc[:, -1] = vc.iloc[:, -1] / cur_count
            vc = vc.set_index(vc.columns[:-1].tolist())
            return vc
        elif 'row' in self.normalize or 'col' in self.normalize:
            # self.normalize must be a tuple
            col_names = []
            for val in ('row', 'col'):
                if val in self.normalize:
                    col_names.append(self.__dict__[val])
            cur_groups = [data.iloc[0].loc[col_name] for col_name in col_names]
            df = self.normalize_counts.copy()
            b = df[col_names[0]] == cur_groups[0]
            if len(col_names) == 2:
                b = b & (df[col_names[1]] == cur_groups[1])
            cur_counts = df[b].copy()
            cur_counts.columns = cur_counts.columns.tolist()[:-1] + [unique_col_name]
            join_keys = [self.__dict__[name] for name in self.normalize
                         if name not in ('row', 'col')]
            vc1 = vc.copy()
            vc1.columns = vc1.columns.tolist()[:-1] + [unique_col_name]
            vc1 = vc1.merge(cur_counts, on=join_keys)
            vc.iloc[:, -1] = vc1[unique_col_name + '_x'].values / vc1[unique_col_name + '_y'].values
            vc = vc.set_index(vc.columns[:-1].tolist())
            return vc
        else:
            norms = vc.merge(self.normalize_counts, on=join_key)
            norms['pct'] = norms.iloc[:, -2] / norms.iloc[:, -1]
            norms = norms.drop(columns=norms.columns[[-3, -2]])
            norms = norms.set_index(norms.columns[:-1].tolist())
        return norms

    def plot_only_agg(self, ax, data):
        if self.agg_kind == 'O':
            vc = data.groupby(self.agg).size()
            if self.normalize is not None:
                vc = self.do_normalization(vc.reset_index(), data)
            else:
                vc = vc.to_frame()
            self.plot_func(ax, vc)
        elif self.agg_kind in 'ifb':
            if self.kind in ('box', 'hist', 'kde'):
                self.plot_func(ax, data[self.agg], labels=[self.agg])
            else:
                # For bar and point plots only
                value = data[self.agg].agg(self.aggfunc)
                self.plot_func(ax, pd.DataFrame({self.agg: [value]}))
        return ax

    def plot_hue_agg(self, ax, data):
        if self.agg_kind == 'O':
            tbl = pd.crosstab(data[self.agg], data[self.hue])
            if self.normalize is not None:
                tbl = tbl.stack().reset_index()
                tbl = self.do_normalization(tbl, data)
                tbl = tbl.iloc[:, 0].unstack()
            tbl = tbl.reindex(index=self.all_aggs, columns=self.all_hues)
            self.plot_func(ax, tbl)
        else:
            if self.kind in ('box', 'hist', 'kde'):
                data_array = []
                g = data.groupby(self.hue)
                for hue in self.all_hues:
                    if hue in g.groups and len(g.groups[hue]) > 0:
                        data_array.append(g.get_group(hue)[self.agg].values)
                    else:
                        data_array.append([])
                self.plot_func(ax, data_array, labels=self.all_hues)
            else:
                final_data = data.groupby(self.hue).agg({self.agg: self.aggfunc})
                self.plot_func(ax, final_data.reindex(self.all_hues))
        return ax

    def plot_groupby_agg(self, ax, data):
        # not possible to do value counts and normalize here
        if self.kind in ('bar', 'line'):
            grouped = data.groupby(self.groupby).agg({self.agg: self.aggfunc})
            grouped = grouped.reindex(self.all_groups)
            self.plot_func(ax, grouped)
        else:
            data_array = []
            g = data.groupby(self.groupby)
            for group in self.all_groups:
                if group in g.groups:
                    data_array.append(g.get_group(group)[self.agg].values)
                else:
                    data_array.append([])
            self.plot_func(ax, data_array, labels=self.all_groups)
        return ax

    def plot_groupby_hue_agg(self, ax, data):
        # might need to refactor to put in all_hues, all_groups, all_aggs
        if self.kind in ('bar', 'line'):
            tbl = data.pivot_table(index=self.groupby, columns=self.hue,
                                   values=self.agg, aggfunc=self.aggfunc)
            tbl = tbl.reindex(index=self.all_groups, columns=self.all_hues)
            self.plot_func(ax, tbl)
        else:
            # only available to box plots
            groupby_labels = np.sort(data[self.groupby].unique())
            hue_labels = []
            g = data.groupby(self.hue)
            positions = np.arange(len(groupby_labels))
            move = .8 / len(g)
            start = move * (len(g) - 1) / 2
            start_positions = positions - start
            box_plots = []
            for i, (label, sub_df) in enumerate(g):
                data_array = []
                hue_labels.append(label)
                g2 = sub_df.groupby(self.groupby)
                color = plt.cm.tab10(i)
                for groupby_label in groupby_labels:
                    if groupby_label in g2.groups:
                        data_array.append(g2.get_group(groupby_label)[self.agg].values)
                    else:
                        data_array.append([])
                bp = self.plot_func(ax, data_array, labels=groupby_labels, widths=move,
                                    positions=start_positions + i * move, patch_artist=True,
                                    boxprops={'facecolor': color}, medianprops={'color': 'black'})
                patch = bp['boxes'][0]
                box_plots.append(patch)
            if self.is_single_plot():
                ax.legend(handles=box_plots, labels=hue_labels)
            else:
                ax.figure.legend(handles=box_plots, labels=hue_labels,
                                 bbox_to_anchor=(1.02, .5), loc='center left')
            if self.orient == 'v':
                ax.set_xlim(min(positions) - .5, max(positions) + .5)
                ax.set_xticks(positions)
            else:
                ax.set_ylim(min(positions) - .5, max(positions) + .5)
                ax.set_yticks(positions)
            self.no_legend = False
        return ax

    def plot_row_or_col(self, fig, axes):
        split_by = self.row or self.col
        g = self.data.groupby(split_by)
        how = 'F' if self.row else 'C'
        axes_flat = axes.flatten(how)
        for i, (ax, (val, sub_df)) in enumerate(zip(axes_flat, g)):
            if not (self.groupby or self.hue):
                self.plot_only_agg(ax, sub_df)
            elif self.hue and not self.groupby:
                self.plot_hue_agg(ax, sub_df)
            elif not self.hue and self.groupby:
                self.plot_groupby_agg(ax, sub_df)
            elif self.hue and self.groupby:
                self.plot_groupby_hue_agg(ax, sub_df)
            ax.set_title(val)
        return fig

    def plot_row_and_col(self, fig, axes):
        g = self.data.groupby([self.row, self.col])
        axes_flat = axes.flatten()
        groups = [(r, c) for r in self.all_rows for c in self.all_cols]

        for ax, group in zip(axes_flat, groups):
            p = False
            ax.set_title(' | '.join(group))
            if group not in g.groups:
                continue
            else:
                sub_df = g.get_group(group)
            if not (self.groupby or self.hue):
                self.plot_only_agg(ax, sub_df)
            elif self.hue and not self.groupby:
                self.plot_hue_agg(ax, sub_df)
            elif not self.hue and self.groupby:
                self.plot_groupby_agg(ax, sub_df)
            elif self.hue and self.groupby:
                self.plot_groupby_hue_agg(ax, sub_df)


def aggplot(agg, groupby=None, data=None, hue=None, row=None, col=None, kind='bar', orient='v',
            sort=False, aggfunc='mean', normalize=None, wrap=None, figsize=None, rot=0,
            title=None, sharex=True, sharey=True, xlabel=None, ylabel=None, xlim=None, ylim=None,
            xscale='linear', yscale='linear', kwargs=None):
    """
    The `aggplot` function aggregates a single column of data. To begin,
    choose the column you would like to aggregate and set it as the `agg`
    parameter. The behavior of `aggplot` changes based on the type of
    variable used for `agg`.

    For numeric columns, the average of the values are calculated by default.
    Use the `aggfunc` parameter to choose the type of aggregation. You may
    use strings such as 'min', 'max', 'median', etc...

    For string and categorical columns, the counts of the unique values are
    calculated by default. Use the `normalize` parameter to return the
    percentages instead of the counts. Choose how you would like to
    `normalize` by setting it to one of the strings 'agg', 'hue', 'row',
    'col', or 'all'.

    Use the `groupby` parameter to select a column to group by. This column
    is passed to the pandas DataFrame `groupby` method. Choose the aggregation
    method with `aggfunc`. Note, that you cannot use set `groupby` if the `agg`
    variable is string/categorical.

    Parameters
    ----------
    agg: str
        Column name of DataFrame you would like to aggregate. By default, the
        mean of numeric columns and the counts of unique values of
        string/categorical columns are returned.

    groupby: str
        Column name of the grouping variable. Only available when the `agg`
        variable is numeric.

    data: DataFrame
        A Pandas or Dexplo DataFrame that typically has non-aggregated data.
        This type of data is often referred to as "tidy" or "long" data.

    hue: str
        Column name to further group the `agg` variable within a single plot.
        Each unique value in the `hue` column forms a new group.

    row: str
        Column name used to group data into separate plots. Each unique value
        in the `row` column forms a new row.

    col: str
        Column name used to group data into separate plots. Each unique value
        in the `col` column forms a new row.

    kind: str
        Type of plot to use. Possible choices for all `agg` variables:
        * 'bar'
        * 'line'

        Additional choices for numeric `agg` variables
        * 'hist'
        * 'kde'
        * 'box'

    orient: str {'v', 'h'}
        Choose the orientation of the plots. By default, they are vertical
        ('v'). Use 'h' for horizontal

    sort: bool - default is False
        Whether to sort the `groupby` variables

    aggfunc: str or function
        Used to aggregate `agg` variable. Use any of the strings that Pandas
        can understand. You can also use a custom function as long as it
        aggregates, i.e. returns a single value.

    normalize: str, tuple
        When aggregating a string/categorical column, return the percentage
        instead of the counts. Choose what columns you would like to
        normalize over with the strings 'agg', 'hue', 'row', 'col', or 'all'.
        Use a tuple of these strings to normalize over two or more of the
        above. For example, use ('hue', 'row') to normalize over the
        `hue`, `row` combination.

    wrap: int
        When using either `row` or either `col` and not both, determines the
        maximum number of rows/cols before a new row/col is used.

    figsize: tuple
        Use a tuple of integers. Passed directly to Matplotlib to set the
        size of the figure in inches.

    rot: int
        Long labels will be automatically wrapped, but you can still use
        this parameter to rotate x-tick labels. Only applied to strings.

    title: str
        Sets the figure title NOT the Axes title

    sharex: bool
        Whether all plots should share the x-axis or not. Default is True

    sharey: bool
        Whether all plots should share the y-axis or not. Default is True

    xlabel: str
        Label used for x-axis on figures with a single plot

    ylabel: str
        Label used for y-axis on figures with a single plot

    xlim: 2-item tuple of numerics
        Determines x-axis limits for figures with a single plot

    ylim: 2-item tuple of numerics
        Determines y-axis limits for figures with a single plot

    xscale: {'linear', 'log', 'symlog', 'logit'}
        Sets the scale of the x-axis.

    yscale: {'linear', 'log', 'symlog', 'logit'}
        Sets the scale of the y-axis

    kwargs: dict
        Extra arguments used to control the plot used as the `kind`
        parameter.

    Returns
    -------
    A Matplotlib Axes whenver both `row` and `col` are not defined and a
    Matplotlib Figure when one or both are.

    """
    # TODO: textwrap titles
    # TODO: automate figsize for grid
    # TODO: Allow user to pass in ax to put plot in own figure

    return AggPlot(agg, groupby, data, hue, row, col, kind, orient, sort, aggfunc, normalize,
                    wrap, figsize, rot, title, sharex, sharey, xlabel, ylabel, xlim, ylim,
                    xscale, yscale, kwargs).plot()


def _fit_reg(fit_reg, ci, ax, x, y, data, color, line_kws):
    if not fit_reg:
        return None
    if ci is None:
        ci = 0
    if ci < 0 or ci >= 100:
        raise ValueError('ci must be between 0 and 100 or `None`')

    if line_kws is None:
        line_kws = {}

    if 'lw' not in line_kws:
        line_kws['lw'] = 3

    X = data[x].values
    if len(X) == 1:
        return None
    idx_order = X.argsort()
    y = data[y].values
    if len(X) == 2:
        ax.plot(X, y, color=color, **line_kws)
        return None
    X = sm.add_constant(X)

    # if all x's are the same value, there can be no regression line
    if X.shape[1] == 1:
        return 1
    ols = sm.OLS(y, X).fit()
    pred_obj = ols.get_prediction()
    pred = pred_obj.predicted_mean[idx_order]
    try:
        ax.plot(X[idx_order, 1], pred, color=color, **line_kws)
    except IndexError:
        print(f"col is {x}")
        print(X.shape)
        print(data[x].values)
        print(X)

    if ci != 0:
        st, data, ss2 = summary_table(ols, alpha=1 - ci / 100)
        ax.fill_between(X[idx_order, 1], data[idx_order, 4], data[idx_order, 5],
                        alpha=.3, color=color)


def _calculate_figsize(nrows, ncols):
    return ncols * 2 + 8, nrows * 2 + 4


def scatterplot(x, y, data=None, hue=None, row=None, col=None, figsize=None, wrap=None, s=None,
                fit_reg=False, ci=95, sharex=True, sharey=True, xlabel=None, ylabel=None,
                xlim=None, ylim=None, xscale='linear', yscale='linear', title=None,
                scatter_kws=None, line_kws=None):
    """
    Creates a scatterplot between numeric variables `x` and `y`. Within a
    single plot, use `hue` to color points. Fit a regression line with
    confidence bands by setting `fit_reg` to `True`

    Parameters
    ----------
    x: str
        Column name of numeric variable for x-axis

    y: str
        Column name of numeric variable for y-axis

    data: Pandas or Dexplo DataFrame
        DataFrame whos column names may be used for x, y, hue, row, col, and s

    hue: str
        Column name of string/categorical variable who's unique values split
        are used to color points

    row: str
        Column name of string/categorical variable who's unique values
        split data into separate plots by row

    col: str
        Column name of string/categorical variable who's unique values
        split data int separate plots by column

    figsize: 2-item tuple of ints
        Determines figsize of figure. If left as `None`, the figsize will
        be automatically set based on the number of rows and columns

    wrap: int
        Used whenever exactly one of `row` or `col` is given. Starts a new
        row/column for every `wrap` plots

    s: int or str
        If `s` is an int, then all markers will be this size in points.
        If `s` is a str, then it corresponds to a numeric column in the
        DataFrame that contains the size of each point.

    fit_reg: bool
        When `True`, fit a regression line. By default it is False

    ci: int [0, 100)
        Confidence interval of regression line

    sharex: bool, 'row', or 'col'
        Determines whether the x-axis limits will be shared for each plot.
        Use False so that each plot has its own unique limits or 'row'/'col'
        for all rows/cols to share their limits. Default is True

    sharey: bool, 'row', or 'col'
        Determines whether the y-axis limits will be shared for each plot.
        Use False so that each plot has its own unique limits or 'row'/'col'
        for all rows/cols to share their limits. Default is Tru

    xlabel: str
        Label used for x-axis on figures with a single plot

    ylabel: str
        Label used for y-axis on figures with a single plot

    xlim: 2-item tuple of numerics
        Determines x-axis limits for figures with a single plot

    ylim: 2-item tuple of numerics
        Determines y-axis limits for figures with a single plot

    xscale: {'linear', 'log', 'symlog', 'logit'}
        Sets the scale of the x-axis.

    yscale: {'linear', 'log', 'symlog', 'logit'}
        Sets the scale of the y-axis

    title: str
        Sets the figure title NOT the Axes title

    scatter_kws: dict
        Extra keyword parameters passed to Matplotlib's Axes.scatter function

    line_kws: dict
        Extra keyword parameters passed to Matplotlib's Axes.plot function

    Returns
    -------
    A Matplotlib Axes when making a single plot or a one item tuple of a
    Matplotlib Figure when using `row` or `col`.
    """
    # TODO: Adjust font size of titles and x/y labels to fit larger/smaller figures
    # TODO: Missing data in x and y
    orig_figsize = figsize

    if figsize is None:
        figsize = (12, 6)

    if not isinstance(data, pd.DataFrame):
        raise TypeError('`data` must be a DataFrame')

    if xlabel is None:
        xlabel = x

    if ylabel is None:
        ylabel = y

    if title is None:
        title = ''

    if scatter_kws is None:
        scatter_kws = {}

    prop_dict = {'xlabel': xlabel, 'ylabel': ylabel, 'xlim': xlim, 'ylim': ylim, 'title': title,
                 'xscale':xscale, 'yscale':yscale}

    if not (hue or row or col):
        fig, ax = plt.subplots(figsize=figsize)
        scat = ax.scatter(x, y, data=data, s=s, **scatter_kws)
        ax.set(**prop_dict)
        _fit_reg(fit_reg, ci, ax, x, y, data, scat.get_facecolor()[0], line_kws)
        return ax

    if hue and not (row or col):
        fig, ax = plt.subplots(figsize=figsize)
        hue_map = _map_val_to_color(np.sort(data[hue].unique()))
        for val, sub_df in data.groupby(hue):
            scat = ax.scatter(x, y, data=sub_df, label=val, c=hue_map[val], s=s, **scatter_kws)
            _fit_reg(fit_reg, ci, ax, x, y, sub_df, scat.get_facecolor()[0], line_kws)
            ax.legend()
        ax.set(**prop_dict)
        return ax

    # exactly one of row or col is given
    if bool(row) != bool(col):
        g = data.groupby(row or col)
        if wrap is None:
            n_fixed = len(g)
            n_calc = 1
        else:
            n_fixed = min(wrap, len(g))
            n_calc = (len(g) - 1) // n_fixed + 1

        nrows = n_fixed if row else n_calc
        ncols = n_calc if row else n_fixed
        if orig_figsize is None:
            figsize = _calculate_figsize(nrows, ncols)
        fig, axes = plt.subplots(nrows, ncols, figsize=figsize, sharex=sharex, sharey=sharey)
        scats = []
        labels = []
        label_set = set()
        how = 'F' if row else 'C'
        axes_flat = axes.flatten(how)
        if hue:
            hue_map = _map_val_to_color(np.sort(data[hue].unique()))
        for i, ((val, sub_df), ax) in enumerate(zip(g, axes_flat)):
            if hue:
                for val_hue, sub_sub_df in sub_df.groupby(hue):
                    scat = ax.scatter(x, y, data=sub_sub_df, c=hue_map[val_hue], s=s, **scatter_kws)
                    color = scat.get_facecolor()[0]
                    _fit_reg(fit_reg, ci, ax, x, y, sub_sub_df, color, line_kws)
                    if val_hue not in label_set:
                        scats.append(scat)
                        labels.append(val_hue)
                        label_set.add(val_hue)
            else:
                scat = ax.scatter(x, y, data=sub_df, s=s, **scatter_kws)
                color = scat.get_facecolor()[0]
                _fit_reg(fit_reg, ci, ax, x, y, sub_df, color, line_kws)
            ax_title = '\n'.join(textwrap.wrap(val, 30))
            ax.set_title(ax_title)
        left_over = len(axes_flat) - len(g)

        # Add label to the extra Axes
        if row:
            ax.tick_params(axis='x', reset=True)
        elif left_over > 0:
            for ax in axes[-2, -left_over:]:
                ax.tick_params(axis='x', reset=True)

        for ax in axes_flat[i + 1:]:
            ax.remove()

        fig.legend(scats, labels, bbox_to_anchor=(1.02, .5), loc='lower left')
        fig.suptitle(title, x=.5, y=1.01, ha='center', va='center', fontsize=25)
        fontsize = plt.rcParams['font.size'] * 1.5
        fig.text(-.01, .5, y, va='center', ha='center', rotation=90, fontsize=fontsize)
        fig.text(.5, -.01, x, va='center', ha='center', fontsize=fontsize)
        fig.tight_layout()
        return fig,

    if row and col:
        g = data.groupby([row, col])
        groups = g.groups.keys()
        row_vals = []
        col_vals = []
        row_set = set()
        col_set = set()
        for group in groups:
            row_val, col_val = group
            if row_val not in row_set:
                row_set.add(row_val)
                row_vals.append(row_val)

            if col_val not in col_set:
                col_set.add(col_val)
                col_vals.append(col_val)

        nrows = len(row_vals)
        ncols = len(col_vals)
        if orig_figsize is None:
            figsize = _calculate_figsize(nrows, ncols)
        fig, axes = plt.subplots(nrows, ncols, figsize=figsize, sharex=sharex, sharey=sharey)
        axes_flat = axes.flatten()

        scats = []
        labels = []
        label_set = set()

        i = 0
        if hue:
            hue_map = _map_val_to_color(np.sort(data[hue].unique()))
        for row_val in row_vals:
            for col_val in col_vals:
                ax = axes_flat[i]
                ax_title = '\n'.join(textwrap.wrap(f'{row_val} | {col_val}', 30))
                ax.set_title(ax_title)
                i += 1
                try:
                    sub_df = g.get_group((row_val, col_val))
                except KeyError:
                    continue
                if hue:
                    for val_hue, sub_sub_df in sub_df.groupby(hue):
                        scat = ax.scatter(x, y, data=sub_sub_df, c=hue_map[val_hue],
                                          s=s, **scatter_kws)
                        color = scat.get_facecolor()[0]
                        _fit_reg(fit_reg, ci, ax, x, y, sub_sub_df, color, line_kws)
                        if val_hue not in label_set:
                            scats.append(scat)
                            labels.append(val_hue)
                            label_set.add(val_hue)
                else:
                    scat = ax.scatter(x, y, data=sub_df, s=s, **scatter_kws)
                    color = scat.get_facecolor()[0]
                    _fit_reg(fit_reg, ci, ax, x, y, sub_df, color, line_kws)
        fig.legend(scats, labels, bbox_to_anchor=(1.02, .5), loc='lower left', title=hue)
        fig.suptitle(title, x=.5, y=1.01, ha='center', va='center', fontsize=25)
        fontsize = plt.rcParams['font.size'] * 1.5
        fig.text(-.01, .5, y, va='center', ha='center', rotation=90, fontsize=fontsize)
        fig.text(.5, -.01, x, va='center', ha='center', fontsize=fontsize)
        fig.tight_layout()
        return fig,


def columnplot(kind='line', data=None):
    """
    Plots all columns of dataframe similar to how pandas does it. The x-axis is used as the index
    Supports line plots, bar, kde, hist
    Returns
    -------

    """
    return data.plot(kind=kind)