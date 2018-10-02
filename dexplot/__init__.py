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


class _AggPlot:

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
                for val in normalize:
                    if val not in valid_normalize[1:]:
                        raise ValueError(NORMALIZE_ERROR_MSG)
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
            return self.data.groupby(group_cols).size().reset_index()

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

        return plt.subplots(nrows, ncols, figsize=self.figsize,
                            sharex=self.sharex, sharey=self.sharey)

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
        is_numeric = self.agg_kind != 'O'
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
        if self.hue and not fig.legends:
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

        for ax in fig.axes:
            xlabels = ax.get_xticklabels()
            print(xlabels[0])
            num_labels = len(xlabels)
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
                    ratio = max_width / width
                    text_width = int(ratio * len_text)
                    wrapper = textwrap.TextWrapper(text_width)
                    new_labels.append('\n'.join(wrapper.wrap(text)))
                else:
                    new_labels.append(text)
            print(new_labels)
            ax.set_xticklabels(new_labels)

            ylabels = ax.get_yticklabels()
            old_labels = [label.get_text() for label in ylabels]
            new_labels = []
            for label in ylabels:
                text = label.get_text()
                if len(text) > 30:
                    wrapper = textwrap.TextWrapper(30)
                    new_labels.append('\n'.join(wrapper.wrap(text)))
                else:
                    new_labels.append(text)
            # print(new_labels)
            if old_labels != new_labels:
                ax.set_yticklabels(new_labels)

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
            fig = self.plot_row_or_col(fig, ax)

        if self.single_plot:
            self.apply_single_plot_changes(ax)
        else:
            self.set_figure_plot_labels(fig)
        self.wrap_labels(fig)
        fig.tight_layout()
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
        if self.normalize in ('row', 'col') or 'row' in self.normalize or 'col' in self.normalize:
            if isinstance(self.normalize, str):
                col_name = self.__dict__[self.normalize]
                cur_group = data.iloc[0].loc[col_name]
                df = self.normalize_counts
                cur_count = df.loc[df[col_name] == cur_group].iloc[0, -1]
                vc.iloc[:, -1] = vc.iloc[:, -1] / cur_count
                vc = vc.set_index(vc.columns[:-1].tolist())
                return vc
            else:
                pass
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
            tbl = tbl.reindex(index=self.all_aggs, columns=self.all_hues)
            tbl = tbl.stack().reset_index()
            tbl = self.do_normalization(tbl, data)
            tbl = tbl.squeeze().unstack()
            self.plot_func(ax, tbl)
        else:
            if self.kind in ('box', 'hist', 'kde'):
                data_array = []
                g = data.groupby(self.hue)
                for hue in self.all_hues:
                    if hue in g.groups:
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
        ax.tick_params(axis='x', reset=True)
        for ax in axes_flat[i+1:]:
            ax.remove()
        return fig


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

    return _AggPlot(agg, groupby, data, hue, row, col, kind, orient, sort, aggfunc, normalize,
                    wrap, figsize, rot, title, sharex, sharey, xlabel, ylabel, xlim, ylim,
                    xscale, yscale, kwargs).plot()

    if figsize is None:
        figsize = plt.rcParams['figure.figsize']

    if not isinstance(data, pd.DataFrame):
        raise TypeError('`data` must be a DataFrame')

    params = ['agg', 'groupby', 'hue', 'row', 'col']
    locs = locals()
    for param in params:
        col_name = locs[param]
        if col_name and col_name not in data.columns:
            raise ValueError(f'You passed {col_name} to parameter {param} which is not a '
                             'column name')

    if orient not in ['v', 'h']:
        raise ValueError('`orient` must be either "v" or "h".')

    if wrap is not None:
        if not isinstance(wrap, int):
            raise TypeError(f'`wrap` must either be None or an integer. You passed {type(wrap)}')

    if kind not in {'bar', 'line', 'box', 'hist', 'kde'}:
        raise ValueError('`kind` must be either bar, point, box, hist, or kde')

    if kwargs is None:
        kwargs = {}

    agg_data = data[agg]
    agg_kind = agg_data.dtype.kind

    if agg_kind not in ['i', 'f', 'b', 'O']:
        raise TypeError(f'The data type for the `agg` column must either be boolean, integer, '
                        'float, or categorical/object and not {agg_data.dtype}')

    if agg_kind == 'O':
        if kind not in ('bar', 'line'):
            raise ValueError('When the `agg` variable is object/categorical, `kind` can '
                             'only be "bar" or "line"')

    if groupby is not None:
        groupby_data = data[groupby]
        groupby_kind = groupby_data.dtype.kind

    if hue is not None:
        hue_data = data[hue]
        hue_kind = hue_data.dtype.kind

    if row is not None:
        row_data = data[row]
        row_kind = row_data.dtype.kind

    if col is not None:
        col_data = data[col]
        col_kind = col_data.dtype.kind

    width = .8

    # only agg is given
    if not (groupby or hue or row or col):
        fig, ax = plt.subplots(figsize=figsize)
        if agg_kind == 'O':
            if normalize in ['agg', 'all']:
                normalize = True
            elif normalize is not None:
                raise ValueError(NORMALIZE_ERROR_MSG)
            vc = agg_data.value_counts(sort=sort, normalize=normalize)
            if orient == 'v':
                if kind == 'bar':
                    ax.bar(vc.index, vc.values)
                else:
                    ax.plot(vc.index, vc.values, marker='o', lw=3, **kwargs)
                    ax.set_ylim(0)
                ax.tick_params(axis='x', labelrotation=rot)
            else:
                if kind == 'bar':
                    ax.barh(vc.index, vc.values)
                else:
                    ax.plot(vc.values, vc.index, marker='o', lw=3)
                    ax.set_xlim(0)
        elif agg_kind in 'ifb':
            if kind in ('box', 'hist', 'kde'):
                if kind == 'box':
                    ax.boxplot(agg_data, vert=orient == 'v', labels=[agg], **kwargs)
                elif kind == 'hist':
                    if orient == 'v':
                        orientation = 'vertical'
                    else:
                        orientation = 'horizontal'
                    ax.hist(agg_data, orientation=orientation)
                else:
                    x, density = _calculate_density(agg_data)
                    if orient == 'v':
                        ax.plot(x, density)
                    else:
                        ax.plot(density, x)
                return ax

            # For bar and point plots only
            value = agg_data.agg(aggfunc)
            if orient == 'v':
                if kind == 'bar':
                    ax.bar(agg, value)
                elif kind == 'line':
                    ax.plot(agg, value, marker='o', lw=3)
            else:
                if kind == 'bar':
                    ax.barh(agg, value)
                elif kind == 'line':
                    ax.plot([value], [agg], marker='o', lw=3)
        return ax

    if hue and not (groupby or row or col):
        fig, ax = plt.subplots(figsize=figsize)
        if normalize == 'hue':
            normalize = 'index'
        elif normalize == 'agg':
            normalize = 'columns'
        elif normalize == 'all':
            pass
        elif normalize is None:
            normalize=False
        else:
            raise ValueError(NORMALIZE_ERROR_MSG)
        if agg_kind == 'O':
            tbl = pd.crosstab(agg_data, hue_data, normalize=normalize)
            n_rows, n_cols = tbl.shape
            width = .8 / n_cols

            if orient == 'v':
                x_range = np.arange(n_rows)
                for i in range(n_cols):
                    height = tbl.iloc[:, i].values
                    if kind == 'bar':
                        x_data = x_range + i * width
                        ax.bar(x_data, height, width, label=tbl.columns[i])
                    else:
                        ax.plot(x_range, height, label=tbl.columns[i])

                if kind == 'bar':
                    ax.set_xticks(x_range + width * (n_cols - 1) / 2)
                else:
                    ax.set_xticks(x_range)
                    ax.set_ylim(0)

                ax.set_xticklabels(tbl.index)
                ax.tick_params(axis='x', labelrotation=rot)
            else:
                x_range = np.arange(n_rows, 0, -1)
                for i in range(n_cols):
                    height = tbl.iloc[:, i].values
                    if kind == 'bar':
                        x_data = x_range - i * width
                        ax.barh(x_data, height, width, label=tbl.columns[i])
                    else:
                        ax.plot(height, x_range, label=tbl.columns[i])
                if kind == 'bar':
                    ax.set_yticks(x_range - width * (n_cols - 1) / 2)
                else:
                    ax.set_yticks(x_range)
                    ax.set_xlim(0)
                ax.set_yticklabels(tbl.index)
            ax.legend()
        else:
            if kind in ('box', 'hist', 'kde'):
                data_groups = []
                hue_vals = []
                for hue_val, sub_df in data.groupby(hue):
                    data_groups.append(sub_df[agg].values)
                    hue_vals.append(hue_val)
                if kind == 'box':
                    ax.boxplot(data_groups, labels=hue_vals, vert=orient == 'v')
                    ax.tick_params(axis='x', labelrotation=rot)
                elif kind == 'hist':
                    orientation = 'vertical' if orient == 'v' else 'horizontal'
                    ax.hist(data_groups, label=hue_vals, orientation=orientation)
                    ax.legend()
                elif kind == 'kde':
                    for hue_val, data_group in zip(hue_vals, data_groups):
                        x, density = _calculate_density(data_group)
                        if orient == 'v':
                            ax.plot(x, density, label=hue_val)
                        else:
                            ax.plot(density, x, label=hue_val)
                        ax.legend()
            else:
                grouped = data.groupby(hue).agg({agg: aggfunc})
                if orient == 'v':
                    if kind == 'bar':
                        ax.bar(grouped.index, grouped.values[:, 0])
                    else:
                        ax.plot(grouped.index, grouped.values[:, 0], marker='o', lw=3)
                    ax.tick_params(axis='x', labelrotation=rot)
                else:
                    if kind == 'bar':
                        ax.barh(grouped.index[::-1], grouped.values[::-1, 0])
                    else:
                        ax.plot(grouped.values[::-1, 0], grouped.index[::-1], marker='o', lw=3)
        return ax

    if groupby and not (row or col):
        if agg_kind == 'O':
            raise TypeError('When the `agg` column is categorical, you cannot use `groupby`. '
                            'Instead, place the groupby column as either '
                            ' `hue`, `row`, or `col`.')
        fig, ax = plt.subplots(figsize=figsize)

        if hue is None:
            grouped = data.groupby(groupby).agg({agg: aggfunc})
            if orient == 'v':
                if kind == 'bar':
                    ax.bar(grouped.index, grouped.values[:, 0])
                else:
                    ax.plot(grouped.index, grouped.values[:, 0])
                ax.tick_params(axis='x', labelrotation=rot)
            else:
                ax.barh(grouped.index[::-1], grouped.values[::-1, 0])
        else:
            tbl = data.pivot_table(index=groupby, columns=hue, values=agg, aggfunc=aggfunc)
            n_rows, n_cols = tbl.shape
            width = .8 / n_cols
            if orient == 'v':
                x_range = np.arange(n_rows)
                for i in range(n_cols):
                    x_data = x_range + i * width
                    height = tbl.iloc[:, i].values
                    ax.bar(x_data, height, width, label=tbl.columns[i])
                ax.set_xticks(x_range + width * (n_cols - 1) / 2)
                ax.set_xticklabels(tbl.index)
                ax.tick_params(axis='x', labelrotation=rot)
            else:
                x_range = np.arange(n_rows, 0, -1)
                for i in range(n_cols):
                    x_data = x_range - i * width
                    height = tbl.iloc[:, i].values
                    ax.barh(x_data, height, width, label=tbl.columns[i])
                ax.set_yticks(x_range - width * (n_cols - 1) / 2)
                ax.set_yticklabels(tbl.index)
            ax.legend()

    if (row is not None and col is None) or (row is None and col is not None):
        if row is not None:
            rc_name = 'row'
            rc_col_name = row
            is_row = True
            rc_data = row_data
        else:
            rc_name = 'col'
            rc_col_name = col
            is_row = False
            rc_data = col_data

        if groupby is None:
            if hue is None:
                if agg_kind == 'O':
                    if normalize == rc_name:
                        normalize = 'columns'
                    elif normalize == 'agg':
                        normalize = 'index'
                    elif normalize == 'all':
                        pass
                    elif normalize is None:
                        normalize=False
                    else:
                        raise ValueError(NORMALIZE_ERROR_MSG)
                    tbl = pd.crosstab(agg_data, rc_data, normalize=normalize)
                    n_rows, n_cols = tbl.shape
                    fig_rows, fig_cols = _get_fig_shape(n_cols, wrap, is_row)
                    fig, ax_array = plt.subplots(fig_rows, fig_cols, figsize=figsize,
                                                 sharex=sharex, sharey=sharey)
                    width = .8

                    x_range = np.arange(n_rows)
                    if is_row:
                        ax_flat = ax_array.T.flatten()
                    else:
                        ax_flat = ax_array.flatten()
                    for i, ax in enumerate(ax_flat[:n_cols]):
                        height = tbl.iloc[:, i].values
                        if orient == 'v':
                            x_data = x_range
                            ax.bar(x_data, height, width)
                            ax.set_xticks(x_range)
                            ax.set_xticklabels(tbl.index)

                        if orient == 'h':
                            y_data = x_range[::-1]
                            ax.barh(y_data, height, width)

                        ax.set_title(f'{rc_col_name} = {tbl.columns[i]}')

                    if orient == 'v':
                        if ax_array.ndim == 1:
                            for ax in ax_array:
                                ax.set_xticks(x_range)
                                ax.set_xticklabels(tbl.index)
                                ax.tick_params(axis='x', labelrotation=rot)
                        else:
                            n_full_cols = n_cols % fig_cols
                            if n_full_cols == 0:
                                n_full_cols = fig_cols
                            last_row_axes = ax_array[-1, :n_full_cols]
                            for ax in last_row_axes:
                                ax.tick_params(axis='x', labelrotation=rot, labelbottom=True)
                            second_last_row_axes = ax_array[-2, n_full_cols:]
                            for ax in second_last_row_axes:
                                ax.tick_params(axis='x', labelrotation=rot, labelbottom=True)
                    else:
                        if ax_array.ndim == 1:
                            first_col_axes = ax_array[:1]
                        else:
                            first_col_axes = ax_array[:, 0]
                        for ax in first_col_axes:
                            ax.set_yticks(y_data)
                            ax.set_yticklabels(tbl.index)

                        n_full_cols = n_cols % wrap
                        if n_full_cols == 0:
                            n_full_cols = wrap

                        second_last_row_axes = ax_array[-2, n_full_cols:]
                        for ax in second_last_row_axes:
                            ax.tick_params(axis='x', labelbottom=True)

                        fig.tight_layout()

                    for ax in ax_flat[n_cols:]:
                        ax.remove()
                else:
                    # just one aggregation per Axes
                    # will rarely happen
                    tbl = data.groupby(rc_col_name).agg({agg: aggfunc})
                    n_rows = tbl.shape[0]
                    fig_rows, fig_cols = _get_fig_shape(n_rows, wrap, is_row)
                    fig, ax_array = plt.subplots(fig_rows, fig_cols, figsize=figsize,
                                                 sharex=sharex, sharey=sharey)
                    if is_row:
                        ax_flat = ax_array.T.flatten()
                    else:
                        ax_flat = ax_array.flatten()

                    for i, ax in enumerate(ax_flat[:n_rows]):
                        height = tbl.iloc[i]
                        idx = tbl.index[i]
                        if orient == 'v':
                            ax.bar(idx, height, width=.5)
                            ax.tick_params(labelbottom=False)

                        if orient == 'h':
                            ax.barh(idx, height)
                            ax.tick_params(labelleft=False)

                        ax.set_title(f'{rc_col_name} = {idx}')

                    if orient == 'h':
                        if ax_array.ndim == 1:
                            for ax in ax_array:
                                ax.tick_params(labelbottom=True)
                        else:
                            ax.tick_params(labelbottom=True)

                    for ax in ax_flat[n_rows:]:
                        ax.remove()

                    fig.tight_layout()
            else:
                # hue is not None
                if agg_kind == 'O':
                    tbl = data.groupby([rc_col_name, agg, hue]).size().unstack(fill_value=0)
                    row_levels = tbl.index.levels[0]

                    if normalize == 'all':
                        tbl = tbl / tbl.values.sum()
                    elif normalize == rc_name:
                        tbl = tbl.div(tbl.groupby(rc_col_name).sum().sum(axis=1), axis=0, level=0)
                    elif normalize == 'agg':
                        tbl = tbl.div(tbl.groupby(agg).sum().sum(axis=1), axis=0, level=1)
                    elif normalize == 'hue':
                        tbl = tbl / tbl.sum()
                    elif normalize is None:
                        pass
                    elif isinstance(normalize, tuple):
                        n_set = set(normalize)
                        if n_set == set((rc_name, 'hue')):
                            tbl = tbl.div(tbl.groupby(rc_col_name).sum(), axis=0)
                        elif n_set == set((rc_name, 'agg')):
                            tbl = tbl.div(tbl.sum(1), axis=0, level=1)
                        elif n_set == set(('agg', 'hue')):
                            t1 = tbl.groupby(agg).sum()
                            tbl = tbl / t1
                    else:
                        raise ValueError(NORMALIZE_ERROR_MSG)

                    # number of total plotting surfaces
                    n_axes = len(row_levels)
                    fig_rows, fig_cols = _get_fig_shape(n_axes, wrap, is_row)
                    fig, ax_array = plt.subplots(fig_rows, fig_cols, figsize=figsize,
                                                 sharex=sharex, sharey=sharey)
                    if is_row:
                        ax_flat = ax_array.T.flatten()
                    else:
                        ax_flat = ax_array.flatten()
                    for i, (row_level, ax) in enumerate(zip(row_levels, ax_flat[:n_axes])):
                        cur_tbl = tbl.loc[row_level]
                        n_rows, n_cols = cur_tbl.shape
                        width = .8 / n_cols

                        if orient == 'v':
                            x_range = np.arange(n_rows)
                            for j in range(n_cols):
                                x_data = x_range + j * width
                                height = cur_tbl.iloc[:, j].values
                                if i == 0:
                                    ax.bar(x_data, height, width, label=cur_tbl.columns[j])
                                else:
                                    ax.bar(x_data, height, width)
                            ax.set_xticks(x_range + width * (n_cols - 1) / 2)
                            ax.set_xticklabels(cur_tbl.index)
                            ax.tick_params(axis='x', labelrotation=rot)
                            ax.set_title(f'{rc_col_name} = {row_level}')
                        else:
                            x_range = np.arange(n_rows, 0, -1)
                            for j in range(n_cols):
                                x_data = x_range - j * width
                                height = cur_tbl.iloc[:, j].values
                                if i == 0:
                                    ax.barh(x_data, height, width, label=cur_tbl.columns[j])
                                else:
                                    ax.barh(x_data, height, width)
                            ax.set_yticks(x_range - width * (n_cols - 1) / 2)
                            ax.set_yticklabels(cur_tbl.index)
                            ax.tick_params(axis='x', labelrotation=rot)
                            ax.set_title(f'{rc_col_name} = {row_level}')

                    if normalize is False:
                        fig.suptitle('Total Count', y=1.01, fontsize=20)
                    elif normalize == 'all':
                        fig.suptitle('Normalized Count by all', y=1.01, fontsize=20)
                    elif isinstance(normalize, tuple):
                        locs = locals()
                        cols = [locs[norm] for norm in normalize]
                        normalized_columns = ', '.join(cols)
                        fig.suptitle(f'Normalized Count by {normalized_columns}', y=1.01,
                                     fontsize=20)
                    else:
                        normalized_column = locals()[normalize]
                        fig.suptitle(f'Normalized Count by {normalized_column}', y=1.01, fontsize=20)

                    fig.legend()
                    fig.tight_layout()
                else:
                    tbl = data.groupby([rc_col_name, hue]).agg({agg: aggfunc}).squeeze().unstack()
                    row_levels = tbl.index
                    n_axes = len(row_levels)
                    fig_rows, fig_cols = _get_fig_shape(n_axes, wrap, is_row)
                    fig, ax_array = plt.subplots(fig_rows, fig_cols, figsize=figsize,
                                                 sharex=sharex, sharey=sharey)
                    if is_row:
                        ax_flat = ax_array.T.flatten()
                    else:
                        ax_flat = ax_array.flatten()
                    for i, (row_level, ax) in enumerate(zip(row_levels, ax_flat[:n_axes])):
                        n_cols = len(tbl.columns)
                        height = tbl.iloc[i].values
                        ax.set_title(f'{rc_col_name} = {row_level}')
                        if orient == 'v':
                            x_range = np.arange(n_cols)
                            for x1, h1 in zip(x_range, height):
                                ax.bar(x1, h1)
                            ax.set_xticks(x_range)
                            ax.set_xticklabels(tbl.columns)
                            ax.tick_params(axis='x', labelrotation=rot)
                        else:
                            x_range = np.arange(n_cols, 0, -1)
                            for x1, h1 in zip(x_range, height):
                                ax.barh(x1, h1)
                            ax.set_yticks(x_range)
                            ax.set_yticklabels(tbl.columns)
                    fig.text(-.05, 0.5, f'{agg}', va='center', rotation='vertical', fontsize=20)
                    fig.tight_layout()

                for ax in ax_flat[n_axes:]:
                    ax.remove()

                if ax_array.ndim == 2:
                    n_full_cols = n_axes % fig_cols
                    if n_full_cols == 0:
                        n_full_cols = fig_cols
                    last_row_axes = ax_array[-1, :n_full_cols]
                    for ax in last_row_axes:
                        ax.tick_params(axis='x', labelrotation=rot, labelbottom=True)
                    second_last_row_axes = ax_array[-2, n_full_cols:]
                    for ax in second_last_row_axes:
                        ax.tick_params(axis='x', labelrotation=rot, labelbottom=True)

            return fig,
        else:
            # groupby is not none
            if agg_kind == 'O':
                raise TypeError('When the `agg` column is categorical, you cannot use `groupby`. '
                                'Instead, place the groupby column as either '
                                ' `hue`, `row`, or `col`.')
            else:
                if hue is None:
                    tbl = data.groupby([groupby, rc_col_name]).agg({agg: aggfunc}).squeeze().unstack()
                    n_axes = len(tbl.columns)
                    n_bars = len(tbl)
                    fig_rows, fig_cols = _get_fig_shape(n_axes, wrap, is_row)
                    fig, ax_array = plt.subplots(fig_rows, fig_cols, figsize=figsize,
                                                 sharex=sharex, sharey=sharey)
                    if is_row:
                        ax_flat = ax_array.T.flatten()
                    else:
                        ax_flat = ax_array.flatten()
                    for i, ax in enumerate(ax_flat[:n_axes]):
                        height = tbl.iloc[:, i].values
                        ax.set_title(f'{rc_col_name} = {tbl.columns[i]}')
                        if orient == 'v':
                            x_range = np.arange(n_bars)
                            for x1, h1 in zip(x_range, height):
                                ax.bar(x1, h1)
                            ax.set_xticks(x_range)
                            ax.set_xticklabels(tbl.index)
                            ax.tick_params(axis='x', labelrotation=rot)
                        else:
                            x_range = np.arange(n_bars, 0, -1)
                            for x1, h1 in zip(x_range, height):
                                ax.barh(x1, h1)
                            ax.set_yticks(x_range)
                            ax.set_yticklabels(tbl.index)
                    for ax in ax_flat[n_axes:]:
                        ax.remove()

                    if ax_array.ndim == 2:
                        n_full_cols = n_axes % fig_cols
                        if n_full_cols == 0:
                            n_full_cols = fig_cols
                        second_last_row_axes = ax_array[-2, n_full_cols:]
                        for ax in second_last_row_axes:
                            ax.tick_params(axis='x', labelrotation=rot, labelbottom=True)
                else:
                    tbl = data.groupby([groupby, hue, rc_col_name]).agg({agg: aggfunc}).squeeze().unstack()
                    n_axes = len(tbl.columns)
                    n_bars = len(tbl)
                    fig_rows, fig_cols = _get_fig_shape(n_axes, wrap, is_row)
                    fig, ax_array = plt.subplots(fig_rows, fig_cols, figsize=figsize,
                                                 sharex=sharex, sharey=sharey)
                    if is_row:
                        ax_flat = ax_array.T.flatten()
                    else:
                        ax_flat = ax_array.flatten()
                    for i, (col_name, ax) in enumerate(zip(tbl.columns, ax_flat[:n_axes])):
                        cur_tbl = tbl[col_name].unstack()
                        n_groups = len(cur_tbl.columns)
                        ax.set_title(f'{rc_col_name} = {col_name}')
                        x_labels = cur_tbl.index
                        n_bars = len(x_labels)
                        width = .8 / n_groups
                        x_range = np.arange(n_bars)
                        for j, hue_name in enumerate(cur_tbl.columns):
                            height = cur_tbl[hue_name].values
                            label_name = hue_name
                            if i > 0:
                                label_name = ''
                            if orient == 'v':
                                ax.bar(x_range + j * width, height, width, label=label_name)
                                ax.set_xticks(x_range + j / 2 * width)
                                ax.set_xticklabels(x_labels)
                                ax.tick_params(axis='x', labelrotation=rot)
                            else:
                                x_range = np.arange(n_bars, 0, -1)
                                ax.barh(x_range + 1 - j * width, height, width, label=label_name)
                                ax.set_yticks(x_range + width * n_groups / 2 + width)
                                ax.set_yticklabels(x_labels[::-1])
                    for ax in ax_flat[n_axes:]:
                        ax.remove()

                    fig.legend()
                    fig.tight_layout()
                return fig,
    if row is not None and col is not None:
        if groupby is None and hue is None:
            if agg_kind == 'O':
                tbl = data.groupby([row, col, agg]).size().unstack()
            else:
                tbl = data.groupby([row, col]).agg({agg: aggfunc})
            row_levels = tbl.index.levels[0]
            col_levels = tbl.index.levels[1]
            fig_rows = len(row_levels)
            fig_cols = len(col_levels)
            fig, ax_array = plt.subplots(fig_rows, fig_cols, figsize=figsize,
                                         sharex=sharex, sharey=sharey)
            for i, row_level in enumerate(row_levels):
                for j, col_level in enumerate(col_levels):
                    ax = ax_array[i, j]
                    cur_tbl = tbl.loc[(row_level, col_level)]
                    height = cur_tbl.values
                    x_labels = cur_tbl.index
                    ax.bar(x_labels, height)
                    ax.set_title(f'{row} = {row_level} | {col} = {col_level}')
                    ax.tick_params(axis='x', labelrotation=rot)
            fig.tight_layout()
            return fig,

        if (groupby is not None and hue is None) or (groupby is None and hue is not None):
            grouping_col = groupby or hue
            if agg_kind == 'O':
                raise TypeError('When the `agg` column is categorical, you cannot use `groupby`. '
                                'Instead, place the groupby column as either '
                                ' `hue`, `row`, or `col`.')
            tbl = data.groupby([row, col, grouping_col]).agg({agg: aggfunc}).squeeze().unstack()
            row_levels = tbl.index.levels[0]
            col_levels = tbl.index.levels[1]
            fig_rows = len(row_levels)
            fig_cols = len(col_levels)
            fig, ax_array = plt.subplots(fig_rows, fig_cols, figsize=figsize,
                                         sharex=sharex, sharey=sharey)
            for i, row_level in enumerate(row_levels):
                for j, col_level in enumerate(col_levels):
                    ax = ax_array[i, j]
                    cur_tbl = tbl.loc[(row_level, col_level)]
                    height = cur_tbl.values
                    x_labels = cur_tbl.index
                    ax.bar(x_labels, height)
                    ax.set_title(f'{row} = {row_level} | {col} = {col_level}')
                    ax.tick_params(axis='x', labelrotation=rot)
            return fig,

        if groupby is not None and hue is not None:
            if agg_kind == 'O':
                raise TypeError('When the `agg` column is categorical, you cannot use `groupby`. '
                                'Instead, place the groupby column as either '
                                ' `hue`, `row`, or `col`.')
            tbl = data.groupby([row, col, groupby, hue]).agg({agg: aggfunc}).squeeze().unstack()
            row_levels = tbl.index.levels[0]
            col_levels = tbl.index.levels[1]
            fig_rows = len(row_levels)
            fig_cols = len(col_levels)
            fig, ax_array = plt.subplots(fig_rows, fig_cols, figsize=figsize,
                                         sharex=sharex, sharey=sharey)
            for i, row_level in enumerate(row_levels):
                for j, col_level in enumerate(col_levels):
                    ax = ax_array[i, j]
                    cur_tbl = tbl.loc[(row_level, col_level)]
                    n_groups = len(cur_tbl.columns)
                    n_bars = len(cur_tbl.index)
                    x_range = np.arange(n_bars)
                    x_labels = cur_tbl.index
                    hue_labels = cur_tbl.columns
                    width = .8 / n_groups

                    for k in range(n_groups):
                        if i == 0 and j == 0:
                            label_name = hue_labels[k]
                        else:
                            label_name = ''
                        height = cur_tbl.iloc[:, k]
                        if kind == 'line':
                            ax.plot(x_range, height, label=label_name, marker="o")
                        elif kind == 'box':
                            ax.boxplot
                        else:
                            if orient == 'v':
                                ax.bar(x_range + k * width, height, width, label=label_name)
                            else:
                                ax.barh(x_range[::-1] + 1 - k * width, height, width, label=label_name)
                    ax.set_title(f'{row} = {row_level} | {col} = {col_level}')
                    ax.tick_params(axis='x', labelrotation=rot)
                    if kind == 'line':
                        ax.set_xticks(x_range)
                        ax.set_xticklabels(x_labels)
                    else:
                        if orient == 'v':
                            ax.set_xticks(x_range + width * (n_groups - 1) / 2)
                            ax.set_xticklabels(x_labels)
                        else:
                            ax.set_yticks(x_range + 1 - width * (n_groups - 1) / 2)
                            ax.set_yticklabels(x_labels[::-1])


                        # ax.tick_params(axis='y', labelleft=True)
            fig.legend()
            # fig.subplots_adjust(wspace=.2)
            # fig.tight_layout()
            return fig,


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
    return ncols * 2 + 6, nrows * 2 + 4


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
        figsize = plt.rcParams['figure.figsize']

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