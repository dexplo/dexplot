import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from . import _utils
from ._common_plot import CommonPlot

NORMALIZE_ERROR_MSG = '`normalize` can only be None, "all", one of the values passed to ' \
                      ' the parameter names "agg", "hue", "row", "col", or a combination' \
                      ' of those parameter names in a tuple, if they are defined.'


class AggPlot(CommonPlot):

    def __init__(self, agg, groupby, data, hue, row, col, kind, orient, sort, aggfunc, normalize,
                 wrap, stacked, figsize, rot, title, sharex, sharey, xlabel, ylabel, xlim,
                 ylim,xscale, yscale, kwargs):
        self.validate_figsize(figsize)
        self.validate_data(data)

        param_dict = {'agg': agg, 'groupby': groupby, 'hue': hue, 'row': row, 'col': col}
        self.validate_column_names(param_dict)

        self.validate_plot_args(orient, wrap, kind, sort, stacked)
        self.validate_agg_kind()
        self.validate_groupby_agg()
        self.validate_kde_hist()
        self.validate_normalize(normalize)
        self.validate_kwargs(kwargs)
        self.validate_mpl_args(rot, title, sharex, sharey, xlabel, ylabel,
                               xlim, ylim, xscale, yscale)
        self.get_uniques()
        self.normalize_counts = self.get_normalize_counts()
        self.single_plot = self.is_single_plot()
        self.plot_func = getattr(self, self.kind + 'plot')
        self.aggfunc = aggfunc
        self.width = .8
        self.no_legend = True

    def validate_plot_args(self, orient, wrap, kind, sort, stacked):
        if orient not in ['v', 'h']:
            raise ValueError('`orient` must be either "v" or "h".')

        if wrap is not None:
            if not isinstance(wrap, int):
                raise TypeError('`wrap` must either be None or an integer. '
                                f'You passed {type(wrap)}')

        if kind not in {'bar', 'line', 'box', 'hist', 'kde'}:
            raise ValueError('`kind` must be either "bar", "line", "box", "hist", "kde"')

        if not isinstance(sort, bool):
            raise TypeError('`sort` must be a bool')

        if not isinstance(stacked, bool):
            raise TypeError("`stacked` must be a boolean")
        else:
            self.stacked = stacked

        self.orient = orient
        self.wrap = wrap
        self.kind = kind
        self.sort = sort
        self.stacked = stacked

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

    def validate_groupby_agg(self):
        if self.groupby:
            if self.agg_kind == 'O' and self.data[self.groupby].dtype.kind == 'O':
                raise TypeError('When the `agg` column is categorical, you cannot use `groupby`. '
                                'Instead, place the groupby column as either '
                                ' `hue`, `row`, or `col`.')

    def validate_kde_hist(self):
        if self.kind in ('hist', 'kde'):
            if self.groupby and self.hue:
                raise NotImplementedError('When plotting a "hist" or "kde", you can set at most '
                                          'one of `groupby` or `hue` but not both')

    def validate_normalize(self, normalize):
        if self.agg_kind == 'O':
            valid_normalize = ['all'] + list(self.col_name_dict)
            if self.groupby in valid_normalize:
                valid_normalize.remove(self.groupby)
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

    def get_uniques(self):
        if self.hue:
            self.all_hues = np.sort(self.data[self.hue].dropna().unique())
        if self.groupby:
            self.all_groupbys = np.sort(self.data[self.groupby].dropna().unique())
        if self.agg_kind == 'O':
            self.all_aggs = np.sort(self.data[self.agg].dropna().unique())
        if self.row:
            self.all_rows = np.sort(self.data[self.row].dropna().unique())
        if self.col:
            self.all_cols = np.sort(self.data[self.col].dropna().unique())

    def get_normalize_counts(self):
        if self.agg_kind != 'O' or not self.normalize:
            return None
        if self.normalize == 'all':
            return self.data.groupby(list(self.col_name_dict)).size().sum()
        if isinstance(self.normalize, str):
            return self.data[self.normalize].value_counts() \
                       .rename_axis(self.normalize).rename(None).reset_index()
        if isinstance(self.normalize, tuple):
            group_cols = list(self.normalize)
            uniques, names = [], []
            for val in self.normalize:
                param_name = self.col_name_dict[val]
                uniques.append(getattr(self, f'all_{param_name}s'))
                names.append(val)

            df = self.data.groupby(group_cols).size()
            mi = pd.MultiIndex.from_product(uniques, names=names)
            return df.reindex(mi).reset_index()

    # DONE with initialization

    def set_single_plot_labels(self, ax):
        if self.kind in ('bar', 'line', 'box'):
            if self.orient == 'v' and self.agg_kind != 'O':
                ax.set_ylabel(self.ylabel or self.agg)
                if not self.groupby:
                    ax.set_xticklabels([])
                    ax.set_xticks([])
                else:
                    ax.tick_params(axis='x', labelrotation=self.rot)
            elif self.orient == 'h' and self.agg_kind != 'O':
                if not (self.groupby or self.normalize):
                    ax.set_yticklabels([])
                    ax.set_yticks([])
                    ax.set_xlabel(self.xlabel or self.agg)
                else:
                    ax.set_xlabel(self.xlabel or self.agg)
            elif self.orient == 'v' and self.agg_kind == 'O':
                pass

            if self.hue and self.no_legend:
                ax.figure.legend(bbox_to_anchor=(1.01, .5), loc='center left')
        else:
            if self.orient == 'v':
                ax.set_xlabel(self.xlabel or self.agg)
            else:
                ax.set_ylabel(self.ylabel or self.agg)
            if (self.groupby or self.hue) and self.no_legend:
                ax.figure.legend(bbox_to_anchor=(1.01, .5), loc='center left')

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
            handles, labels = fig.axes[0].get_legend_handles_labels()
            fig.legend(handles, labels, bbox_to_anchor=(1.02, .5), loc='center left')

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
        not_stacked = 1 - self.stacked
        width = self.width / n_cols ** not_stacked
        bar_start = (n_cols - 1) / 2 * width * not_stacked
        x_range = np.arange(n_rows)
        bottom = 0
        for i, (height, col) in enumerate(zip(data.values.T, data.columns)):
            x_data = x_range - bar_start + i * width * not_stacked
            if self.orient == 'v':
                ax.bar(x_data, height, width, bottom, label=col, tick_label=data.index)
            else:
                ax.barh(x_data, height, width, bottom, label=col, tick_label=data.index)
            bottom += np.nan_to_num(height) * (1 - not_stacked)
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
        try:
            data = data.dropna().values
        except AttributeError:
            data = [d[~np.isnan(d)] for d in data]

        return ax.hist(data, orientation=orientation, label=labels, stacked=self.stacked,
                       **self.kwargs)

    def kdeplot(self, ax, data, **kwargs):
        labels = kwargs['labels']
        if not isinstance(data, list):
            data = [data]
        for label, cur_data in zip(labels, data):
            cur_data = cur_data[~np.isnan(cur_data)]
            if len(cur_data) > 1:
                x, density = _utils._calculate_density(cur_data)
            else:
                x, density = [], []
            if self.orient == 'h':
                x, density = density, x
            ax.plot(x, density, label=label, **self.kwargs)

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
            self.plot_row_or_col(ax)
        elif self.row and self.col:
            self.plot_row_and_col(ax)

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
        self.add_last_tick_labels(fig)

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
            join_key = list(self.normalize)
        else:
            join_key = self.normalize

        unique_col_name = "@@@@@count"

        if self.normalize in (self.row, self.col):
            col_name = self.normalize
            cur_group = data.iloc[0].loc[col_name]
            df = self.normalize_counts
            cur_count = df[df[col_name] == cur_group].iloc[0, -1]
            vc.iloc[:, -1] = vc.iloc[:, -1] / cur_count
            vc = vc.set_index(vc.columns[:-1].tolist())
            return vc
        elif set(self.normalize) == {self.row, self.col}:
            cur_group1, cur_group2 = data.iloc[0].loc[[self.row, self.col]].values
            df = self.normalize_counts
            b = (df[self.row] == cur_group1) & (df[self.col] == cur_group2)
            cur_count = df[b].iloc[0, -1]
            vc.iloc[:, -1] = vc.iloc[:, -1] / cur_count
            vc = vc.set_index(vc.columns[:-1].tolist())
            return vc
        elif (isinstance(self.normalize, tuple) and
              (self.row in self.normalize or self.col in self.normalize)):
            col_names = []
            for val in (self.row, self.col):
                if val in self.normalize:
                    col_names.append(val)
            cur_groups = [data.iloc[0].loc[col_name] for col_name in col_names]
            df = self.normalize_counts.copy()
            b = df[col_names[0]] == cur_groups[0]
            if len(col_names) == 2:
                b = b & (df[col_names[1]] == cur_groups[1])
            cur_counts = df[b].copy()
            cur_counts.columns = cur_counts.columns.tolist()[:-1] + [unique_col_name]
            join_keys = [name for name in self.normalize if name not in (self.row, self.col)]
            vc1 = vc.copy()
            vc1.columns = vc1.columns.tolist()[:-1] + [unique_col_name]
            vc1 = vc1.merge(cur_counts, on=join_keys)
            vc1.iloc[:, -1] = vc1[unique_col_name + '_x'].values / vc1[unique_col_name + '_y'].values
            int_cols = list(range(vc.shape[1] - 1)) + [-1]
            vc1 = vc1.iloc[:, int_cols]
            vc1 = vc1.set_index(vc1.columns[:-1].tolist())
            return vc1
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
                bp = self.plot_func(ax, data_array, labels=self.all_hues)
                if self.kind == 'box':
                    patches = bp['boxes']
                    for i, patch in enumerate(patches):
                        patch.set(facecolor=plt.cm.tab10(i))
                    if self.no_legend:
                        ax.figure.legend(handles=patches, labels=self.all_hues.tolist(),
                                         bbox_to_anchor=(1, .5), loc='center left')
                        self.no_legend = False
            else:
                final_data = data.pivot_table(columns=self.hue, values=self.agg, aggfunc=self.aggfunc)
                final_data = final_data.reindex(columns=self.all_hues)
                self.plot_func(ax, final_data)
        return ax

    def plot_groupby_agg(self, ax, data):
        # not possible to do value counts and normalize here
        if self.kind in ('bar', 'line'):
            grouped = data.groupby(self.groupby).agg({self.agg: self.aggfunc})
            grouped = grouped.reindex(self.all_groupbys)
            self.plot_func(ax, grouped)
        else:
            data_array = []
            g = data.groupby(self.groupby)
            for group in self.all_groupbys:
                if group in g.groups:
                    data_array.append(g.get_group(group)[self.agg].values)
                else:
                    data_array.append([])
            self.plot_func(ax, data_array, labels=self.all_groupbys)

        return ax

    def plot_groupby_hue_agg(self, ax, data):
        # might need to refactor to put in all_hues, all_groups, all_aggs
        if self.kind in ('bar', 'line'):
            tbl = data.pivot_table(index=self.groupby, columns=self.hue,
                                   values=self.agg, aggfunc=self.aggfunc)
            tbl = tbl.reindex(index=self.all_groupbys, columns=self.all_hues)
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

    def plot_row_or_col(self, axes):
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

    def plot_row_and_col(self, axes):
        g = self.data.groupby([self.row, self.col])
        axes_flat = axes.flatten()
        groups = [(r, c) for r in self.all_rows for c in self.all_cols]

        for ax, group in zip(axes_flat, groups):
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
            sort=False, aggfunc='mean', normalize=None, wrap=None, stacked=False,
            figsize=None, rot=0, title=None, sharex=True, sharey=True, xlabel=None,
            ylabel=None, xlim=None, ylim=None, xscale='linear', yscale='linear', kwargs=None):
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
        A Pandas DataFrame that typically has non-aggregated data.
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

    stacked: bool
        Controls whether bars will be stacked on top of each other

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
                    wrap, stacked, figsize, rot, title, sharex, sharey, xlabel, ylabel, xlim, ylim,
                    xscale, yscale, kwargs).plot()