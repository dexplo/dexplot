import textwrap

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from . import _utils


class CommonPlot:

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

    def validate_column_names(self, param_dict):
        self.col_name_dict = {}
        for arg_name, col_name in param_dict.items():
            if col_name:
                if col_name not in self.data.columns:
                    raise KeyError(f'You passed {col_name} to parameter {arg_name} which is not a '
                                   'column name')
                if col_name in self.col_name_dict:
                    param = self.col_name_dict[col_name]
                    raise ValueError(f'The column {col_name} was already used for '
                                     f'parameter {param}.')
                self.col_name_dict[col_name] = arg_name
            self.__dict__[arg_name] = col_name

    def validate_mpl_args(self, rot, title, sharex, sharey,
                          xlabel, ylabel, xlim, ylim, xscale, yscale):
        NoneType = type(None)
        if not isinstance(rot, (int, float)):
            raise ValueError('`rot` must be an int or float')
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
        self.rot = rot
        self.title = title
        self.sharex = sharex
        self.sharey = sharey
        self.xlabel = xlabel
        self.ylabel = ylabel
        self.xlim = xlim
        self.ylim = ylim
        self.xscale = xscale
        self.yscale = yscale

    def is_single_plot(self):
        return not (self.row or self.col)

    def validate_kwargs(self, kwargs, has_lines=True):
        if kwargs is None:
            self.kwargs = {}
        elif not isinstance(kwargs, dict):
            raise TypeError('`kwargs` must be `None` or a dict')
        else:
            self.kwargs = kwargs

        if has_lines:
            if self.kind == 'line':
                if 'lw' not in self.kwargs:
                    self.kwargs['lw'] = 3
                if 'marker' not in self.kwargs:
                    self.kwargs['marker'] = 'o'

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
            self.figsize = _utils._calculate_figsize(nrows, ncols)

        self.nrows, self.ncols = nrows, ncols
        return plt.subplots(nrows, ncols, figsize=self.figsize)

    def align_axes(self, axes):
        def set_lim(cur_axes, axis):
            axes_flat = cur_axes.flatten()
            lims = []
            if axis == 'x':
                get_func, set_func = 'get_xlim', 'set_xlim'
            else:
                get_func, set_func = 'get_ylim', 'set_ylim'
            for ax in axes_flat:
                if ax.lines or ax.patches or ax.collections:
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
                    # good_labels = ax_flat[self.wrap - 1].get_xticklabels()
                    # labels = [label.get_text() for label in good_labels]
                    # ax_flat[-left_over - 1].set_xticklabels(labels)
                    for ax in ax_flat[-left_over:]:
                        ax.remove()
        elif self.col and not self.row:
            if self.nrows > 1:
                num_plots = len(self.all_cols)
                left_over = self.nrows * self.ncols - num_plots
                if left_over > 0:
                    ax_flat = axes.flatten('C')
                    # good_labels = ax_flat[-self.wrap].get_xticklabels()
                    # labels = [label.get_text() for label in good_labels]
                    # for ax in ax_flat[-self.ncols - left_over: -self.ncols]:
                    #     ax.set_xticklabels(labels)
                    for ax in ax_flat[-left_over:]:
                        ax.remove()

    def add_last_tick_labels(self, fig):
        axes = fig.axes
        num_plots = len(axes)
        left_over = self.nrows * self.ncols - num_plots
        if left_over:
            fig.canvas.draw()
            if self.row:
                good_labels = axes[-1].get_xticklabels()
                labels = [label.get_text() for label in good_labels]
                pos = (self.nrows - left_over) * self.ncols - 1
                axes[pos].set_xticklabels(labels)
            else:
                good_labels = axes[-1].get_xticklabels()
                labels = [label.get_text() for label in good_labels]
                for ax in axes[-self.ncols: self.ncols * (self.nrows - 1)]:
                    ax.set_xticklabels(labels)

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