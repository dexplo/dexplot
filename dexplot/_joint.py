import numpy as np
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import summary_table
import scipy.stats as st

from . import _utils
from ._common_plot import CommonPlot


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


def jointplot(x, y, data=None, hue=None, row=None, col=None, kind='scatter', figsize=None,
             wrap=None, s=None, fit_reg=False, ci=95, rot=0, sharex=True, sharey=True, xlabel=None,
             ylabel=None, xlim=None, ylim=None, xscale='linear', yscale='linear', title=None,
             scatter_kws=None, line_kws=None):
    """
    Creates a plot between the raw numeric variables `x` and `y`. No
    aggregation is performed. The default plot is a scatterplot. Use
    the parameter `kind` to create these other plots:
    * line
    * kde
    * bar

    Within a single plot, use `hue` to subdivide and color points/lines.
    Fit a regression line with confidence bands by setting `fit_reg`
    to `True`

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

    kind: str
        Kind of plot to be created. Either 'scatter', 'line', 'kde', 'bar'

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

    rot: int
        Long labels will be automatically wrapped, but you can still use
        this parameter to rotate x-tick labels. Only applied to strings.

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

    return JointPlot(x, y, data, hue, row, col, kind, figsize, wrap, s, fit_reg, ci, rot, sharex,
                     sharey, xlabel, ylabel, xlim, ylim, xscale, yscale, title, scatter_kws,
                     line_kws).plot()


class JointPlot(CommonPlot):

    def __init__(self, x, y, data, hue, row, col, kind, figsize,
                 wrap=None, s=None, fit_reg=False, ci=95, rot=0, sharex=True, sharey=True, xlabel=None,
                 ylabel=None, xlim=None, ylim=None, xscale='linear', yscale='linear', title=None,
                 scatter_kws=None, line_kws=None):
        self.validate_figsize(figsize)
        self.validate_data(data)

        param_dict = {'x': x, 'y': y, 'hue': hue, 'row': row, 'col': col, 's': s}
        self.validate_column_names(param_dict)

        self.validate_plot_args(wrap, kind)
        self.validate_mpl_args(rot, title, sharex, sharey, xlabel, ylabel,
                               xlim, ylim, xscale, yscale)
        self.get_uniques()
        self.fit_reg = fit_reg
        self.ci = ci
        self.set_kws(scatter_kws, line_kws)
        self.single_plot = self.is_single_plot()
        self.plot_func = self.get_plotting_func()
        self.no_legend = True

    def validate_plot_args(self, wrap, kind):
        if wrap is not None:
            if not isinstance(wrap, int):
                raise TypeError('`wrap` must either be None or an integer. '
                                f'You passed {type(wrap)}')

        if kind not in ('scatter', 'line', 'kde', 'bar'):
            raise ValueError("`kind` must be either 'scatter', 'line', 'kde', 'bar'")

        self.wrap = wrap
        self.kind = kind

    def set_kws(self, scatter_kws, line_kws):
        if scatter_kws is None:
            self.scatter_kws = {}
        else:
            self.scatter_kws = scatter_kws

        if line_kws is None:
            self.line_kws = {}
        else:
            self.line_kws = line_kws

    def get_uniques(self):
        if self.hue:
            self.all_hues = np.sort(self.data[self.hue].unique())
        if self.row:
            self.all_rows = np.sort(self.data[self.row].unique())
        if self.col:
            self.all_cols = np.sort(self.data[self.col].unique())

    def get_plotting_func(self):
        if self.kind == 'scatter' and self.data[self.x].dtype.kind == 'M':
            return self.date_scatter
        return getattr(self, self.kind + 'plot')

    def apply_single_plot_changes(self, ax):
        if self.hue:
            ax.legend()

        ax.set_xlabel(self.x)
        ax.set_ylabel(self.y)

        if self.kind == 'kde' and self.orig_figsize is None:
            ax.figure.set_size_inches(8, 6)

    def apply_figure_changes(self, fig):
        if self.hue:
            handles, labels = fig.axes[0].get_legend_handles_labels()
            fig.legend(handles, labels, bbox_to_anchor=(1.01, .5), loc='center left')

        fig.text(.5, -.01, self.x, ha='center', va='center')
        fig.text(-.01, .5, self.y, ha='center', va='center', rotation=90)

    def plot(self):
        fig, axes = self.create_figure()
        if not (self.hue or self.row or self.col):
            ax = self.plot_only_xy(axes, self.data)
        elif self.hue and not (self.row or self.col):
            ax = self.plot_hue_xy(axes, self.data)
        elif bool(self.row) != bool(self.col):
            self.plot_row_or_col(axes)
        elif self.row and self.col:
            self.plot_row_and_col(axes)

        if self.single_plot:
            self.apply_single_plot_changes(ax)
        else:
            self.apply_figure_changes(fig)
            self.align_axes(axes)
            self.remove_yticklabels(axes)
            self.remove_xticklabels(axes)

        self.wrap_labels(fig)
        self.remove_ax(axes)
        fig.tight_layout()
        self.add_last_tick_labels(fig)

        if self.single_plot:
            return ax
        return fig,

    def plot_only_xy(self, ax, data):
        self.plot_func(ax, data)
        return ax

    def plot_hue_xy(self, ax, data):
        hue_map = _utils._map_val_to_color(self.all_hues)
        for val, sub_df in data.groupby(self.hue):
            self.plot_func(ax, sub_df, label=val, c=hue_map[val])
        return ax

    def plot_row_or_col(self, axes):
        split_by = self.row or self.col
        g = self.data.groupby(split_by)
        how = 'F' if self.row else 'C'
        axes_flat = axes.flatten(how)
        for i, (ax, (val, sub_df)) in enumerate(zip(axes_flat, g)):
            if not self.hue:
                self.plot_only_xy(ax, sub_df)
            else:
                self.plot_hue_xy(ax, sub_df)
            ax.set_title(val)

    def plot_row_and_col(self, axes):
        g = self.data.groupby([self.row, self.col])
        axes_flat = axes.flatten()
        groups = [(r, c) for r in self.all_rows for c in self.all_cols]

        for ax, group in zip(axes_flat, groups):
            ax.set_title(f'{group[0]} | {group[1]}')
            if group not in g.groups:
                continue
            else:
                sub_df = g.get_group(group)
            if not self.hue:
                self.plot_only_xy(ax, sub_df)
            else:
                self.plot_hue_xy(ax, sub_df)

    def scatterplot(self, ax, data, **kwargs):
        label = kwargs.get('label', '')
        c = kwargs.get('c', None)
        scat = ax.scatter(self.x, self.y, data=data, s=self.s,
                          label=label, c=c, **self.scatter_kws)
        _fit_reg(self.fit_reg, self.ci, ax, self.x, self.y, data,
                 scat.get_facecolor()[0], self.line_kws)
        return ax

    def date_scatter(self, ax, data, **kwargs):
        label = kwargs.get('label', '')
        c = kwargs.get('c', None)
        ax.plot_date(self.x, self.y, data=data, label=label, c=c, **self.scatter_kws)
        return ax

    def lineplot(self, ax, data, **kwargs):
        label = kwargs.get('label', '')
        ax.plot(self.x, self.y, data=data, label=label, **self.line_kws)
        return ax

    def kdeplot(self, ax, data, **kwargs):
        x, y = data[self.x].values, data[self.y].values
        xmin, xmax = x.min(), x.max()
        ymin, ymax = y.min(), y.max()

        # Peform the kernel density estimate
        xx, yy = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]
        positions = np.vstack([xx.ravel(), yy.ravel()])
        values = np.vstack([x, y])
        kernel = st.gaussian_kde(values)
        f = np.reshape(kernel(positions).T, xx.shape)

        ax.contourf(xx, yy, f, cmap='Blues')
        return ax

    def barplot(self, ax, data, **kwargs):
        label = kwargs.get('label', '')
        ax.bar(self.x, self.y, data=data, label=label)
        return ax
