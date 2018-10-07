import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


def heatmap(x=None, y=None, agg=None, aggfunc=None, data=None, normalize=None, corr=False,
            annot=False, fmt='.2f', ax=None, figsize=None, title=None, cmap=None,
            cbarlabel="", cbar_kw={}, **kwargs):
    """
    Create a heatmap from a Pandas DataFrame. This function works with either
    tidy data or aggregated data.

    If using tidy data, pass it categorical/string variables to `x` and `y`
    and a numeric variable to `values`. Pass an aggregation function
    as a string to `aggfunc`.  You may also choose to leave `values` as None
    which result in a raw frequency count for the co-occurence of the `x` and
    `y` variables. Set normalize to True to get relative percentages.

    If using aggregated data, only use the `data` parameter. The index and
    columns will label the x and y. The values of the DataFrame will form
    will be used for the heat map.

    Parameters
    ----------
    x: str
        Column name who's unique values will be used to form groups. Can
        only be used with tidy data and should be a categorical/string.

    y: str
        Column name who's unique values will be used to form groups. Can
        only be used with tidy data and should be a categorical/string.

    agg: str
        Column name who's values will be aggregated across the groups
        formed by `x` and `y`.

    aggfunc: str or function
        Used to aggregate `agg` variable. Use any of the strings that Pandas
        can understand. You can also use a custom function as long as it
        aggregates, i.e. returns a single value.

    data: DataFrame
        A Pandas DataFrame containing either tidy or aggregated data

    normalize: str
        Must be one of three strings, "all" or the name of one of the column
        names provided to `x` or `y`.

    corr: bool - Default False
        When set to True, will calcaulte the correlation of the co-occurence
        between each of the unique values in `x` and `y`.  Only works with
        tidy data.

    annot: bool - Default False
        Controls whether the aggregated values will be plotted as
        text in the heatmap.

    fmt: str
        Formatting style for annotations

    ax: Matplotlib Axes
        The Matplotlib Axes object to use for plotting. If not given, then
        create a new Figure and Axes

    figsize: tuple
        A two item tuple of ints used to control the figure size

    title: str
        Sets the title of the figure

    cmap: str
        Matplotlib colormap name

    cbarlabel: str
        Labels the colorbar

    cbar_kw: dict
        Keyword arguments passed to the `colorbar` Figure function

    kwargs: dict
        Keyword arguments passed to the `imshow` Axes function

    Returns
    -------
    A one-item tuple containing a Matplotlib Figure

    References
    ----------
    Code was inspired from Matplotlib page
    https://matplotlib.org/gallery/images_contours_and_fields/image_annotated_heatmap.html
    """

    if figsize is None:
        figsize = (10, 8)

    if not isinstance(data, pd.DataFrame):
        raise TypeError('`data` must be a DataFrame')

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
        if title:
            fig.suptitle(title)
    else:
        fig = ax.figure

    if aggfunc:
        if not agg:
            raise ValueError('If you are setting `aggfunc`, you need to set `agg` as well.')

    if not normalize:
        normalize = False

    if cmap is None:
        cmap = 'RdYlBu_r'

    if x or y:
        if not (x and y):
            raise ValueError('If you supply one of x or y, you must both of them')

        if normalize not in (False, 'all', x, y):
            raise ValueError('If you are setting `normalize`, it must be either '
                             f'"all", "{x}" or "{y}"')
        elif normalize == x:
            normalize = 'columns'
        elif normalize == y:
            normalize = 'index'

        if agg:
            data_values = data[agg]
            if not aggfunc:
                aggfunc = 'mean'
        else:
            data_values = None

        agg_data = pd.crosstab(index=data[y], columns=data[x], values=data_values, aggfunc=aggfunc,
                               normalize=normalize)
    else:
        agg_data = data

    if corr:
        agg_data = agg_data.corr()

    agg_values = agg_data.values
    col_labels = agg_data.columns.tolist()
    row_labels = agg_data.index.tolist()

    # Plot the heatmap
    im = ax.imshow(agg_values, cmap=cmap, **kwargs)

    # Create colorbar
    cbar = fig.colorbar(im, ax=ax, **cbar_kw)
    cbar.ax.set_ylabel(cbarlabel, rotation=-90, va='bottom')

    x_range, y_range = np.arange(agg_data.shape[1]), np.arange(agg_data.shape[0])
    ax.set_xticks(x_range)
    ax.set_yticks(y_range)

    ax.set_xticklabels(col_labels)
    ax.set_yticklabels(row_labels)

    # Let the horizontal axes labeling appear on top.
    ax.tick_params(top=True, bottom=False,
                   labeltop=True, labelbottom=False)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=-30, ha='right', rotation_mode='anchor')

    # Turn spines off and create white grid.
    for edge, spine in ax.spines.items():
        spine.set_visible(False)

    ax.set_xticks(x_range - .5, minor=True)
    ax.set_yticks(y_range - .5, minor=True)
    ax.grid(which='minor', color='w', linestyle='-', linewidth=3)
    ax.tick_params(which='minor', bottom=False, left=False)

    if annot:
        annotate_heatmap(im, agg_values, fmt='{0:' + fmt + '}')

    return fig,


def annotate_heatmap(im, values, fmt="{0:.2f}", **textkw):
    """
    Annotates the heatmap

    https://matplotlib.org/gallery/images_contours_and_fields/image_annotated_heatmap.html
    """

    kw = dict(horizontalalignment="center",
              verticalalignment="center")
    kw.update(textkw)
    n_rows, n_cols = values.shape

    for i in range(n_rows):
        for j in range(n_cols):
            val = values[i, j]
            if not np.isnan(val):
                im.axes.text(j, i, fmt.format(val), **kw)