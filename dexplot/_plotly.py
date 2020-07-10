import warnings
import textwrap

import numpy as np
import plotly.graph_objects as go

from ._common_plot import PlotlyCommon, PlotlyCount


def wrap_labels(labels, wrap):
    return [textwrap.fill(label, wrap).replace('\n', '<br>') for label in labels]


def line_plotly(x=None, y=None, data=None, aggfunc=None, split=None, row=None, col=None, 
        x_order=None, y_order=None, split_order=None, row_order=None, col_order=None,
        orientation='v', sort_values=None, wrap=None, figsize=None, title=None, sharex=True, 
        sharey=True, xlabel=None, ylabel=None, xlim=None, ylim=None, xscale='linear', 
        yscale='linear', cmap=None, x_textwrap=10, y_textwrap=None, x_rot=None, y_rot=None):
    
    self = PlotlyCommon(x, y, data, aggfunc, split, row, col, 
                        x_order, y_order, split_order, row_order, col_order,
                        orientation, sort_values, wrap, figsize, title, sharex, 
                        sharey, xlabel, ylabel, xlim, ylim, xscale, yscale, cmap,
                        x_textwrap, y_textwrap, x_rot, y_rot)
    
    showlegend = True
    for (row, col), info in self.final_data.items():
        for i, (x, y, label, col_name, row_label, col_label) in enumerate(info):
            self.fig.add_scatter(x=x, y=y, name=label, row=row, col=col,
                            marker_color=self.colors[i % len(self.colors)], 
                            showlegend=showlegend)
        showlegend = False
    return self.fig


def scatter_plotly(x=None, y=None, data=None, aggfunc=None, split=None, row=None, col=None, 
        x_order=None, y_order=None, split_order=None, row_order=None, col_order=None,
        orientation='v', sort_values=None, wrap=None, figsize=None, title=None, sharex=True, 
        sharey=True, xlabel=None, ylabel=None, xlim=None, ylim=None, xscale='linear', 
        yscale='linear', cmap=None, x_textwrap=10, y_textwrap=None, x_rot=None, y_rot=None):
    
    self = PlotlyCommon(x, y, data, aggfunc, split, row, col, 
                    x_order, y_order, split_order, row_order, col_order,
                    orientation, sort_values, wrap, figsize, title, sharex, 
                    sharey, xlabel, ylabel, xlim, ylim, xscale, yscale, cmap,
                    x_textwrap, y_textwrap, x_rot, y_rot)

    showlegend = True
    for (row, col), info in self.final_data.items():
        for i, (x, y, label, col_name, row_label, col_label) in enumerate(info):
            self.fig.add_scatter(x=x, y=y, name=label, row=row, col=col,
                                    marker_color=self.colors[i % len(self.colors)], 
                                    showlegend=showlegend, mode='markers')
        showlegend = False
    return self.fig


def bar_plotly(x=None, y=None, data=None, aggfunc=None, split=None, row=None, col=None, 
            x_order=None, y_order=None, split_order=None, row_order=None, col_order=None,
            orientation='v', sort_values=None, wrap=None, figsize=None, title=None, 
            sharex=True, sharey=True, xlabel=None, ylabel=None, xlim=None, 
            ylim=None, xscale='linear', yscale='linear', cmap=None, x_textwrap=10, 
            y_textwrap=None, x_rot=None, y_rot=None, mode='group', gap=.2,
            groupgap=0, bar_kwargs=None):

    self = PlotlyCommon(x, y, data, aggfunc, split, row, col, 
                    x_order, y_order, split_order, row_order, col_order,
                    orientation, sort_values, wrap, figsize, title, sharex, 
                    sharey, xlabel, ylabel, xlim, ylim, xscale, yscale, cmap,
                    x_textwrap, y_textwrap, x_rot, y_rot)

    showlegend = self.split is not None
    self.fig.update_layout(barmode=mode, bargap=gap, bargroupgap=groupgap)
    for (row, col), info in self.final_data.items():
        for i, (x, y, label, col_name, row_label, col_label) in enumerate(info):
            if len(x) > 200:
                warnings.warn('You are plotting more than 200 bars. '
                                'Did you forget to provide an `aggfunc`?')

            self.fig.add_bar(x=x, y=y, orientation=self.orientation, 
                             name=label, row=row, col=col, 
                             marker_color=self.colors[i % len(self.colors)], 
                             showlegend=showlegend)
        showlegend = False

    return self.fig


def count_plotly(val, data=None, normalize=False, split=None, row=None, col=None, 
        x_order=None, y_order=None, split_order=None, row_order=None, col_order=None,
        orientation='v', sort_values='desc', wrap=None, figsize=None, title=None, 
        sharex=True, sharey=True, xlabel=None, ylabel=None, xlim=None, ylim=None, 
        xscale='linear', yscale='linear', cmap=None, x_textwrap=10, y_textwrap=None, 
        x_rot=None, y_rot=None, mode='group', gap=.2, groupgap=0, 
        bar_kwargs=None):
    
    x, y = (val, None) if orientation == 'v' else (None, val)
    aggfunc = '__distribution__'
    self = PlotlyCount(x, y, data, aggfunc, split, row, col, 
                        x_order, y_order, split_order, row_order, col_order,
                        orientation, None, wrap, figsize, title, sharex, 
                        sharey, xlabel, ylabel, xlim, ylim, xscale, yscale, cmap,
                        x_textwrap, y_textwrap, x_rot, y_rot, kind='count')

    count_dict = self.get_count_dict(normalize)
    showlegend = self.split is not None
    self.fig.update_layout(barmode=mode, bargap=gap, bargroupgap=groupgap)
    for (row, col), df in count_dict.items():
        if sort_values == 'asc' and not (self.split or self.row or self.col):
            df = df.iloc[::-1]

        labels = df.index.values
        for i, column in enumerate(df.columns):
            values = df[column].values
            x, y = (labels, values) if self.orientation == 'v' else (values, labels)
            self.fig.add_bar(x=x, y=y, orientation=self.orientation, name=column, 
                             row=row, col=col, marker_color=self.colors[i % len(self.colors)], 
                             showlegend=showlegend)
        showlegend = False
    return self.fig


def box_plotly(x=None, y=None, data=None, split=None, row=None, col=None, x_order=None, 
        y_order=None, split_order=None, row_order=None, col_order=None, orientation='h', 
        wrap=None, figsize=None, title=None, sharex=True, sharey=True, xlabel=None, 
        ylabel=None, xlim=None, ylim=None, xscale='linear', yscale='linear', cmap=None, 
        x_textwrap=10, y_textwrap=None, x_rot=None, y_rot=None, mode='group', 
        gap=.2, groupgap=0, box_kwargs=None):

    aggfunc = None
    sort_values = None
    self = PlotlyCommon(x, y, data, aggfunc, split, row, col, 
                    x_order, y_order, split_order, row_order, col_order,
                    orientation, sort_values, wrap, figsize, title, sharex, 
                    sharey, xlabel, ylabel, xlim, ylim, xscale, yscale, cmap,
                    x_textwrap, y_textwrap, x_rot, y_rot)

    showlegend = self.split is not None
    self.fig.update_layout(boxmode=mode, boxgap=gap, boxgroupgap=groupgap)
    for (row, col), info in self.final_data.items():
        for i, (x, y, label, col_name, row_label, col_label) in enumerate(info):
            self.fig.add_box(x=x, y=y, orientation=self.orientation, 
                             name=label, row=row, col=col, 
                             marker_color=self.colors[i % len(self.colors)], 
                             showlegend=showlegend)
        showlegend = False
        
    return self.fig

def violin_plotly(x=None, y=None, data=None, split=None, row=None, col=None, x_order=None, 
        y_order=None, split_order=None, row_order=None, col_order=None, orientation='h', 
        wrap=None, figsize=None, title=None, sharex=True, sharey=True, xlabel=None, 
        ylabel=None, xlim=None, ylim=None, xscale='linear', yscale='linear', cmap=None, 
        x_textwrap=10, y_textwrap=None, x_rot=None, y_rot=None, mode='group', 
        gap=.2, groupgap=0, box_kwargs=None):

    aggfunc = None
    sort_values = None
    self = PlotlyCommon(x, y, data, aggfunc, split, row, col, 
                    x_order, y_order, split_order, row_order, col_order,
                    orientation, sort_values, wrap, figsize, title, sharex, 
                    sharey, xlabel, ylabel, xlim, ylim, xscale, yscale, cmap,
                    x_textwrap, y_textwrap, x_rot, y_rot)

    showlegend = self.split is not None
    self.fig.update_layout(violinmode=mode, violingap=gap, violingroupgap=groupgap)
    for (row, col), info in self.final_data.items():
        for i, (x, y, label, col_name, row_label, col_label) in enumerate(info):
            self.fig.add_violin(x=x, y=y, orientation=self.orientation, 
                                name=label, row=row, col=col, 
                                marker_color=self.colors[i % len(self.colors)], 
                                showlegend=showlegend)
        showlegend = False
        
    return self.fig

def kde_plotly(x=None, y=None, data=None, split=None, row=None, col=None, split_order=None, 
               row_order=None, col_order=None, orientation='v', wrap=None, figsize=None, 
               title=None, sharex=True, sharey=True, xlabel=None, ylabel=None, xlim=None, 
               ylim=None, xscale='linear', yscale='linear', cmap=None, x_textwrap=10, 
               y_textwrap=None, x_rot=None, y_rot=None, range=None, cumulative=False):

    aggfunc = None
    sort_values = None
    self = PlotlyCommon(x, y, data, aggfunc, split, row, col, 
                    x_order, y_order, split_order, row_order, col_order,
                    orientation, sort_values, wrap, figsize, title, sharex, 
                    sharey, xlabel, ylabel, xlim, ylim, xscale, yscale, cmap,
                    x_textwrap, y_textwrap, x_rot, y_rot)

    showlegend = self.split is not None
    from ._utils import calculate_density_1d, calculate_density_2d

    x_order = y_order = None
    # x, y = (x, None) if orientation == 'v' else (None, x)

    if x is not None and y is not None and split is not None:
        raise ValueError('Cannot use `split` for 2-dimensional KDE plots')

    aggfunc = '__distribution__' if y is None else None
    sort_values = None
    self = PlotlyCommon(x, y, data, aggfunc, split, row, col, 
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
                self.fig.add_scatter(x=x, y=y, name=split_label, row=row, col=col,
                            marker_color=self.colors[i % len(self.colors)], 
                            showlegend=showlegend)
            else:
                x, y, split_label = vals[:3]
                xmin, xmax, ymin, ymax, Z = calculate_density_2d(x, y)
                ax.imshow(Z, extent=[xmin, xmax, ymin, ymax], aspect='auto')
    
        showlegend = False
        
    return self.fig
