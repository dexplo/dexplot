import re

import pandas as pd

from . import _plots as plots

def get_doc(func):
    doc = func.__doc__
    return re.sub('data :.*(?=split :)', '', doc, count=1, flags=re.S)


@pd.api.extensions.register_dataframe_accessor("dexplot")
class _DexplotAccessor:
    def __init__(self, pandas_obj):
        self._obj = pandas_obj

    def box(self, x=None, y=None, split=None, row=None, col=None, x_order=None, 
        y_order=None, split_order=None, row_order=None, col_order=None, orientation='h', 
        wrap=None, figsize=None, title=None, sharex=True, sharey=True, xlabel=None, 
        ylabel=None, xlim=None, ylim=None, xscale='linear', yscale='linear', cmap=None, 
        x_textwrap=10, y_textwrap=None, x_rot=None, y_rot=None, mode='group', gap=.2,
        groupgap=0, box_kwargs=None):
        return plots.box(x, y, self._obj, split, row, col, x_order, y_order, split_order, 
                         row_order, col_order, orientation, wrap, figsize, title, sharex, 
                         sharey, xlabel, ylabel, xlim, ylim, xscale, yscale, cmap, 
                         x_textwrap, y_textwrap, x_rot, y_rot, mode, gap, groupgap, box_kwargs)

_DexplotAccessor.box.__doc__ = get_doc(plots.box)