import io

from ipywidgets import Dropdown, Image, HBox, HTML, Checkbox, interactive_output, VBox
from IPython.display import display
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np

from dexplot.colors._colormaps import colormaps
from dexplot.colors._categories import (qualitative_colormaps, sequential_colormaps, 
                                        diverging_colormaps, cyclic_colormaps, misc_colormaps, 
                                        all_colormaps)

ARR = np.linspace(0, 1, 256).reshape((1, -1)).repeat(20, 0)

def remove_ticks_spines(ax):
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

cmap_dict = {'qualitative': qualitative_colormaps,
             'sequential': sequential_colormaps,
             'diverging': diverging_colormaps,
             'cyclic': cyclic_colormaps,
             'misc': misc_colormaps,
             'all': all_colormaps}

cmap_default = {'qualitative': 't10',
             'sequential': 'viridis',
             'diverging': 'coolwarm',
             'cyclic': 'edge',
             'misc': 'ocean',
             'all': 'tab10'}

cmap_category = Dropdown(
    options=[('Qualitative', 'qualitative'),
             ('Sequential', 'sequential'), 
             ('Diverging', 'diverging'), 
             ('Cyclic', 'cyclic'), 
             ('Misc', 'misc'), 
             ('All Colormaps', 'all')],
    value=None,
    description='Colormap Category: ',
    style = {'description_width': 'initial'})

def create_image():
    for image in ax.images:
        image.remove()
    img.layout.visibility = 'visible'
    ticks = []
    ticklabels = []
    i = 0
    for i, name in enumerate(checked_colors):
        cmap = ListedColormap(colormaps[name])
        ax.imshow(ARR, cmap=cmap, extent=[0, 10, i + .2, i + .8], aspect='auto')
        ticks.append(i + .5)
        ticklabels.append(name)
        
    ax.set_ylim(0, i + 1)
    ax.set_yticks(ticks)
    ax.set_yticklabels(ticklabels)
    img_bytes = io.BytesIO()
    fig.canvas.print_figure(img_bytes)
    img_bytes.seek(0)
    img.value = img_bytes.read()

def cmap_checkboxes(category):
    rows = []
    row = []
    layout = {'justify_content': 'flex-end', 'margin': '0px'}
    cmaps = cmap_dict[category]
    default = cmap_default[category]
    for name in cmaps:
        row.append(checkbox_maker(name, default))
        if len(row) == 10:
            rows.append(HBox(row, layout=layout))
            row = []
    if row:
        rows.append(HBox(row, layout=layout))
    return rows

def get_checkboxes(category):
    if category is None:
        return
    checked_colors.clear()
    checked_colors.append(cmap_default[category])
    layout.children = rows + cbox_dict[category]
    create_image()
    
def cb_handler(change):
    name = change['owner'].description
    if change['new']:
        checked_colors.append(name)
    else:
        checked_colors.remove(name)
    create_image()
    
def checkbox_maker(name, default):
    value = name == default
    c = Checkbox(value=value, description=name, disabled=False, indent=False, style={'color': 'blue'})
    c.observe(cb_handler, 'value')
    return c
    
checked_colors = []
cbox_dict = {cat: cmap_checkboxes(cat) for cat in cmap_default}
title = HTML('<h1>Color Viewer</h1>')
img = Image(width=700, height=600)
img.layout.visibility = 'hidden'

rows = []
row1 = HBox([title], layout={'justify_content': 'flex-start'})    
row2 = HBox([cmap_category, img], layout={'align_items': 'center'})
rows = [row1, row2]
layout = VBox(rows)

fig = plt.Figure(dpi=144, tight_layout=True, figsize=(6, 3))
ax = fig.add_subplot()
remove_ticks_spines(ax)

interactive_output(get_checkboxes, {'category': cmap_category})
cmap_category.value = 'qualitative'


def ColorViewer():
    display(layout)
