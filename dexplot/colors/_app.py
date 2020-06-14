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

cmap_dropdown = Dropdown(options=[('Qualitative', 'qualitative'),
                                  ('Sequential', 'sequential'), 
                                  ('Diverging', 'diverging'), 
                                  ('Cyclic', 'cyclic'), 
                                  ('Misc', 'misc'), 
                                  ('All Colormaps', 'all')],
                         value=None,
                         description='Colormap Category: ',
                         style = {'description_width': 'initial'})

def remove_ticks_spines(ax):
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)        

class ColorViewer:

    def __init__(self):
        self.checked_colors = []
        self.test_list = []
        self.cbox_dict = {cat: self.cmap_checkboxes(cat) for cat in cmap_default}
        self.layout = self.create_layout()
        self.fig, self.ax = self.create_figure()
        self.add_interaction()

    def checkbox_maker(self, name, default):
        value = name == default
        c = Checkbox(value=value, description=name, disabled=False, 
                     indent=False, style={'color': 'blue'})
        c.observe(self.cb_handler, 'value')
        return c

    def cmap_checkboxes(self, category):
        rows = []
        row = []
        layout = {'justify_content': 'flex-end', 'margin': '0px'}
        cmaps = cmap_dict[category]
        default = cmap_default[category]
        for name in cmaps:
            row.append(self.checkbox_maker(name, default))
            if len(row) == 10:
                rows.append(HBox(row, layout=layout))
                row = []
        if row:
            rows.append(HBox(row, layout=layout))
        return rows

    def create_image(self): 
        for image in self.ax.images:
            image.remove()

        ticks = []
        ticklabels = []
        i = 0
        
        for i, name in enumerate(self.checked_colors):
            cmap = ListedColormap(colormaps[name])
            self.ax.imshow(ARR, cmap=cmap, extent=[0, 10, i + .2, i + .8], aspect='auto')
            ticks.append(i + .5)
            ticklabels.append(name)
            
        self.ax.set_ylim(0, i + 1)
        self.ax.set_yticks(ticks)
        self.ax.set_yticklabels(ticklabels)
        img_bytes = io.BytesIO()
        self.fig.canvas.print_figure(img_bytes)
        img_bytes.seek(0)

        self.img.layout.visibility = 'visible'
        self.img.value = img_bytes.read()

    def get_checkboxes(self, category):
        if category is None:
            return
        self.checked_colors.clear()
        self.checked_colors.append(cmap_default[category])
        self.layout.children = list(self.layout.children[:2]) + self.cbox_dict[category]
        self.test_list.append('end of get_checkboxes')
        self.create_image()
        
    def cb_handler(self, change):
        name = change['owner'].description
        if change['new']:
            self.checked_colors.append(name)
        else:
            self.checked_colors.remove(name)
        self.create_image()

    def create_layout(self):
        title = HTML('<h1>Color Viewer</h1>')
        self.img = Image(width=700, height=600)
        self.img.layout.visibility = 'hidden'

        rows = []
        row1 = HBox([title], layout={'justify_content': 'flex-start'})    
        row2 = HBox([cmap_dropdown, self.img], layout={'align_items': 'center'})
        rows = [row1, row2]
        
        return VBox(rows)
    
    def create_figure(self):
        fig = plt.Figure(dpi=144, tight_layout=True, figsize=(6, 3))
        ax = fig.add_subplot()
        remove_ticks_spines(ax)
        return fig, ax
    
    def add_interaction(self):
        interactive_output(self.get_checkboxes, {'category': cmap_dropdown})
        cmap_dropdown.value = 'qualitative'
        

    def run(self):
        display(self.layout)


def color_viewer():
    ColorViewer().run()