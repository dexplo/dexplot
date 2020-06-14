from ._categories import sequential, diverging, cyclic, qualitative, misc, all_cmaps

import importlib
if importlib.util.find_spec('ipywidgets') and importlib.util.find_spec('IPython'):
    from ._app import color_viewer