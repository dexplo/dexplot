from ._colormaps import colormaps

def set_attrs(obj, cmaps):
    for cmap in cmaps:
        setattr(obj, cmap, colormaps[cmap])

sequential_colormaps = [
    'afmhot', 'afmhot_r', 'aggrnyl', 'aggrnyl_r', 'agsunset', 'agsunset_r', 'algae', 'algae_r', 
    'amp', 'amp_r', 'autumn', 'autumn_r', 'binary', 'binary_r', 'blackbody', 'blackbody_r',
    'bluered', 'bluered_r', 'blues', 'blues_r', 'blugrn', 'blugrn_r', 'bluyl', 'bluyl_r', 
    'bone', 'bone_r', 'brwnyl', 'brwnyl_r', 'bugn', 'bugn_r', 'bupu', 'bupu_r', 'burg', 'burg_r',
    'burgyl', 'burgyl_r', 'cividis', 'cividis_r', 'cool', 'cool_r', 'copper', 'copper_r', 
    'darkmint', 'darkmint_r', 'deep', 'deep_r', 'dense', 'dense_r', 'electric', 'electric_r', 
    'emrld', 'emrld_r', 'gist_gray', 'gist_gray_r', 'gist_heat', 'gist_heat_r', 'gist_yarg',
    'gist_yarg_r', 'gnbu', 'gnbu_r', 'gray', 'gray_r', 'greens', 'greens_r', 'greys', 'greys_r',
    'haline', 'haline_r', 'hot', 'hot_r', 'ice', 'ice_r', 'inferno', 'inferno_r', 'jet', 'jet_r',
    'magenta', 'magenta_r', 'magma', 'magma_r', 'matter', 'matter_r', 'mint', 'mint_r', 'oranges',
    'oranges_r', 'orrd', 'orrd_r', 'oryel', 'oryel_r', 'peach', 'peach_r', 'pink', 'pink_r',
    'pinkyl', 'pinkyl_r', 'plasma', 'plasma_r', 'plotly3', 'plotly3_r', 'pubu', 'pubu_r', 'pubugn',
    'pubugn_r', 'purd', 'purd_r', 'purp', 'purp_r', 'purples', 'purples_r', 'purpor', 'purpor_r',
    'rainbow', 'rainbow_r', 'rdbu', 'rdbu_r', 'rdpu', 'rdpu_r', 'redor', 'redor_r', 'reds',
    'reds_r', 'solar', 'solar_r', 'speed', 'speed_r', 'spring', 'spring_r', 'summer', 'summer_r',
    'sunset', 'sunset_r', 'sunsetdark', 'sunsetdark_r', 'teal', 'teal_r', 'tealgrn', 'tealgrn_r',
    'tempo', 'tempo_r', 'thermal', 'thermal_r', 'turbid', 'turbid_r', 'viridis', 'viridis_r',
    'winter', 'winter_r', 'wistia', 'wistia_r', 'ylgn', 'ylgn_r', 'ylgnbu', 'ylgnbu_r', 'ylorbr',
    'ylorbr_r', 'ylorrd', 'ylorrd_r'
    ]

diverging_colormaps = [
    'armyrose', 'armyrose_r', 'balance', 'balance_r', 'brbg', 'brbg_r', 'bwr', 'bwr_r', 'coolwarm',
    'coolwarm_r', 'curl', 'curl_r', 'delta', 'delta_r', 'earth', 'earth_r', 'fall', 'fall_r', 
    'geyser', 'geyser_r', 'picnic', 'picnic_r', 'piyg', 'piyg_r', 'portland', 'portland_r', 'prgn',
    'prgn_r', 'puor', 'puor_r', 'rdbu', 'rdbu_r', 'rdgy', 'rdgy_r', 'rdylbu', 'rdylbu_r', 'rdylgn',
    'rdylgn_r', 'seismic', 'seismic_r', 'spectral', 'spectral_r', 'tealrose', 'tealrose_r', 'temps',
    'temps_r', 'tropic', 'tropic_r'
    ]

cyclic_colormaps = [
    'edge', 'edge_r', 'hsv', 'hsv_r', 'icefire', 'icefire_r', 'mrybm', 'mrybm_r', 'mygbm',
    'mygbm_r', 'phase', 'phase_r', 'twilight', 'twilight_r', 'twilight_shifted', 
    'twilight_shifted_r'
]

qualitative_colormaps = [
    'accent', 'accent_r', 'alphabet', 'alphabet_r', 'antique', 'antique_r', 'bold', 'bold_r',
    'd3', 'd3_r', 'dark12', 'dark12_r', 'dark2', 'dark24', 'dark24_r', 'dark2_r', 'g10', 'g10_r',
    'light24', 'light24_r', 'paired', 'paired_r', 'pastel', 'pastel1', 'pastel1_r', 'pastel2',
    'pastel2_r', 'pastel_r', 'plotly', 'plotly_r', 'prism', 'prism_r', 'safe', 'safe_r', 'set1',
    'set1_r', 'set2', 'set2_r', 'set3', 'set3_r', 't10', 't10_r', 'tab10', 'tab10_r', 'tab20',
    'tab20_r', 'tab20b', 'tab20b_r', 'tab20c', 'tab20c_r', 'vivid', 'vivid_r'
]

misc_colormaps = [
    'brg', 'brg_r', 'cmrmap', 'cmrmap_r', 'cubehelix', 'cubehelix_r', 'flag', 'flag_r', 
    'gist_earth', 'gist_earth_r', 'gist_ncar', 'gist_ncar_r', 'gist_rainbow', 'gist_rainbow_r', 
    'gist_stern', 'gist_stern_r', 'gnuplot', 'gnuplot2', 'gnuplot2_r', 'gnuplot_r', 'jet', 'jet_r',
    'nipy_spectral', 'nipy_spectral_r', 'ocean', 'ocean_r', 'prism', 'prism_r', 'rainbow',
    'rainbow_r', 'terrain', 'terrain_r'
    ]

all_colormaps = (
    sequential_colormaps + diverging_colormaps + cyclic_colormaps + 
    qualitative_colormaps + misc_colormaps
)

class ColorMaps:
    pass

sequential = ColorMaps()
diverging = ColorMaps()
cyclic = ColorMaps()
qualitative = ColorMaps()
misc = ColorMaps()
all_cmaps = ColorMaps()

set_attrs(sequential, sequential_colormaps)
set_attrs(diverging, diverging_colormaps)
set_attrs(cyclic, cyclic_colormaps)
set_attrs(qualitative, qualitative_colormaps)
set_attrs(misc, misc_colormaps)
set_attrs(all_cmaps, all_colormaps)