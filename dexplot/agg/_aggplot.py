import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from .. import _utils




# count plot
# self.validate_normalize(normalize)
# self.validate_kwargs(kwargs)
# if not isinstance(stacked, bool):
#             raise TypeError("`stacked` must be a boolean")
#         else:
#             self.stacked = stacked

# def validate_normalize(self, normalize):
#         if self.agg_kind == 'O':
#             valid_normalize = ['all'] + list(self.col_name_dict)
#             if self.groupby in valid_normalize:
#                 valid_normalize.remove(self.groupby)
#             if isinstance(normalize, str):
#                 if normalize not in valid_normalize:
#                     raise ValueError(NORMALIZE_ERROR_MSG)
#             elif isinstance(normalize, tuple):
#                 if len(normalize) == 1:
#                     return self.validate_normalize(normalize[0])
#                 for val in normalize:
#                     if val not in valid_normalize[1:]:
#                         raise ValueError(NORMALIZE_ERROR_MSG)
#                 for val in normalize:
#                     if normalize.count(val) > 1:
#                         raise ValueError(f'{val} is duplicated in your `normalize` tuple')
#             elif normalize is not None:
#                 raise TypeError(NORMALIZE_ERROR_MSG)

#             self.normalize = normalize
#         else:
#             # TODO: force normalziation for numerics
#             self.normalize = False

# def get_normalize_counts(self):
#         if self.agg_kind != 'O' or not self.normalize:
#             return None
#         if self.normalize == 'all':
#             return self.data.groupby(list(self.col_name_dict)).size().sum()
#         if isinstance(self.normalize, str):
#             return self.data[self.normalize].value_counts() \
#                        .rename_axis(self.normalize).rename(None).reset_index()
#         if isinstance(self.normalize, tuple):
#             group_cols = list(self.normalize)
#             uniques, names = [], []
#             for val in self.normalize:
#                 param_name = self.col_name_dict[val]
#                 uniques.append(getattr(self, f'all_{param_name}s'))
#                 names.append(val)

#             df = self.data.groupby(group_cols).size()
#             mi = pd.MultiIndex.from_product(uniques, names=names)
#             return df.reindex(mi).reset_index()


      

#     def boxplot(self, ax, data, **kwargs):
#         vert = self.orientation == 'v'
#         if 'boxprops' not in kwargs:
#             kwargs['boxprops'] = {'facecolor': plt.cm.tab10(0)}
#         if 'medianprops' not in kwargs:
#             kwargs['medianprops'] = {'color': 'black'}
#         if 'patch_artist' not in kwargs:
#             kwargs['patch_artist'] = True
#         return ax.boxplot(data, vert=vert, **kwargs)

#     def histplot(self, ax, data, **kwargs):
#         orientation = 'vertical' if self.orientation == 'v' else 'horizontal'
#         labels = kwargs['labels']
#         try:
#             data = data.dropna().values
#         except AttributeError:
#             data = [d[~np.isnan(d)] for d in data]

#         return ax.hist(data, orientation=orientation, label=labels, stacked=self.stacked,
#                        **self.kwargs)

#     def kdeplot(self, ax, data, **kwargs):
#         labels = kwargs['labels']
#         if not isinstance(data, list):
#             data = [data]
#         for label, cur_data in zip(labels, data):
#             cur_data = cur_data[~np.isnan(cur_data)]
#             if len(cur_data) > 1:
#                 x, density = _utils._calculate_density(cur_data)
#             else:
#                 x, density = [], []
#             if self.orientation == 'h':
#                 x, density = density, x
#             ax.plot(x, density, label=label, **self.kwargs)

#     def plot(self):
#         fig, ax = self.create_figure()
#         if not (self.groupby or self.groupby2 or self.row or self.col):
#             ax = self.plot_only_agg(ax, self.data)
#         elif self.groupby2 and not (self.groupby or self.row or self.col):
#             ax = self.plot_groupby2_agg(ax, self.data)
#         elif self.groupby and not (self.groupby2 or self.row or self.col):
#             ax = self.plot_groupby_agg(ax, self.data)
#         elif self.groupby and self.groupby2 and not (self.row or self.col):
#             ax = self.plot_groupby_groupby2_agg(ax, self.data)
#         elif bool(self.row) != bool(self.col):
#             self.plot_row_or_col(ax)
#         elif self.row and self.col:
#             self.plot_row_and_col(ax)

#         if self.single_plot:
#             self.apply_single_plot_changes(ax)
#         else:
#             self.set_figure_plot_labels(fig)
#             self.align_axes(ax)
#             self.remove_yticklabels(ax)
#             self.remove_xticklabels(ax)

#         self.wrap_labels(fig)
#         self.remove_ax(ax)
#         fig.tight_layout()
#         self.add_last_tick_labels(fig)

#         if self.single_plot:
#             return ax
#         return fig,


# def barplot(self, ax, data, **kwargs):
#         n_rows, n_cols = data.shape
#         not_stacked = 1 - self.stacked
#         width = self.width / n_cols ** not_stacked
#         bar_start = (n_cols - 1) / 2 * width * not_stacked
#         x_range = np.arange(n_rows)
#         bottom = 0
#         for i, (height, col) in enumerate(zip(data.values.T, data.columns)):
#             x_data = x_range - bar_start + i * width * not_stacked
#             if self.orientation == 'v':
#                 ax.bar(x_data, height, width, bottom, label=col, tick_label=data.index)
#             else:
#                 ax.barh(x_data, height, width, bottom, label=col, tick_label=data.index)
#             bottom += np.nan_to_num(height) * (1 - not_stacked)
#         if self.orientation == 'v':
#             ax.set_xticks(x_range)
#         else:
#             ax.set_yticks(x_range)

#     def do_normalization(self, vc, data=None):
#         if not self.normalize:
#             return vc
#         elif self.normalize == 'all':
#             vc.iloc[:, -1] = vc.iloc[:, -1] / self.normalize_counts
#             vc = vc.set_index(vc.columns[:-1].tolist())
#             return vc

#         if isinstance(self.normalize, tuple):
#             join_key = list(self.normalize)
#         else:
#             join_key = self.normalize

#         unique_col_name = "@@@@@count"

#         if self.normalize in (self.row, self.col):
#             col_name = self.normalize
#             cur_group = data.iloc[0].loc[col_name]
#             df = self.normalize_counts
#             cur_count = df[df[col_name] == cur_group].iloc[0, -1]
#             vc.iloc[:, -1] = vc.iloc[:, -1] / cur_count
#             vc = vc.set_index(vc.columns[:-1].tolist())
#             return vc
#         elif set(self.normalize) == {self.row, self.col}:
#             cur_group1, cur_group2 = data.iloc[0].loc[[self.row, self.col]].values
#             df = self.normalize_counts
#             b = (df[self.row] == cur_group1) & (df[self.col] == cur_group2)
#             cur_count = df[b].iloc[0, -1]
#             vc.iloc[:, -1] = vc.iloc[:, -1] / cur_count
#             vc = vc.set_index(vc.columns[:-1].tolist())
#             return vc
#         elif (isinstance(self.normalize, tuple) and
#               (self.row in self.normalize or self.col in self.normalize)):
#             col_names = []
#             for val in (self.row, self.col):
#                 if val in self.normalize:
#                     col_names.append(val)
#             cur_groups = [data.iloc[0].loc[col_name] for col_name in col_names]
#             df = self.normalize_counts.copy()
#             b = df[col_names[0]] == cur_groups[0]
#             if len(col_names) == 2:
#                 b = b & (df[col_names[1]] == cur_groups[1])
#             cur_counts = df[b].copy()
#             cur_counts.columns = cur_counts.columns.tolist()[:-1] + [unique_col_name]
#             join_keys = [name for name in self.normalize if name not in (self.row, self.col)]
#             vc1 = vc.copy()
#             vc1.columns = vc1.columns.tolist()[:-1] + [unique_col_name]
#             vc1 = vc1.merge(cur_counts, on=join_keys)
#             vc1.iloc[:, -1] = vc1[unique_col_name + '_x'].values / vc1[unique_col_name + '_y'].values
#             int_cols = list(range(vc.shape[1] - 1)) + [-1]
#             vc1 = vc1.iloc[:, int_cols]
#             vc1 = vc1.set_index(vc1.columns[:-1].tolist())
#             return vc1
#         else:
#             norms = vc.merge(self.normalize_counts, on=join_key)
#             norms['pct'] = norms.iloc[:, -2] / norms.iloc[:, -1]
#             norms = norms.drop(columns=norms.columns[[-3, -2]])
#             norms = norms.set_index(norms.columns[:-1].tolist())
#         return norms

