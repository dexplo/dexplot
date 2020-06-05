import pytest
import numpy as np
import dexplot as dxp


airbnb = dxp.load_dataset('airbnb')

class TestAgg:

    def test_string_name(self):
        dxp.bar(x='neighborhood', y='price', data=airbnb, aggfunc='median')
        dxp.bar(x='neighborhood', y='price', data=airbnb, aggfunc='mean')
        dxp.bar(x='neighborhood', y='price', data=airbnb, aggfunc='min')
        dxp.bar(x='neighborhood', y='price', data=airbnb, aggfunc='max')
        dxp.bar(x='neighborhood', y='price', data=airbnb, aggfunc='size')

    def test_function(self):
        dxp.bar(x='neighborhood', y='price', data=airbnb, aggfunc=np.median)
        dxp.bar(x='neighborhood', y='price', data=airbnb, aggfunc=np.mean)
        dxp.bar(x='neighborhood', y='price', data=airbnb, aggfunc=np.min)
        dxp.bar(x='neighborhood', y='price', data=airbnb, aggfunc=np.max)
        dxp.bar(x='neighborhood', y='price', data=airbnb, aggfunc=np.size)


class TestSort:

    def test_lex_asc(self):
        fig = dxp.bar(x='neighborhood', y='price', data=airbnb, aggfunc='median')
        ticklabels = [t.get_text() for t in fig.axes[0].get_xticklabels()]
        correct = sorted(ticklabels)
        assert ticklabels == correct

    def test_lex_desc(self):
        fig = dxp.bar(x='neighborhood', y='price', data=airbnb, aggfunc='median', sort='lex_desc')
        ticklabels = [t.get_text() for t in fig.axes[0].get_xticklabels()]
        correct = sorted(ticklabels, reverse=True)
        assert ticklabels == correct

    def test_asc_values(self):
        fig = dxp.bar(x='neighborhood', y='price', data=airbnb, aggfunc='median', sort='asc')
        ticklabels = [t.get_text() for t in fig.axes[0].get_xticklabels()]
        ticklabels = [label.replace('\n', ' ') for label in ticklabels]
        values = [p.get_height() for p in fig.axes[0].patches]

        s = airbnb.groupby('neighborhood')['price'].median().sort_values()
        correct_labels = s.index.tolist()
        correct_values = s.values.tolist()
        assert ticklabels == correct_labels
        assert values == correct_values

    def test_desc_values(self):
        fig = dxp.bar(x='neighborhood', y='price', data=airbnb, aggfunc='median', sort='desc')
        ticklabels = [t.get_text() for t in fig.axes[0].get_xticklabels()]
        ticklabels = [label.replace('\n', ' ') for label in ticklabels]
        values = [p.get_height() for p in fig.axes[0].patches]
        
        df = airbnb.groupby('neighborhood').agg({'price': 'median'}).reset_index() \
                   .sort_values(['price', 'neighborhood'], ascending=[False, True])
        s = df.set_index('neighborhood').squeeze()
        correct_labels = s.index.tolist()
        correct_values = s.values.tolist()
        assert ticklabels == correct_labels
        assert values == correct_values


class TestOrder:

    def test_x_order(self):
        pass


    