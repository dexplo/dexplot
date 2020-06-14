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
        fig = dxp.bar(x='neighborhood', y='price', data=airbnb, aggfunc='median', x_order='desc')
        ticklabels = [t.get_text() for t in fig.axes[0].get_xticklabels()]
        correct = sorted(ticklabels, reverse=True)
        assert ticklabels == correct

    def test_asc_values(self):
        fig = dxp.bar(x='neighborhood', y='price', data=airbnb, aggfunc='median', sort_values='asc')
        ticklabels = [t.get_text() for t in fig.axes[0].get_xticklabels()]
        ticklabels = [label.replace('\n', ' ') for label in ticklabels]
        values = [p.get_height() for p in fig.axes[0].patches]

        s = airbnb.groupby('neighborhood')['price'].median().sort_values()
        correct_labels = s.index.tolist()
        correct_values = s.values.tolist()
        assert ticklabels == correct_labels
        assert values == correct_values

    def test_desc_values(self):
        fig = dxp.bar(x='neighborhood', y='price', data=airbnb, aggfunc='median', sort_values='desc')
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
        dxp.bar(x='neighborhood', y='price', data=airbnb, aggfunc='median',
                x_order=['Dupont Circle', 'Edgewood', 'Union Station'])

        with pytest.raises(ValueError):
            dxp.bar(x='neighborhood', y='price', data=airbnb, aggfunc='median',
                x_order=['Dupont Circle', 'Edgewood', 'DOES NOT EXIST'])

class TestHorizontal:

    def test_horiz(self):
        dxp.bar(x='price', y='neighborhood', data=airbnb, aggfunc='median', orientation='h')


class TestSplit:

    def test_split(self):
        dxp.bar(x='neighborhood', y='price', data=airbnb, aggfunc='median', split='superhost')

    def test_split_order(self):
        dxp.bar(x='neighborhood', y='price', data=airbnb, aggfunc='median', 
                split='superhost', split_order=['Yes', 'No'])

    def test_stacked(self):
        dxp.bar(x='neighborhood', y='price', data=airbnb, aggfunc='median', 
                split='superhost', split_order=['Yes', 'No'], stacked=True)

    def test_errors(self):
        with pytest.raises(ValueError):
            dxp.bar(x='neighborhood', y='price', data=airbnb, aggfunc='median', 
                    split='property_type', split_order=['Yes', 'No'])


class TestRowCol:

    def test_col(self):
        dxp.bar(x='neighborhood', y='price', data=airbnb, aggfunc='median', 
                split='superhost', col='property_type')

    def test_col_wrap(self):
        dxp.bar(x='neighborhood', y='price', data=airbnb, aggfunc='median', 
                split='superhost', col='property_type', wrap=2)

    def test_col_order(self):
        dxp.bar(x='neighborhood', y='price', data=airbnb, aggfunc='median',
                split='superhost', col='property_type', col_order=['House', 'Condominium'])

    def test_row(self):
        dxp.bar(x='neighborhood', y='price', data=airbnb, aggfunc='median', 
                split='superhost', row='property_type')

    def test_row_order(self):
        dxp.bar(x='neighborhood', y='price', data=airbnb, aggfunc='median',
        split='superhost', row='property_type', row_order=['House', 'Condominium'])

    def test_row_wrap(self):
        dxp.bar(x='neighborhood', y='price', data=airbnb, aggfunc='median', 
            split='superhost', row='property_type', wrap=2)

    def test_row_col(self):
        dxp.bar(x='neighborhood', y='price', data=airbnb, aggfunc='median',
                split='superhost', col='property_type', 
                col_order=['House', 'Condominium', 'Apartment'],
                row='bedrooms', row_order=[0, 1, 2, 3])

    def test_sharey(self):
        dxp.bar(x='neighborhood', y='price', data=airbnb, aggfunc='median',
        split='superhost', col='property_type', col_order=['House', 'Condominium', 'Apartment'],
        row='bedrooms', row_order=[0, 1, 2, 3], sharey=False)
    
class TestBarProps:

    def test_bar_size(self):
        dxp.bar(x='neighborhood', y='price', data=airbnb, aggfunc='median', split='property_type',
                split_order=['Apartment', 'House'], 
                x_order=['Dupont Circle', 'Capitol Hill', 'Union Station'], size=.5)