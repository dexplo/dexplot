import pytest
import numpy as np
import dexplot as dxp


airbnb = dxp.load_dataset('airbnb')


class TestSort:

    def test_lex_asc(self):
        fig = dxp.box(x='price', y='neighborhood', data=airbnb)
        ticklabels = [t.get_text() for t in fig.axes[0].get_xticklabels()]
        correct = sorted(ticklabels)
        assert ticklabels == correct

    def test_lex_desc(self):
        fig = dxp.box(x='price', y='neighborhood', data=airbnb, y_order='desc')
        ticklabels = [t.get_text() for t in fig.axes[0].get_xticklabels()]
        correct = sorted(ticklabels, reverse=True)
        assert ticklabels == correct

    def test_asc_values(self):
        fig = dxp.box(x='price', y='neighborhood', data=airbnb, sort_values='asc')
        ticklabels = [t.get_text() for t in fig.axes[0].get_xticklabels()]
        ticklabels = [label.replace('\n', ' ') for label in ticklabels]
        values = [p.get_height() for p in fig.axes[0].patches]


    def test_desc_values(self):
        fig = dxp.box(x='price', y='neighborhood', data=airbnb, sort_values='desc')
        ticklabels = [t.get_text() for t in fig.axes[0].get_xticklabels()]
        ticklabels = [label.replace('\n', ' ') for label in ticklabels]


class TestOrder:

    def test_x_order(self):
        dxp.box(x='price', y='neighborhood', data=airbnb,
                y_order=['Dupont Circle', 'Edgewood', 'Union Station'])

        with pytest.raises(ValueError):
            dxp.box(x='price', y='neighborhood', data=airbnb,
                y_order=['Dupont Circle', 'Edgewood', 'DOES NOT EXIST'])


class TestVertical:

    def test_vert(self):
        dxp.box(x='neighborhood', y='price', data=airbnb, orientation='v')


class TestSplit:

    def test_split(self):
        dxp.box(x='price', y='neighborhood', data=airbnb, split='superhost')

    def test_split_order(self):
        dxp.box(x='price', y='neighborhood', data=airbnb, 
                split='superhost', split_order=['Yes', 'No'])

    def test_stacked(self):
        dxp.box(x='price', y='neighborhood', data=airbnb, 
                split='superhost', split_order=['Yes', 'No'])


class TestRowCol:

    def test_col(self):
        dxp.box(x='price', y='neighborhood', data=airbnb, 
                split='superhost', col='property_type')

    def test_col_wrap(self):
        dxp.box(x='price', y='neighborhood', data=airbnb, 
                split='superhost', col='property_type', wrap=2)

    def test_col_order(self):
        dxp.box(x='price', y='neighborhood', data=airbnb,
                split='superhost', col='property_type', col_order=['House', 'Condominium'])

    def test_row(self):
        dxp.box(x='price', y='neighborhood', data=airbnb, 
                split='superhost', row='property_type')

    def test_row_order(self):
        dxp.box(x='price', y='neighborhood', data=airbnb,
        split='superhost', row='property_type', row_order=['House', 'Condominium'])

    def test_row_wrap(self):
        dxp.box(x='price', y='neighborhood', data=airbnb, 
            split='superhost', row='property_type', wrap=2)

    def test_row_col(self):
        dxp.box(x='price', y='neighborhood', data=airbnb,
                split='superhost', col='property_type', 
                col_order=['House', 'Condominium', 'Apartment'],
                row='bedrooms', row_order=[0, 1, 2, 3])

    def test_sharex(self):
        dxp.box(x='price', y='neighborhood', data=airbnb,
        split='superhost', col='property_type', col_order=['House', 'Condominium', 'Apartment'],
        row='bedrooms', row_order=[1, 2, 3], sharex=False)
    