# Dexplot

Dexplot is a Python library for delivering beautiful data visualizations. It's aim is to be powerful with a simple and intuitive user experience.

## Goals

The primary goals for dexplot are:

* Maintain a very consistent API with as few functions as necessary to make the desired statistical plots
* Allow the user to tweak the plots without digging into matplotlib

## Installation

`pip install dexplot`

## Built for long and wide data

Dexplot is primarily built for long data, which is a form of data where each row represents a single observation and each column represents a distinct quantity. It is often referred to as "tidy" data.

Dexplot also has the ability to handle wide data, where multiple columns may contain values that represent the same kind of quantity.

## Usage

Dexplot provides a small number of powerful functions that all work similarly. Most plotting functions have the following signature:

```python
dxp.plotting_func(x, y, data, aggfunc, split, row, col, orientation, ...)
```

* `x` - Column name along the x-axis
* `y` - Column name the y-axis
* `data` - Pandas DataFrame
* `aggfunc` - String of pandas aggregation function, 'min', 'max', 'mean', etc...
* `split` - Column name to split data into distinct groups
* `row` - Column name to split data into distinct subplots row-wise
* `col` - Column name to split data into distinct subplots column-wise
* `orientation` - Either vertical (`'v'`) or horizontal (`'h'`). Default for most plots is vertical.

When `aggfunc` is provided, `x` will be the grouping variable and `y` will be aggregated when vertical and vice-versa when horizontal. The best way to learn how to use dexplot is with the examples below.

## Comparison with Seaborn

If you have used the seaborn library, then you should notice a lot of similarities. Much of Dexplot was inspired by Seaborn. Below is a list of the extra features in dexplot not found in seaborn

* The ability to graph relative frequency percentage and normalize over any number of variables
* Far fewer public functions
* No need for multiple functions to do the same thing
* Ability to make grids with a single function instead of having to use a higher level function like `catplot`
* Pandas `groupby` methods are available as strings
* Ability to sort by values
* Ability to sort x/y labels lexicographically
* Both x/y-labels and titles are automatically wrapped so that they don't overlap
* The figure size (plus several other options) and available to change without using matplotlib
* Only matplotlib objects are returned

## Examples

Most of the examples below use long data.

## Bar Charts

The examples come from the Airbnb dataset, which contains many property rental listings from the Washington D.C. area.


```python
%load_ext autoreload
%autoreload 2
```


```python
import dexplot as dxp
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
pd.set_option('display.max_columns', None)
airbnb = dxp.load_dataset('airbnb')
airbnb.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>neighborhood</th>
      <th>property_type</th>
      <th>accommodates</th>
      <th>bathrooms</th>
      <th>bedrooms</th>
      <th>price</th>
      <th>cleaning_fee</th>
      <th>rating</th>
      <th>superhost</th>
      <th>response_time</th>
      <th>latitude</th>
      <th>longitude</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Shaw</td>
      <td>Townhouse</td>
      <td>16</td>
      <td>3.5</td>
      <td>4</td>
      <td>433</td>
      <td>250</td>
      <td>95.0</td>
      <td>No</td>
      <td>within an hour</td>
      <td>38.90982</td>
      <td>-77.02016</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Brightwood Park</td>
      <td>Townhouse</td>
      <td>4</td>
      <td>3.5</td>
      <td>4</td>
      <td>154</td>
      <td>50</td>
      <td>97.0</td>
      <td>No</td>
      <td>NaN</td>
      <td>38.95888</td>
      <td>-77.02554</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Capitol Hill</td>
      <td>House</td>
      <td>2</td>
      <td>1.5</td>
      <td>1</td>
      <td>83</td>
      <td>35</td>
      <td>97.0</td>
      <td>Yes</td>
      <td>within an hour</td>
      <td>38.88791</td>
      <td>-76.99668</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Shaw</td>
      <td>House</td>
      <td>2</td>
      <td>2.5</td>
      <td>1</td>
      <td>475</td>
      <td>0</td>
      <td>98.0</td>
      <td>No</td>
      <td>NaN</td>
      <td>38.91331</td>
      <td>-77.02436</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Kalorama Heights</td>
      <td>Apartment</td>
      <td>3</td>
      <td>1.0</td>
      <td>1</td>
      <td>118</td>
      <td>15</td>
      <td>91.0</td>
      <td>No</td>
      <td>within an hour</td>
      <td>38.91933</td>
      <td>-77.04124</td>
    </tr>
  </tbody>
</table>
</div>



There are nearly 5,000 listings in our dataset. We will use bar charts to aggregate the data.


```python
airbnb.shape
```




    (4902, 12)



### Vertical bar charts

In order to performa an aggregation, you must supply a value for `aggfunc`. Here, we find the median price per neighborhood. Notice that the column names automatically wrap.


```python
dxp.bar(x='neighborhood', y='price', data=airbnb, aggfunc='median')
```




![png](https://github.com/dexplo/dexplot/raw/gh-pages/images/output_8_0.png)



### Components of the groupby aggregation

Anytime the `aggfunc` parameter is set, you have performed a groupby aggregation, which always consists of three components:

* Grouping column - unique values of this column form independent groups (neighborhood)
* Aggregating column - the column that will get summarized with a single value (price)
* Aggregating function - a function that returns a single value (median)

The general format for doing this in pandas is:

```python
df.groupby('grouping column').agg({'aggregating column': 'aggregating function'})
```

Specifically, the following code is executed within dexplot.


```python
airbnb.groupby('neighborhood').agg({'price': 'median'})
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>price</th>
    </tr>
    <tr>
      <th>neighborhood</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Brightwood Park</th>
      <td>90.0</td>
    </tr>
    <tr>
      <th>Capitol Hill</th>
      <td>140.0</td>
    </tr>
    <tr>
      <th>Columbia Heights</th>
      <td>99.0</td>
    </tr>
    <tr>
      <th>Dupont Circle</th>
      <td>130.0</td>
    </tr>
    <tr>
      <th>Edgewood</th>
      <td>100.0</td>
    </tr>
    <tr>
      <th>Kalorama Heights</th>
      <td>119.0</td>
    </tr>
    <tr>
      <th>Shaw</th>
      <td>140.0</td>
    </tr>
    <tr>
      <th>Union Station</th>
      <td>128.5</td>
    </tr>
  </tbody>
</table>
</div>



### Sorting the bars

By default, the grouping column (x-axis here) will be sorted in alphabetical order. Use the `sort` parameter to specify how its sorted.

* `lex_asc` - sort lexicographically A to Z (default)
* `lex_desc` - sort lexicographically Z to A
* `asc` - sort values from least to greatest
* `desc` - sort values from greatest to least
* None - Use order of appearance in DataFrame


```python
fig = dxp.bar(x='neighborhood', y='price', data=airbnb, aggfunc='median', sort='lex_desc')
fig
```




![png](https://github.com/dexplo/dexplot/raw/gh-pages/images/output_12_0.png)




```python
dxp.bar(x='neighborhood', y='price', data=airbnb, aggfunc='median', sort='asc')
```




![png](https://github.com/dexplo/dexplot/raw/gh-pages/images/output_13_0.png)



### Specify order with `x_order`

Specify a specific order of the values on the x-axis by passing a list of values to `x_order`. This can also act as a filter to limit the number of bars.


```python
dxp.bar(x='neighborhood', y='price', data=airbnb, aggfunc='median',
        x_order=['Dupont Circle', 'Edgewood', 'Union Station'])
```




![png](https://github.com/dexplo/dexplot/raw/gh-pages/images/output_15_0.png)



### Horizontal bars

Set `orientation` to `'h'` for horizontal bars. When you do this, you'll need to switch `x` and `y` since the grouping column (neighborhood) will be along the y-axis and the aggregating column (price) will be along the x-axis.


```python
dxp.bar(x='price', y='neighborhood', data=airbnb, aggfunc='median', orientation='h')
```




![png](https://github.com/dexplo/dexplot/raw/gh-pages/images/output_17_0.png)



### Split bars into groups

You can split each bar into further groups by setting the `split` parameter to another column.


```python
dxp.bar(x='neighborhood', y='price', data=airbnb, aggfunc='median', split='superhost')
```




![png](https://github.com/dexplo/dexplot/raw/gh-pages/images/output_19_0.png)



We can use the `pivot_table` method to replicate the results in pandas.


```python
airbnb.pivot_table(index='neighborhood', columns='superhost', 
                   values='price', aggfunc='median')
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>superhost</th>
      <th>No</th>
      <th>Yes</th>
    </tr>
    <tr>
      <th>neighborhood</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Brightwood Park</th>
      <td>90</td>
      <td>90</td>
    </tr>
    <tr>
      <th>Capitol Hill</th>
      <td>150</td>
      <td>135</td>
    </tr>
    <tr>
      <th>Columbia Heights</th>
      <td>95</td>
      <td>105</td>
    </tr>
    <tr>
      <th>Dupont Circle</th>
      <td>125</td>
      <td>139</td>
    </tr>
    <tr>
      <th>Edgewood</th>
      <td>105</td>
      <td>100</td>
    </tr>
    <tr>
      <th>Kalorama Heights</th>
      <td>115</td>
      <td>124</td>
    </tr>
    <tr>
      <th>Shaw</th>
      <td>149</td>
      <td>139</td>
    </tr>
    <tr>
      <th>Union Station</th>
      <td>130</td>
      <td>125</td>
    </tr>
  </tbody>
</table>
</div>



Set the order of the unique split values with `split_order`, which can also act as a filter.


```python
dxp.bar(x='neighborhood', y='price', data=airbnb, aggfunc='median', 
        split='superhost', split_order=['Yes', 'No'])
```




![png](https://github.com/dexplo/dexplot/raw/gh-pages/images/output_23_0.png)



### Stacked bar charts

Stack all the split groups one on top of the other by setting `stacked` to `True`.


```python
dxp.bar(x='neighborhood', y='price', data=airbnb, aggfunc='median', 
        split='superhost', split_order=['Yes', 'No'], stacked=True)
```




![png](https://github.com/dexplo/dexplot/raw/gh-pages/images/output_25_0.png)



### Split into multiple plots

It's possible to split the data further into separate plots by the unique values in a different column with the `row` or `col` parameter. Here, each kind of `property_type` has its own plot.


```python
dxp.bar(x='neighborhood', y='price', data=airbnb, aggfunc='median', 
        split='superhost', col='property_type')
```




![png](https://github.com/dexplo/dexplot/raw/gh-pages/images/output_27_0.png)



If there isn't room for all of the plots, set the `wrap` parameter to an integer to set the maximum number of plots per row/col.


```python
dxp.bar(x='neighborhood', y='price', data=airbnb, aggfunc='median', 
        split='superhost', col='property_type', wrap=2)
```




![png](https://github.com/dexplo/dexplot/raw/gh-pages/images/output_29_0.png)



Use `col_order` to both filter and set a specific order for the plots.


```python
dxp.bar(x='neighborhood', y='price', data=airbnb, aggfunc='median',
        split='superhost', col='property_type', col_order=['House', 'Condominium'])
```




![png](https://github.com/dexplo/dexplot/raw/gh-pages/images/output_31_0.png)



Splits can be made simultaneously along row and columns.


```python
dxp.bar(x='neighborhood', y='price', data=airbnb, aggfunc='median',
        split='superhost', col='property_type', col_order=['House', 'Condominium', 'Apartment'],
        row='bedrooms', row_order=[0, 1, 2, 3])
```




![png](https://github.com/dexplo/dexplot/raw/gh-pages/images/output_33_0.png)



### Set the width of each bar with `size`

The width of the bars is set with the `size` parameter.


```python
dxp.bar(x='neighborhood', y='price', data=airbnb, aggfunc='median', split='property_type',
       split_order=['Apartment', 'House'], x_order=['Dupont Circle', 'Capitol Hill'], size=.5)
```




![png](https://github.com/dexplo/dexplot/raw/gh-pages/images/output_35_0.png)


