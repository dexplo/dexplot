# Dexplot

[![](https://img.shields.io/pypi/v/dexplot)](https://pypi.org/project/dexplot)
[![PyPI - License](https://img.shields.io/pypi/l/dexplot)](LICENSE)

Dexplot is a Python library for delivering beautiful data visualizations with a simple and intuitive user experience.

## Goals

The primary goals for dexplot are:

* Maintain a very consistent API with as few functions as necessary to make the desired statistical plots
* Allow the user tremendous power without using matplotlib


## Installation

`pip install dexplot`

## Built for long and wide data

Dexplot is primarily built for long data, which is a form of data where each row represents a single observation and each column represents a distinct quantity. It is often referred to as "tidy" data. Here, we have some long data.

![](https://raw.githubusercontent.com/dexplo/dexplot/gh-pages/images/long.png)

Dexplot also has the ability to handle wide data, where multiple columns may contain values that represent the same kind of quantity. The same data above has been aggregated to show the mean for each combination of neighborhood and property type. It is now wide data as each column contains the same quantity (price).

![](https://raw.githubusercontent.com/dexplo/dexplot/gh-pages/images/wide.png)

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

## Families of plots

There are two primary families of plots, **aggregation** and **distribution**. Aggregation plots take a sequence of values and return a **single** value using the function provided to `aggfunc` to do so. Distribution plots take a sequence of values and depict the shape of the distribution in some manner.

* Aggregation
    * bar
    * line
    * scatter
    * count
* Distribution
    * box
    * violin
    * hist
    * kde

## Comparison with Seaborn

If you have used the seaborn library, then you should notice a lot of similarities. Much of dexplot was inspired by Seaborn. Below is a list of the extra features in dexplot not found in seaborn

* Ability to graph relative frequency and normalize over any number of variables
* No need for multiple functions to do the same thing (far fewer public functions)
* Ability to make grids with a single function instead of having to use a higher level function like `catplot`
* Pandas `groupby` methods available as strings
* Ability to sort by values
* Ability to sort x/y labels lexicographically
* Ability to select most/least frequent groups
* x/y labels are wrapped so that they don't overlap
* Figure size (plus several other options) and available to change without using matplotlib
* A matplotlib figure object is returned

## Examples

Most of the examples below use long data.

## Aggregating plots - bar, line and scatter

We'll begin by covering the plots that **aggregate**. An aggregation is defined as a function that summarizes a sequence of numbers with a single value. The examples come from the Airbnb dataset, which contains many property rental listings from the Washington D.C. area.


```python
import dexplot as dxp
import pandas as pd
airbnb = dxp.load_dataset('airbnb')
airbnb.head()
```

<div>
<table border="1" class="dataframe">
  <thead style="border-bottom:1px solid black; vertical-align:bottom;">
    <tr style="text-align: right;">
      <th></th>
      <th style="color:red;">neighborhood</th>
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



There are more than 4,000 listings in our dataset. We will use bar charts to aggregate the data.


```python
airbnb.shape
```




    (4581, 12)



### Vertical bar charts

In order to performa an aggregation, you must supply a value for `aggfunc`. Here, we find the median price per neighborhood. Notice that the column names automatically wrap.


```python
dxp.bar(x='neighborhood', y='price', data=airbnb, aggfunc='median')
```




![png](https://raw.githubusercontent.com/dexplo/dexplot/gh-pages/images/output_7_0.png)



Line and scatter plots can be created with the same command, just substituting the name of the function. They both are not good choices for the visualization since the grouping variable (neighborhood) has no meaningful order.


```python
dxp.line(x='neighborhood', y='price', data=airbnb, aggfunc='median')
```




![png](https://raw.githubusercontent.com/dexplo/dexplot/gh-pages/images/output_9_0.png)




```python
dxp.scatter(x='neighborhood', y='price', data=airbnb, aggfunc='median')
```




![png](https://raw.githubusercontent.com/dexplo/dexplot/gh-pages/images/output_10_0.png)



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
      <td>87.0</td>
    </tr>
    <tr>
      <th>Capitol Hill</th>
      <td>129.5</td>
    </tr>
    <tr>
      <th>Columbia Heights</th>
      <td>95.0</td>
    </tr>
    <tr>
      <th>Dupont Circle</th>
      <td>125.0</td>
    </tr>
    <tr>
      <th>Edgewood</th>
      <td>100.0</td>
    </tr>
    <tr>
      <th>Kalorama Heights</th>
      <td>118.0</td>
    </tr>
    <tr>
      <th>Shaw</th>
      <td>133.5</td>
    </tr>
    <tr>
      <th>Union Station</th>
      <td>120.0</td>
    </tr>
  </tbody>
</table>
</div>



### Number and percent of missing values with  `'countna'` and `'percna'`

In addition to all the common aggregating functions, you can use the strings `'countna'` and `'percna'` to get the number and percentage of missing values per group.


```python
dxp.bar(x='neighborhood', y='response_time', data=airbnb, aggfunc='countna')
```




![png](https://raw.githubusercontent.com/dexplo/dexplot/gh-pages/images/output_14_0.png)



### Sorting the bars by values

By default, the bars will be sorted by the grouping column (x-axis here) in alphabetical order. Use the `sort_values` parameter to sort the bars by value.

* None - sort x/y axis labels alphabetically (default)
* `asc` - sort values from least to greatest
* `desc` - sort values from greatest to least


```python
dxp.bar(x='neighborhood', y='price', data=airbnb, aggfunc='median', sort_values='asc')
```




![png](https://raw.githubusercontent.com/dexplo/dexplot/gh-pages/images/output_16_0.png)



Here, we sort the values from greatest to least.


```python
dxp.bar(x='neighborhood', y='price', data=airbnb, aggfunc='median', sort_values='desc')
```




![png](https://raw.githubusercontent.com/dexplo/dexplot/gh-pages/images/output_18_0.png)



### Specify order with `x_order`

Specify a specific order of the labels on the x-axis by passing a list of values to `x_order`. This can also act as a filter to limit the number of bars.


```python
dxp.bar(x='neighborhood', y='price', data=airbnb, aggfunc='median',
        x_order=['Dupont Circle', 'Edgewood', 'Union Station'])
```




![png](https://raw.githubusercontent.com/dexplo/dexplot/gh-pages/images/output_20_0.png)



By default, `x_order` and all of the `_order` parameters are set to `'asc'` by default, which will order them alphabetically. Use the string `'desc'` to sort in the opposite direction.


```python
dxp.bar(x='neighborhood', y='price', data=airbnb, aggfunc='median', x_order='desc')
```




![png](https://raw.githubusercontent.com/dexplo/dexplot/gh-pages/images/output_22_0.png)



### Filter for the neighborhoods with most/least frequency of occurrence

You can use `x_order` again to filter for the x-values that appear the most/least often by setting it to the string `'top n'` or `'bottom n'` where `n` is an integer. Here, we filter for the top 4 most frequently occurring neighborhoods. This option is useful when there are dozens of unique values in the grouping column.


```python
dxp.bar(x='neighborhood', y='price', data=airbnb, aggfunc='median',
        x_order='top 4')
```




![png](https://raw.githubusercontent.com/dexplo/dexplot/gh-pages/images/output_24_0.png)



We can verify that the four neighborhoods are the most common.


```python
airbnb['neighborhood'].value_counts()
```




    Columbia Heights    773
    Union Station       713
    Capitol Hill        654
    Edgewood            610
    Dupont Circle       549
    Shaw                514
    Brightwood Park     406
    Kalorama Heights    362
    Name: neighborhood, dtype: int64



### Horizontal bars

Set `orientation` to `'h'` for horizontal bars. When you do this, you'll need to switch `x` and `y` since the grouping column (neighborhood) will be along the y-axis and the aggregating column (price) will be along the x-axis.


```python
dxp.bar(x='price', y='neighborhood', data=airbnb, aggfunc='median', 
        orientation='h', sort_values='desc')
```




![png](https://raw.githubusercontent.com/dexplo/dexplot/gh-pages/images/output_28_0.png)



Switching orientation is possible for most other plots.


```python
dxp.line(x='price', y='neighborhood', data=airbnb, aggfunc='median', orientation='h')
```




![png](https://raw.githubusercontent.com/dexplo/dexplot/gh-pages/images/output_30_0.png)



### Split bars into groups

You can split each bar into further groups by setting the `split` parameter to another column.


```python
dxp.bar(x='neighborhood', y='price', data=airbnb, aggfunc='median', split='superhost')
```




![png](https://raw.githubusercontent.com/dexplo/dexplot/gh-pages/images/output_32_0.png)



We can use the `pivot_table` method to verify the results in pandas.


```python
airbnb.pivot_table(index='superhost', columns='neighborhood', 
                   values='price', aggfunc='median')
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>neighborhood</th>
      <th>Brightwood Park</th>
      <th>Capitol Hill</th>
      <th>Columbia Heights</th>
      <th>Dupont Circle</th>
      <th>Edgewood</th>
      <th>Kalorama Heights</th>
      <th>Shaw</th>
      <th>Union Station</th>
    </tr>
    <tr>
      <th>superhost</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>No</th>
      <td>85.0</td>
      <td>129.0</td>
      <td>90.5</td>
      <td>120.0</td>
      <td>100.0</td>
      <td>110.0</td>
      <td>130.0</td>
      <td>120.0</td>
    </tr>
    <tr>
      <th>Yes</th>
      <td>90.0</td>
      <td>130.0</td>
      <td>103.0</td>
      <td>135.0</td>
      <td>100.0</td>
      <td>124.0</td>
      <td>135.0</td>
      <td>125.0</td>
    </tr>
  </tbody>
</table>
</div>



Set the order of the unique split values with `split_order`, which can also act as a filter.


```python
dxp.bar(x='neighborhood', y='price', data=airbnb, aggfunc='median', 
        split='superhost', split_order=['Yes', 'No'])
```




![png](https://raw.githubusercontent.com/dexplo/dexplot/gh-pages/images/output_36_0.png)



Like all the `_order` parameters, `split_order` defaults to `'asc'` (alphabetical) order. Set it to `'desc'` for the opposite.


```python
dxp.bar(x='neighborhood', y='price', data=airbnb, aggfunc='median',
        split='property_type', split_order='desc')
```




![png](https://raw.githubusercontent.com/dexplo/dexplot/gh-pages/images/output_38_0.png)



Filtering for the most/least frequent split categories is possible.


```python
dxp.bar(x='neighborhood', y='price', data=airbnb, aggfunc='median', 
        split='property_type', split_order='bottom 2')
```




![png](https://raw.githubusercontent.com/dexplo/dexplot/gh-pages/images/output_40_0.png)



Verifying that the least frequent property types are Townhouse and Condominium.


```python
airbnb['property_type'].value_counts()
```




    Apartment      2403
    House           877
    Townhouse       824
    Condominium     477
    Name: property_type, dtype: int64



### Stacked bar charts

Stack all the split groups one on top of the other by setting `stacked` to `True`.


```python
dxp.bar(x='neighborhood', y='price', data=airbnb, aggfunc='median', 
        split='superhost', split_order=['Yes', 'No'], stacked=True)
```




![png](https://raw.githubusercontent.com/dexplo/dexplot/gh-pages/images/output_44_0.png)



### Split into multiple plots

It's possible to split the data further into separate plots by the unique values in a different column with the `row` and `col` parameters. Here, each kind of `property_type` has its own plot.


```python
dxp.bar(x='neighborhood', y='price', data=airbnb, aggfunc='median', 
        split='superhost', col='property_type')
```




![png](https://raw.githubusercontent.com/dexplo/dexplot/gh-pages/images/output_46_0.png)



If there isn't room for all of the plots, set the `wrap` parameter to an integer to set the maximum number of plots per row/col. We also specify the `col_order` to be descending alphabetically.


```python
dxp.bar(x='neighborhood', y='price', data=airbnb, aggfunc='median', 
        split='superhost', col='property_type', wrap=2, col_order='desc')
```




![png](https://raw.githubusercontent.com/dexplo/dexplot/gh-pages/images/output_48_0.png)



Use `col_order` to both filter and set a specific order for the plots.


```python
dxp.bar(x='neighborhood', y='price', data=airbnb, aggfunc='median',
        split='superhost', col='property_type', col_order=['House', 'Condominium'])
```




![png](https://raw.githubusercontent.com/dexplo/dexplot/gh-pages/images/output_50_0.png)



Splits can be made simultaneously along row and columns.


```python
dxp.bar(x='neighborhood', y='price', data=airbnb, aggfunc='median', split='superhost', 
        col='property_type', col_order=['House', 'Condominium', 'Apartment'],
        row='bedrooms', row_order=[1, 2, 3])
```




![png](https://raw.githubusercontent.com/dexplo/dexplot/gh-pages/images/output_52_0.png)



By default, all axis limits are shared. Allow each plot to set its own limits by setting `sharex` and `sharey` to `False`.


```python
dxp.bar(x='neighborhood', y='price', data=airbnb, aggfunc='median', split='superhost', 
        col='property_type', col_order=['House', 'Condominium', 'Apartment'],
        row='bedrooms', row_order=[1, 2, 3], sharey=False)
```




![png](https://raw.githubusercontent.com/dexplo/dexplot/gh-pages/images/output_54_0.png)



### Set the width of each bar with `size`

The width (height when horizontal) of the bars is set with the `size` parameter. By default, this value is .9. Think of this number as the relative width of all the bars for a particular x/y value, where 1 is the distance between each x/y value.


```python
dxp.bar(x='neighborhood', y='price', data=airbnb, 
        aggfunc='median', split='property_type',
        split_order=['Apartment', 'House'], 
        x_order=['Dupont Circle', 'Capitol Hill', 'Union Station'], size=.5)
```




![png](https://raw.githubusercontent.com/dexplo/dexplot/gh-pages/images/output_56_0.png)



### Splitting line plots

All the other aggregating plots work similarly.


```python
dxp.line(x='neighborhood', y='price', data=airbnb, 
        aggfunc='median', split='property_type',
        split_order=['Apartment', 'House'], 
        x_order=['Dupont Circle', 'Capitol Hill', 'Union Station'])
```




![png](https://raw.githubusercontent.com/dexplo/dexplot/gh-pages/images/output_58_0.png)



## Distribution plots - box, violin, histogram, kde

Distribution plots work similarly, but do not have an `aggfunc` since they do not aggregate. They take their group of values and draw some kind of shape that gives information on how that variable is distributed. 

### Box plots

Box plots have colored boxes with ends at the first and third quartiles and a line at the median. The whiskers are placed at 1.5 times the difference between the third and first quartiles (Interquartile range (IQR)). Fliers are the points outside this range and plotted individually. By default, both box and violin plots are plotted horizontally.


```python
dxp.box(x='price', y='neighborhood', data=airbnb)
```




![png](https://raw.githubusercontent.com/dexplo/dexplot/gh-pages/images/output_60_0.png)



Split the groups in the same manner as with the aggregation plots.


```python
dxp.box(x='price', y='neighborhood', data=airbnb, 
        split='superhost', split_order=['Yes', 'No'])
```




![png](https://raw.githubusercontent.com/dexplo/dexplot/gh-pages/images/output_62_0.png)



Order the appearance of the splits alphabetically (in descending order here).


```python
dxp.box(x='price', y='neighborhood', data=airbnb, 
        split='property_type', split_order='desc')
```




![png](https://raw.githubusercontent.com/dexplo/dexplot/gh-pages/images/output_64_0.png)



### Filter range of values with `x_order`

It's possible to filter the range of possible values by passing in a list of the minimum and maximum to `x_order`.


```python
dxp.box(x='price', y='neighborhood', data=airbnb, 
        split='superhost', x_order=[50, 250])
```




![png](https://raw.githubusercontent.com/dexplo/dexplot/gh-pages/images/output_66_0.png)



Change the `x` and `y` while setting `orientation` to make vertical bar plots.


```python
dxp.box(x='neighborhood', y='price', data=airbnb, orientation='v',
        split='property_type', split_order='top 2')
```




![png](https://raw.githubusercontent.com/dexplo/dexplot/gh-pages/images/output_68_0.png)



Violin plots work identically to box plots, but show "violins", kernel density plots duplicated on both sides of a line.


```python
dxp.violin(x='price', y='neighborhood', data=airbnb, 
          split='superhost', split_order=['Yes', 'No'])
```




![png](https://raw.githubusercontent.com/dexplo/dexplot/gh-pages/images/output_70_0.png)



Splitting by rows and columns is possible as well with distribution plots.


```python
dxp.box(x='price', y='neighborhood', data=airbnb,split='superhost', 
        col='property_type', col_order=['House', 'Condominium', 'Apartment'],
        row='bedrooms', row_order=[1, 2])
```




![png](https://raw.githubusercontent.com/dexplo/dexplot/gh-pages/images/output_72_0.png)



### Histograms

Histograms work in a slightly different manner. Instead of passing both `x` and `y`, you give it a single numeric column. A vertical histogram with 20 bins of the counts is created by default.


```python
dxp.hist(val='price', data=airbnb)
```




![png](https://raw.githubusercontent.com/dexplo/dexplot/gh-pages/images/output_74_0.png)



We can use `split` just like we did above and also create horizontal histograms.


```python
dxp.hist(val='price', data=airbnb, orientation='h', split='superhost', bins=15)
```




![png](https://raw.githubusercontent.com/dexplo/dexplot/gh-pages/images/output_76_0.png)



Here, we customize our histogram by plotting the cumulative density as opposed to the raw frequency count using the outline of the bars ('step').


```python
dxp.hist(val='price', data=airbnb, split='bedrooms', split_order=[1, 2, 3], 
         bins=30, density=True, histtype='step', cumulative=True)
```




![png](https://raw.githubusercontent.com/dexplo/dexplot/gh-pages/images/output_78_0.png)



### KDE Plots

Kernel density estimates provide an estimate for the probability distribution of a continuous variable. Here, we examine how price is distributed by bedroom.


```python
dxp.kde(x='price', data=airbnb, split='bedrooms', split_order=[1, 2, 3])
```




![png](https://raw.githubusercontent.com/dexplo/dexplot/gh-pages/images/output_80_0.png)



Graph the cumulative distribution instead on multiple plots.


```python
dxp.kde(x='price', data=airbnb, split='bedrooms', 
        split_order=[1, 2, 3], cumulative=True, col='property_type', wrap=2)
```




![png](https://raw.githubusercontent.com/dexplo/dexplot/gh-pages/images/output_82_0.png)



### Two-dimensional KDE's

Provide two numeric columns to `x` and `y` to get a two dimensional KDE.


```python
dxp.kde(x='price', y='cleaning_fee', data=airbnb)
```




![png](https://raw.githubusercontent.com/dexplo/dexplot/gh-pages/images/output_84_0.png)



Create a grid of two-dimensional KDE's.


```python
dxp.kde(x='price', y='cleaning_fee', data=airbnb, row='neighborhood', wrap=3)
```




![png](https://raw.githubusercontent.com/dexplo/dexplot/gh-pages/images/output_86_0.png)



## Count plots

The `count` function graphs the frequency of unique values as bars. By default, it plots the values in descending order.


```python
dxp.count(val='neighborhood', data=airbnb)
```




![png](https://raw.githubusercontent.com/dexplo/dexplot/gh-pages/images/output_88_0.png)



In pandas, this is a straightforward call to the `value_counts` method.


```python
airbnb['neighborhood'].value_counts()
```




    Columbia Heights    773
    Union Station       713
    Capitol Hill        654
    Edgewood            610
    Dupont Circle       549
    Shaw                514
    Brightwood Park     406
    Kalorama Heights    362
    Name: neighborhood, dtype: int64



### Relative frequency with `normalize`

Instead of the raw counts, get the relative frequency by setting normalize to `True`.


```python
dxp.count(val='neighborhood', data=airbnb, normalize=True)
```




![png](https://raw.githubusercontent.com/dexplo/dexplot/gh-pages/images/output_92_0.png)



Here, we split by property type.


```python
dxp.count(val='neighborhood', data=airbnb, split='property_type')
```




![png](https://raw.githubusercontent.com/dexplo/dexplot/gh-pages/images/output_94_0.png)



In pandas, this is done with the `crosstab` function.


```python
pd.crosstab(index=airbnb['property_type'], columns=airbnb['neighborhood'])
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>neighborhood</th>
      <th>Brightwood Park</th>
      <th>Capitol Hill</th>
      <th>Columbia Heights</th>
      <th>Dupont Circle</th>
      <th>Edgewood</th>
      <th>Kalorama Heights</th>
      <th>Shaw</th>
      <th>Union Station</th>
    </tr>
    <tr>
      <th>property_type</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Apartment</th>
      <td>167</td>
      <td>299</td>
      <td>374</td>
      <td>397</td>
      <td>244</td>
      <td>284</td>
      <td>315</td>
      <td>323</td>
    </tr>
    <tr>
      <th>Condominium</th>
      <td>35</td>
      <td>70</td>
      <td>97</td>
      <td>62</td>
      <td>65</td>
      <td>42</td>
      <td>52</td>
      <td>54</td>
    </tr>
    <tr>
      <th>House</th>
      <td>131</td>
      <td>137</td>
      <td>157</td>
      <td>47</td>
      <td>146</td>
      <td>23</td>
      <td>61</td>
      <td>175</td>
    </tr>
    <tr>
      <th>Townhouse</th>
      <td>73</td>
      <td>148</td>
      <td>145</td>
      <td>43</td>
      <td>155</td>
      <td>13</td>
      <td>86</td>
      <td>161</td>
    </tr>
  </tbody>
</table>
</div>



Horizontal stacked count plots.


```python
dxp.count(val='neighborhood', data=airbnb, split='property_type', 
          orientation='h', stacked=True, col='superhost')
```




![png](https://raw.githubusercontent.com/dexplo/dexplot/gh-pages/images/output_98_0.png)



### Normalize over different variables

Setting `normalize` to `True`, returns the relative frequency with respect to all of the data. You can normalize over any of the variables provided.


```python
dxp.count(val='neighborhood', data=airbnb, split='property_type', normalize='neighborhood', 
                title='Relative Frequency by Neighborhood')
```




![png](https://raw.githubusercontent.com/dexplo/dexplot/gh-pages/images/output_100_0.png)



Normalize over several variables at once with a list.


```python
dxp.count(val='neighborhood', data=airbnb, split='superhost', 
          row='property_type', col='bedrooms', col_order=[1, 2],
          normalize=['neighborhood', 'property_type', 'bedrooms'], stacked=True)
```




![png](https://raw.githubusercontent.com/dexplo/dexplot/gh-pages/images/output_102_0.png)



## Wide data

Dexplot can also plot wide data, or data where no aggregation happens. Here is a scatter plot of the location of each listing.


```python
dxp.scatter(x='longitude', y='latitude', data=airbnb, 
            split='neighborhood', col='bedrooms', col_order=[2, 3])
```




![png](https://raw.githubusercontent.com/dexplo/dexplot/gh-pages/images/output_104_0.png)



If you've already aggregated your data, you can plot it directly without specifying `x` or `y`.


```python
df = airbnb.pivot_table(index='neighborhood', columns='property_type', 
                        values='price', aggfunc='mean')
df
```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>property_type</th>
      <th>Apartment</th>
      <th>Condominium</th>
      <th>House</th>
      <th>Townhouse</th>
    </tr>
    <tr>
      <th>neighborhood</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Brightwood Park</th>
      <td>96.119760</td>
      <td>105.000000</td>
      <td>121.671756</td>
      <td>133.479452</td>
    </tr>
    <tr>
      <th>Capitol Hill</th>
      <td>141.210702</td>
      <td>104.200000</td>
      <td>170.153285</td>
      <td>184.459459</td>
    </tr>
    <tr>
      <th>Columbia Heights</th>
      <td>114.676471</td>
      <td>126.773196</td>
      <td>135.292994</td>
      <td>124.358621</td>
    </tr>
    <tr>
      <th>Dupont Circle</th>
      <td>146.858942</td>
      <td>130.709677</td>
      <td>179.574468</td>
      <td>139.348837</td>
    </tr>
    <tr>
      <th>Edgewood</th>
      <td>108.508197</td>
      <td>112.846154</td>
      <td>156.335616</td>
      <td>147.503226</td>
    </tr>
    <tr>
      <th>Kalorama Heights</th>
      <td>122.542254</td>
      <td>155.928571</td>
      <td>92.695652</td>
      <td>158.230769</td>
    </tr>
    <tr>
      <th>Shaw</th>
      <td>153.888889</td>
      <td>158.500000</td>
      <td>202.114754</td>
      <td>173.279070</td>
    </tr>
    <tr>
      <th>Union Station</th>
      <td>128.458204</td>
      <td>133.833333</td>
      <td>162.748571</td>
      <td>162.167702</td>
    </tr>
  </tbody>
</table>
</div>




```python
dxp.bar(data=df, orientation='h')
```




![png](https://raw.githubusercontent.com/dexplo/dexplot/gh-pages/images/output_107_0.png)



### Time series


```python
stocks = pd.read_csv('../data/stocks10.csv', parse_dates=['date'], index_col='date')
stocks.head()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>MSFT</th>
      <th>AAPL</th>
      <th>SLB</th>
      <th>AMZN</th>
      <th>TSLA</th>
      <th>XOM</th>
      <th>WMT</th>
      <th>T</th>
      <th>FB</th>
      <th>V</th>
    </tr>
    <tr>
      <th>date</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1999-10-25</th>
      <td>29.84</td>
      <td>2.32</td>
      <td>17.02</td>
      <td>82.75</td>
      <td>NaN</td>
      <td>21.45</td>
      <td>38.99</td>
      <td>16.78</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1999-10-26</th>
      <td>29.82</td>
      <td>2.34</td>
      <td>16.65</td>
      <td>81.25</td>
      <td>NaN</td>
      <td>20.89</td>
      <td>37.11</td>
      <td>17.28</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1999-10-27</th>
      <td>29.33</td>
      <td>2.38</td>
      <td>16.52</td>
      <td>75.94</td>
      <td>NaN</td>
      <td>20.80</td>
      <td>36.94</td>
      <td>18.27</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1999-10-28</th>
      <td>29.01</td>
      <td>2.43</td>
      <td>16.59</td>
      <td>71.00</td>
      <td>NaN</td>
      <td>21.19</td>
      <td>38.85</td>
      <td>19.79</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1999-10-29</th>
      <td>29.88</td>
      <td>2.50</td>
      <td>17.21</td>
      <td>70.62</td>
      <td>NaN</td>
      <td>21.47</td>
      <td>39.25</td>
      <td>20.00</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>




```python
dxp.line(data=stocks.head(500))
```




![png](https://raw.githubusercontent.com/dexplo/dexplot/gh-pages/images/output_110_0.png)


