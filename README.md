
# Dexplot

A Python library for making data visualizations.

The current aim of Dexplot is to make data visualization creation in Python more robust and straightforward. Dexplot is built on top of Matplotlib and accepts Pandas DataFrames as inputs. 

## Installation

`pip install dexplot`

## Goals

The primary goals for Dexplot are:

* Maintain a very consistent API with as few functions as necessary to make the desired statistical plots
* Allow the user to tweak the plots without digging into Matplotlib


## Tidy Data from Pandas
Dexplot only accepts Pandas DataFrames as input for its plotting functions that are in "tidy" form. 

## Sample plots
Dexplot currently maintains two primary functions, `aggplot` which is used to aggregate data and `jointplot`, which is used to plot raw values from two variables against each other. `heatmap` is another function available that produces just a single heatmap.

`aggplot` can create five different kinds of plots.

* `bar`
* `line`
* `box`
* `hist`
* `kde`

`jointplot` can create four different kinds of plots

* `scatter`
* `line`
* `2D kde`
* `bar`

There are 7 primary parameters to `aggplot`:

* `agg` - Name of column to be aggregated. If it is a column with string/categorical values, then the counts or relative frequency percentage will be returned.
* `groupby` - Name of column whose unique values will form independent groups. This is used in a similar fashion as the `group by` SQL clause.
* `data` - The Pandas DataFrame
* `hue` - The name of the column to further group the data within a single plot
* `row` - The name of the column who's unique values split the data in to separate rows
* `col` - The name of the column who's unique values split the data in to separate columns
* `kind` - The kind of plot to create. One of the five strings from above.

`jointplot` uses `x` and `y` instead of `groupby` and `agg`.

### City of Houston Data

To get started, we will use City of Houston employee data collected from the year 2016. It contains public information from about 1500 employees and is located in Dexplot's GitHub repository.



```python
import pandas as pd
import dexplot as dxp
```


```python
emp = pd.read_csv('notebooks/data/employee.csv')
emp.head()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>title</th>
      <th>dept</th>
      <th>salary</th>
      <th>race</th>
      <th>gender</th>
      <th>experience</th>
      <th>experience_level</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>POLICE OFFICER</td>
      <td>Houston Police Department-HPD</td>
      <td>45279.0</td>
      <td>White</td>
      <td>Male</td>
      <td>1</td>
      <td>Novice</td>
    </tr>
    <tr>
      <th>1</th>
      <td>ENGINEER/OPERATOR</td>
      <td>Houston Fire Department (HFD)</td>
      <td>63166.0</td>
      <td>White</td>
      <td>Male</td>
      <td>34</td>
      <td>Veteran</td>
    </tr>
    <tr>
      <th>2</th>
      <td>SENIOR POLICE OFFICER</td>
      <td>Houston Police Department-HPD</td>
      <td>66614.0</td>
      <td>Black</td>
      <td>Male</td>
      <td>32</td>
      <td>Veteran</td>
    </tr>
    <tr>
      <th>3</th>
      <td>ENGINEER</td>
      <td>Public Works &amp; Engineering-PWE</td>
      <td>71680.0</td>
      <td>Asian</td>
      <td>Male</td>
      <td>4</td>
      <td>Novice</td>
    </tr>
    <tr>
      <th>4</th>
      <td>CARPENTER</td>
      <td>Houston Airport System (HAS)</td>
      <td>42390.0</td>
      <td>White</td>
      <td>Male</td>
      <td>3</td>
      <td>Novice</td>
    </tr>
  </tbody>
</table>
</div>



### Plotting the average salary by department
The `agg` parameter is very important and is what will be aggregated (summarized by a single point statistic, like the mean or median). It is the first parameter and only parameter you must specify (besides `data`). If this column is numeric, then by default, the mean of it will be calculated. Here, we specify the `groupby` parameter, who's unique values form the independent groups and label the x-axis.


```python
dxp.aggplot(agg='salary', groupby='dept', data=emp)
```




    <matplotlib.axes._subplots.AxesSubplot at 0x1190d2128>




![png](images/output_6_1.png)


### Make horizontal with the `orient` parameter
The `orient` parameter controls whether the plot will be horizontal or vertical. By default it is set to `'h'`.


```python
dxp.aggplot(agg='salary', groupby='dept', data=emp, orient='h')
```




    <matplotlib.axes._subplots.AxesSubplot at 0x1192f7160>




![png](images/output_8_1.png)


### Controlling the figure size
One of the goals of Dexplot is to not have you dip down into the details of Matplotlib. We can use the `figsize` parameter to change the size of our plot.


```python
dxp.aggplot(agg='salary', groupby='dept', data=emp, orient='h', figsize=(8, 4))
```




    <matplotlib.axes._subplots.AxesSubplot at 0x119377b00>




![png](images/output_10_1.png)


### Adding another dimension with `hue`
The `hue` parameter may be used to further subdivide each unique value in the `groupby` column. Notice that long tick labels are automatically wrapped.


```python
dxp.aggplot(agg='salary', groupby='dept', data=emp, hue='gender')
```




    <matplotlib.axes._subplots.AxesSubplot at 0x1193b1208>




![png](images/output_12_1.png)


## Aggregating a String/Categorical column
It is possible to use a string/categorical column as the aggregating variable. In this instance, the counts of the unique values of that column will be returned. Because this is already doing a `groupby`, you cannot specify a `groupby` column in this instance. Let's get the count of employees by race.


```python
dxp.aggplot(agg='race', data=emp, figsize=(8, 4))
```




    <matplotlib.axes._subplots.AxesSubplot at 0x119377cf8>




![png](images/output_14_1.png)


## Using `hue` with a String/Categorical column
Using a `groupby` is not allowed when a string/categorical column is being aggregated. But, we can still sub-divide the groups further by specifying `hue`.


```python
dxp.aggplot(agg='race', data=emp, hue='dept')
```




    <matplotlib.axes._subplots.AxesSubplot at 0x11b7d1588>




![png](images/output_16_1.png)


## Getting the relative frequency percentage with `normalize`
It is possible to turn the raw counts into percentages by passing a value to `normalize`. Let's find the percentage of all employees by race.


```python
dxp.aggplot(agg='race', data=emp, normalize='all', figsize=(8, 4))
```




    <matplotlib.axes._subplots.AxesSubplot at 0x11b7f1e10>




![png](images/output_18_1.png)


## You can normalize over any variable
The parameter `normalize` can be one of the values passed to the parameters `'agg'`, `'hue'`, `'row'`, `'col'`, or a tuple containing any number of these or `'all'`. For instance, in the following plot, you can normalize by either `race` or `dept`.


```python
dxp.aggplot(agg='race', data=emp, hue='dept', normalize='race')
```




    <matplotlib.axes._subplots.AxesSubplot at 0x11bb0d048>




![png](images/output_20_1.png)


## Data normalized by race
As you can see, the data was normalized by race. For example, from the graph, we can tell that about 30% of black employees were members of the police department. We can also normalize by department. From the graph, about 10% of the Health & Human Services employees were Asian.


```python
dxp.aggplot(agg='race', data=emp, hue='dept', normalize='dept')
```




    <matplotlib.axes._subplots.AxesSubplot at 0x11bf4f0b8>




![png](images/output_22_1.png)


## Stacked Bar Plots
All bar plots that have use the `hue` variable, can be stacked. Here, we stack the maximum salary by department grouped by race.


```python
dxp.aggplot(agg='salary', data=emp, hue='dept', groupby='race', aggfunc='max', stacked=True)
```




    <matplotlib.axes._subplots.AxesSubplot at 0x11b7d1208>




![png](images/output_24_1.png)


## Stacking counts
The raw counts of each department by experience level are stacked here.


```python
dxp.aggplot(agg='experience_level', data=emp, hue='dept', aggfunc='max', stacked=True)
```




    <matplotlib.axes._subplots.AxesSubplot at 0x11c41b0f0>




![png](images/output_26_1.png)


## Stacking relative frequencies
The relative frequencies of each department by each race and experience level.


```python
dxp.aggplot(agg='experience_level', data=emp, hue='dept', row='race', 
            normalize=('race', 'experience_level'), wrap=3, stacked=True)
```




    (<Figure size 864x720 with 5 Axes>,)




![png](images/output_28_1.png)


# Other kinds of plots `line`, `box`, `hist`, and `kde`
`aggplot` is capable of making four other kinds of plots. The `line` plot is very similar to the bar plot but simply connects the values together. Let's go back to a numeric column and calculate the **median** salary by department across each gender.


```python
dxp.aggplot(agg='salary', data=emp, groupby='dept', hue='gender', kind='line', aggfunc='median')
```




    <matplotlib.axes._subplots.AxesSubplot at 0x11c994eb8>




![png](images/output_30_1.png)


## `aggfunc` can take any string value that Pandas can
There are more than a dozen string values that `aggfunc` can take. These are simply passed to Pandas `groupby` method which does the aggregation.

## All plots can be both vertical and horizontal
We can rotate all plots with `orient`. 


```python
dxp.aggplot(agg='salary', data=emp, groupby='dept', hue='gender', kind='line', aggfunc='median', orient='h')
```




    <matplotlib.axes._subplots.AxesSubplot at 0x11cd2ad68>




![png](images/output_32_1.png)


## Boxplots
Here is the same data plotted as a box plot. This isn't actually an aggregation, so the `aggfunc` parameter is meaningless here. Instead, all the values of the particular group are plotted.


```python
dxp.aggplot(agg='salary', data=emp, groupby='dept', hue='gender', kind='box', orient='h')
```




    <matplotlib.axes._subplots.AxesSubplot at 0x11d0c27f0>




![png](images/output_34_1.png)


## Histograms and KDE's
As with boxplots, histograms and kdes do not function with `aggfunc` as they aren't aggregating but simply displaying all the data for us. Also, it is not possible to use both `groupby` and `agg` with these plots.


```python
dxp.aggplot(agg='salary', data=emp, groupby='dept', kind='hist', orient='v')
```




    <matplotlib.axes._subplots.AxesSubplot at 0x11d37c780>




![png](images/output_36_1.png)



```python
dxp.aggplot(agg='salary', data=emp, groupby='dept', kind='kde', orient='v')
```




    <matplotlib.axes._subplots.AxesSubplot at 0x11d5ee748>




![png](images/output_37_1.png)


## Splitting into separate plots
The `row` and `col` parameters can be used to split the data into separate plots. Each unique value of `row` or `col` will create a new plot. A one-item tuple consisting of the entire Figure is returned.


```python
dxp.aggplot(agg='salary', data=emp, groupby='experience_level', kind='kde', orient='v', row='dept')
```




    (<Figure size 720x1152 with 6 Axes>,)




![png](images/output_39_1.png)


## Use the `wrap` parameter to make new rows/columns
Set the `wrap` parameter to an integer to determine where a new row/column will be formed.


```python
dxp.aggplot(agg='salary', data=emp, groupby='experience_level', kind='box', orient='v', row='dept', wrap=3)
```




    (<Figure size 864x720 with 6 Axes>,)




![png](images/output_41_1.png)


## `wrap` works for both `row` or `col`


```python
dxp.aggplot(agg='salary', data=emp, groupby='experience_level', kind='box', orient='v', col='dept', wrap=5)
```




    (<Figure size 1296x576 with 6 Axes>,)




![png](images/output_43_1.png)


# Use both `row` and `col` for a entire grid
By using both `row` and `col`, you can maximize the number of variables you divide the data into.


```python
dxp.aggplot(agg='salary', data=emp, groupby='gender', kind='kde', row='dept', col='experience_level')
```




    (<Figure size 1008x1152 with 18 Axes>,)




![png](images/output_45_1.png)


# Normalize by more than one variable

Before, we normalized by just a single variable. It is possible to normalize by multiple variables with a tuple. Here we normalize by department and gender. Adding up all the blue bars for each department should add to 1.


```python
dxp.aggplot(agg='dept', data=emp, hue='gender', kind='bar', row='race', normalize=('dept', 'gender'))
```




    (<Figure size 720x1008 with 5 Axes>,)




![png](images/output_47_1.png)


## Normalize by three variables
Here we normalize by race, experience level, and gender. Each set of orange/blue bars within each plot will add to 1.


```python
dxp.aggplot(agg='dept', data=emp, hue='gender', kind='bar', row='race', 
            col='experience_level', normalize=('gender', 'experience_level', 'race'), orient='h')
```




    (<Figure size 1008x1008 with 15 Axes>,)




![png](images/output_49_1.png)


# Joint Plots
`joinplot` works differently than `aggplot` in that no aggregation takes place. It plots the raw values between two variables. It can split the data into groups or new plots with `hue`, `row`, and `col`. The default plot is a scatter plot, but you can also provide a string value to the `kind` parameter to make line, kde, or bar plots. 


```python
dxp.jointplot('experience', 'salary', data=emp)
```




    <matplotlib.axes._subplots.AxesSubplot at 0x120b9af60>




![png](images/output_51_1.png)


## Split data in the same plot with `hue`


```python
dxp.jointplot('experience', 'salary', data=emp, hue='gender')
```




    <matplotlib.axes._subplots.AxesSubplot at 0x12171e6d8>




![png](images/output_53_1.png)


## Plot a regression line by setting `fit_reg` equal to `True`
By default it plots the 95% confidence interval around the mean.


```python
dxp.jointplot('experience', 'salary', data=emp, hue='gender', fit_reg=True)
```




    <matplotlib.axes._subplots.AxesSubplot at 0x1218e6c18>




![png](images/output_55_1.png)


## Further split the data into separate plots with `row` and `col`


```python
dxp.jointplot('experience', 'salary', data=emp, hue='gender', row='dept', wrap=3, fit_reg=True)
```




    (<Figure size 864x720 with 6 Axes>,)




![png](images/output_57_1.png)



```python
dxp.jointplot('experience', 'salary', data=emp, hue='gender', row='dept', col='experience_level')
```




    (<Figure size 1008x1152 with 18 Axes>,)




![png](images/output_58_1.png)


## Use the `s` parameter to change the size of each marker
Let `s` equal a column name containing numeric values to set each marker size individually. We need to create another numeric variable first since the dataset only contains two.


```python
import numpy as np
emp['num'] = np.random.randint(10, 300, len(emp))
```


```python
dxp.jointplot('experience', 'salary', data=emp, hue='gender', row='dept', wrap=3, s='num')
```




    (<Figure size 864x720 with 6 Axes>,)




![png](images/output_61_1.png)


# Line Plots


```python
df_stocks = pd.read_csv('notebooks/data/stocks.csv', parse_dates=['date'])
df_stocks.head()
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
      <th>date</th>
      <th>close</th>
      <th>symbol</th>
      <th>percent_gain</th>
      <th>year</th>
      <th>month</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2013-10-07</td>
      <td>63.7997</td>
      <td>aapl</td>
      <td>0.0</td>
      <td>2013</td>
      <td>10</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2013-10-07</td>
      <td>96.6579</td>
      <td>cvx</td>
      <td>0.0</td>
      <td>2013</td>
      <td>10</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2013-10-07</td>
      <td>35.0541</td>
      <td>txn</td>
      <td>0.0</td>
      <td>2013</td>
      <td>10</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2013-10-07</td>
      <td>19.4912</td>
      <td>csco</td>
      <td>0.0</td>
      <td>2013</td>
      <td>10</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2013-10-07</td>
      <td>310.0300</td>
      <td>amzn</td>
      <td>0.0</td>
      <td>2013</td>
      <td>10</td>
    </tr>
  </tbody>
</table>
</div>




```python
dxp.jointplot(x='date', y='percent_gain', data=df_stocks, hue='symbol', kind='line')
```




    <matplotlib.axes._subplots.AxesSubplot at 0x121ad34a8>




![png](images/output_64_1.png)



```python
dxp.jointplot(x='date', y='percent_gain', data=df_stocks, kind='line', hue='symbol', row='year', wrap=3,
             sharex=False, sharey=False)
```




    (<Figure size 864x720 with 6 Axes>,)




![png](images/output_65_1.png)


# 2D KDE Plots


```python
dxp.jointplot('experience', 'salary', data=emp, kind='kde')
```




    <matplotlib.axes._subplots.AxesSubplot at 0x12290e898>




![png](images/output_67_1.png)



```python
dxp.jointplot('experience', 'salary', data=emp, kind='kde', row='dept', col='gender', sharex=False, sharey=False)
```




    (<Figure size 864x1152 with 12 Axes>,)




![png](images/output_68_1.png)


# Bar Plots for aggregated data

If your data is already aggregated, you can use `jointplot` to plot it.


```python
df = emp.groupby('dept').agg({'salary':'mean'}).reset_index()
df
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
      <th>dept</th>
      <th>salary</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Health &amp; Human Services</td>
      <td>51324.980583</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Houston Airport System (HAS)</td>
      <td>53990.368932</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Houston Fire Department (HFD)</td>
      <td>59960.441096</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Houston Police Department-HPD</td>
      <td>60428.745614</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Parks &amp; Recreation</td>
      <td>39426.150943</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Public Works &amp; Engineering-PWE</td>
      <td>50207.806452</td>
    </tr>
  </tbody>
</table>
</div>




```python
dxp.jointplot('dept', 'salary', data=df, kind='bar')
```




    <matplotlib.axes._subplots.AxesSubplot at 0x12512fe10>




![png](images/output_71_1.png)


# Heatmaps

Heatmaps work with both tidy and aggregated data. 

## Frequency
When working with tidy data, passing it just `x` and `y` will plot the frequency of occurrences for all of the combinations of their unique values. Place the count as text in the box with `annot`. The default formatting has two decimals.


```python
dxp.heatmap(x='dept', y='race', data=emp, annot=True, fmt='.0f')
```




    (<Figure size 720x576 with 2 Axes>,)




![png](images/output_73_1.png)


## Aggregating a variable with heatmaps
Set the `agg` parameter to aggregate a particular variable. Choose how you will aggregate with the `aggfunc` parameter, which takes any string that Pandas can. The default it the mean.


```python
dxp.heatmap(x='dept', y='race', agg='salary', aggfunc='max', data=emp, annot=True, fmt='.0f')
```




    (<Figure size 720x576 with 2 Axes>,)




![png](images/output_75_1.png)


## Normalize heatmaps by row, column, or all data
You can normalize the data by row, column, or all data with. Use the string name of the column for row and column normalization. Below we find the total percentage of all combined years of experience normalized by race. For example, of all the total years of experience for White employees, 89% of those years are male.


```python
dxp.heatmap(x='race', y='gender', agg='experience', aggfunc='sum', 
            data=emp, annot=True, fmt='.3f', normalize='race')
```




    (<Figure size 720x576 with 2 Axes>,)




![png](images/output_77_1.png)



```python
dxp.heatmap(x='race', y='dept', agg='experience', aggfunc='sum', 
            data=emp, annot=True, fmt='.3f', normalize='race', corr=True)
```




    (<Figure size 720x576 with 2 Axes>,)




![png](images/output_78_1.png)


## Heatmaps without aggregating data
If you pass just the DataFrame into `heatmap` then those raw values will be used to create the colors. Here we plot some random numbers from a normal distribution.


```python
df = pd.DataFrame(np.random.randn(10, 5), columns=list('abcde'))
fig, = dxp.heatmap(data=df, annot=True)
```


![png](images/output_80_0.png)


## Find correlations by setting `corr` equal to `True`

Setting the `corr` parameter to True computes the pairwise correlation matrix between the columns. Any string columns are discarded. Below, we use the popular Kaggle housing dataset.


```python
housing = pd.read_csv('notebooks/data/housing.csv')
fig, = dxp.heatmap(data=housing, corr=True, figsize=(16, 16))
```


![png](images/output_82_0.png)


# Comparison with Seaborn
If you have used the Seaborn library, then you should notice a lot of similarities. Much of Dexplot was inspired by Seaborn. Below is a list of the extra features in Dexplot not found in Seaborn

* The ability to graph relative frequency percentage and normalize over any number of variables
* Far fewer public functions. Only two at the moment
* No need for multiple functions to do the same thing. Seaborn has both `countplot` and `barplot`
* Ability to make grids with a single function instead of having to use a higher level function like `catplot`
* Pandas `groupby` methods are available as strings
* Both x/y-labels and titles are automatically wrapped so that they don't overlap
* The figure size (plus several other options) and available to change without dipping down into matplotlib
* No new types like FacetGrid. Only matplotlib objects are returned
