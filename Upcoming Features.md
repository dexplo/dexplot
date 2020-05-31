# Upcoming Features

* [ ] kde with annotations, allow for binning
* [ ] scatter with kde
* [ ] allow kde and historgrams plots to be grouped
* [ ] make size parameter in scatterplot be relative. Don't use literal value unless explicitly told to do so
* [ ] use a categorical variable to size scatter plot
* [ ] allow user to specify a specific matplotlib axes
* [ ] add interaction with ipywidgets
* [ ] stacked area plot
* [ ] rolling averages for line plots
* [ ] annotate aggplot with counts of groups
* [ ] add parameter `bins` to bin numeric x for agg plot
* [ ] mosaic plot
* [ ] option to add counts to all aggregate plots
* [ ] have `agg` and `y` for `aggplot`/`jointplot` take a list of variables. Have option to split the plots
        into new Axes

## Redeign API

* df.agg.plotname(data, agg, aggfunc, groupby, groupby2, groupby_row, groupby_col, data, orient, 
                  sort,wrap, figsize, title, sharex, sharey, xlabel, ylabel, xlim,
                  ylim, xscale, yscale, kwargs)
        * line - c, cmap, lw, ls, marker, ms, mec, mew
        * bar - stacked
        * count - no agg, line - boolean to use line instead
        * scatter - marker
        * box
        * hist
        * kde

* df.raw.plotname(x, y)


### Next idea

* df.long
* df.wide(x, y)
  * x will be index
  * y will be all columns by default
  * can be explicit with df.wide.scatter(x='', y='')

dxp.long.bar(x='dept', y='salary', data=df, groupby='dept', groupby2='race', aggfunc='mean')
dxp.long.scatter(x='exp', y='salary', groupby='dept') no aggregation needed

### wide

Used when pivot table already created

dxp.wide.bar() - default - plot all columns, use index as x, values as y
dxp.wide.bar(x='dept', y='salary', groupby='race')

### One entry point again

simpler

could have main entry point as 
dxp.bar
dxp.wide.bar for wide data

How to tell direction? Cant tell without groupby or orientation
dxp.bar(x='dept', y='salary', data=df)

groupby would only be for aggregation
if and only if groupby supplied, must have aggfunc
split, row, col when supplied separately must not be given aggfunc
dxp.scatter(x='exp', y='salary') groupby, split, row, col

Distribution functions can't really have groupby/aggfunc
dxp.hist(val, data, split, row, col) - only one without x and y
dxp.kde(x, y, data, split, row, col) - 2d KDEs
dxp.box(x, y, data, split, row, col) boxplots can have x and y

Other plots
* mosaic