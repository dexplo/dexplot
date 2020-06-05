# Bugs

* x and y label position - not centered if 
  * ax.get_position() after fig.canvas.draw()
* size figure by number of bars/boxes/etc
* allow user to set by axes
* font size proportion to number of labels
* all text in figure proportional to number of items in figuer


# Upcoming Features
* parameter for wrapping text (default 10)

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
* kde
* heat
* hexplot
* mosaic