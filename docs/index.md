
# Dexplot

A Python library for making data visualizations.

The aim of dexplot is to make data visualization creation in Python more robust and straightforward. Dexplot is built on top of Matplotlib and accepts Pandas DataFrames as inputs.

## Goals

The primary goals for dexplot are:

* Maintain a very consistent API with as few functions as necessary to make the desired statistical plots
* Allow the user to tweak the plots without digging into matplotlib


## Installation

`pip install dexplot`

# Comparison with Seaborn

If you have used the seaborn library, then you should notice a lot of similarities. Much of Dexplot was inspired by Seaborn. Below is a list of the extra features in dexplot not found in seaborn

* The ability to graph relative frequency percentage and normalize over any number of variables
* Far fewer public functions
* No need for multiple functions to do the same thing
* Ability to make grids with a single function instead of having to use a higher level function like `catplot`
* Pandas `groupby` methods are available as strings
* Both x/y-labels and titles are automatically wrapped so that they don't overlap
* The figure size (plus several other options) and available to change without dipping down into matplotlib
* No new types like FacetGrid. Only matplotlib objects are returned
