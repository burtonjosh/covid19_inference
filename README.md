# covid19_inference
MCMC methods to infer parameters for doubling rates of the spread of COVID-19

## File structure

- `/src/` contains the main Python module for this repository, `covid_inference.py`. It contains the model classes and MCMC algorithms.
- `/test/` contains the tests for the functions in `covid_inference.py`, and the Jupyter notebook `data_inference_example.ipynb`, which illustrates how to use the code.

## Dependencies (tested on Ubuntu 18.04):

- nosetests
- matplotlib
- numpy
- seaborn
- pandas

## Running the code

nosetests are used for the testing infrastructure to run all code in this project. Code can be run using the nostests command like this:

~~~
cd ./test
nosetests -s test_inference.py
~~~

This will run all functions in `test/test_inference.py` whose name starts with `test_`, and it is possible to toggle whether individual functions are run by renaming them, for example by replacing `test_` with `xest_`.
