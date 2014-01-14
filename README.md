Using Eigen and Boost::Python
=============================

This is an example of wrapping some code that uses the Eigen matrix library in
order to call it from Python.  Additionally, it shows how to pass complex
valued data from Python to Eigen and vice versa.

You can build the code in two ways:

1. Using distutils by executing
    `python setup.py build_ext --inplace`
   which will build `np_test.so` in the current directory.

2. Using cmake
    `cd build; cmake ..; make`
   which will build `np_test.so` in the build directory.


TODO
----

* Write tests for list code and whatnot.
