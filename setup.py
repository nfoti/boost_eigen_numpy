import numpy as np

import os
import sys
import subprocess

from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext

args = sys.argv[1:]

# Make a `cleanall` rule to get rid of intermediate and library files
if "cleanall" in args:
    print "Deleting cython files..."
    # Note that shell=True should be OK because the command is constant.
    # Just in case the build directory was created by accident, delete it
    subprocess.Popen("rm -rf build", shell=True, executable="/bin/bash")
    subprocess.Popen("rm -rf *.c", shell=True, executable="/bin/bash")
    subprocess.Popen("rm -rf *.so", shell=True, executable="/bin/bash")

    # Now do a normal clean
    sys.argv[1] = "clean"

# We want to always use build_ext --inplace
if args.count("build_ext") > 0 and args.count("--inplace") == 0:
    sys.argv.insert(sys.argv.index("build_ext")+1, "--inplace")

# Only build for 64-bit target
os.environ['ARCHFLAGS'] = "-arch x86_64"


setup(name="np_test",
      cmdclass={'build_ext': build_ext},
      ext_modules=[
          Extension("np_test", ["np_test.cpp"],
          language='c++',
          include_dirs=[np.get_include(), '/usr/local/include/eigen3'],
          libraries=["boost_python"],
          extra_compile_args=['-std=c++11', '-stdlib=libc++'],
          extra_link_args=['-std=c++11', '-stdlib=libc++'])
      ])
