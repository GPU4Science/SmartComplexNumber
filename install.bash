#!/bin/bash

# Update Pybind11 submodule
git submodule init
git submodule update

if [ ! -d "build" ]; then
    mkdir build
fi

cd build
cmake ..
make -j4
export PYTHONPATH="$PWD:$PYTHONPATH"
cd ..
