#!/usr/bin/env bash

PYTHON=${PYTHON:-"python"}

echo "Building bbox overlap ops..."
cd bbox_overlap
if [ -d "build" ]; then
    rm -r build
fi
$PYTHON setup.py build_ext --inplace

cd ..

echo "Building geometry ops..."
cd geometry
if [ -d "build" ]; then
    rm -r build
fi
$PYTHON setup.py build_ext --inplace

cd ..