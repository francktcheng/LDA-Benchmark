#!/bin/bash

#export CXX=icpc
#export CC=icc

{
mkdir -p debug
pushd debug
cmake .. -DCMAKE_BUILD_TYPE=Debug
popd
}

