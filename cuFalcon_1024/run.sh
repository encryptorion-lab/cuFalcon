#!/bin/bash

mkdir -p build
cd build

cmake ..
cmake --build .

./cuFalcon1024_opt
./cuFalcon1024_tree
./cuFalcon1024_balance
