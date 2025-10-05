#!/bin/bash

mkdir -p build
cd build

cmake ..
cmake --build .

./cuFalcon512_opt
./cuFalcon512_tree
./cuFalcon512_balance
