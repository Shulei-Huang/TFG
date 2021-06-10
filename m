#!/bin/bash

rm -f libnvsample_cudaprocess.so
rm -f nvsample_cudaprocess.o

make || exit
