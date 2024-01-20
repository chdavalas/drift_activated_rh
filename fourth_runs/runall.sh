#!/bin/bash
cd mnist-exp3/mnist-exp3-buf50/
python3 drift.py | tee -a logs.txt
cd ../../
cd cifar10-exp3/cifar10-exp3-buf50
python3 drift.py | tee -a logs.txt
cd ../../
cd cifar10-exp3/cifar10-exp3-buf500
python3 drift.py | tee -a logs.txt

