#!/usr/bin/env bash
python run.py -c configs/mnist_svhn/concat.yaml
python run.py -c configs/mnist_svhn/set.yaml
