#!/usr/bin/env bash
python run.py -c configs/mnist_svhn/poe_mvae.yaml --resume
python run.py -c configs/mnist_svhn/poe_vaevae.yaml
python run.py -c configs/mnist_svhn/set.yaml
