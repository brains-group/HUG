#!/bin/bash

python main.py --data-dir ../KuaiRand-1K/data --epochs 20 --cache-dir ./cache/1K --model-type dual --device cuda:0 --kg-alignment 1

python main.py --data-dir ../KuaiRand-27K/data --epochs 20 --cache-dir ./cache/27K --model-type single --device cuda:0 --scale 27k

python main.py --data-dir ../KuaiRand-27K/data --epochs 20 --cache-dir ./cache/27K --model-type dual --device cuda:0 --kg-alignment 0 --scale 27k

python main.py --data-dir ../KuaiRand-27K/data --epochs 20 --cache-dir ./cache/27K --model-type dual --device cuda:0 --kg-alignment 1 --scale 27k
