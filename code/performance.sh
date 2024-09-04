#!/bin/bash

datasets=('yelp2018' 'amazon-book' 'ml-20m')
levels=('drop' 'cross' 'prune')

for dataset in "${datasets[@]}"; do
  for level in "${levels[@]}"; do
    python main.py --dataset="$dataset" --model='hcmkr' --contrast_level="$level"
  done
done
