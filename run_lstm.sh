#!/bin/bash
# Define the two lists
#names=('sst2' 'pubmed' 'trec' 'cardio' 'bbc')
augs=('aeda' 'backtranslation' 'charswap' 'checklist' 'clare' 'deletion' 'eda' 'embedding' 'wordnet')
percents=('10' '20' '50' '100')
examples=('1' '2' '4')
names=('cardio')



for name in "${names[@]}"; do
  for aug in "${augs[@]}"; do
    for percent in "${percents[@]}"; do
      for example in "${examples[@]}"; do
        python run_lstm_single.py --name "$name" --percent "$percent" --aug "$aug" --example "$example"
      done
    done
  done
done

