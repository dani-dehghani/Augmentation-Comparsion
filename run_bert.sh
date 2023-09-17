#!/bin/bash
# Define the two lists
#names=('sst2' 'pubmed' 'trec' 'cardio' 'bbc') #pc12740
names=('cr' 'agnews' 'subj' 'pc' 'yelp') #pc12739
augs=('aeda' 'backtranslation' 'charswap' 'checklist' 'clare' 'deletion' 'eda' 'embedding' 'wordnet')
percents=('10' '20' '50' '100')
examples=('1' '2' '4')



for name in "${names[@]}"; do
  for aug in "${augs[@]}"; do
    for percent in "${percents[@]}"; do
      for example in "${examples[@]}"; do
        python run_bert_single.py --name "$name" --percent "$percent" --aug "$aug" --example "$example"
      done
    done
  done
done

