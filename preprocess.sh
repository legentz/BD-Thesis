#!/bin/sh
# Original file from Shimaoka et al. (2017)

set -o errexit
set -o nounset

echo "Downloading corpus"
wget http://www.cl.ecei.tohoku.ac.jp/~shimaoka/corpus.zip
unzip corpus.zip
rm corpus.zip

echo "Downloading word embeddings..."
wget http://nlp.stanford.edu/data/glove.840B.300d.zip
unzip glove.840B.300d.zip
rm glove.840B.300d.zip
mv glove.840B.300d.txt resource/

echo "Preprocessing (creating ids for words, features, and labels)"

echo "Wiki"
mkdir ./resource/Wiki/
python ./resource/create_X2id.py corpus/Wiki/all.txt resource/Wiki/word2id_figer.txt resource/Wiki/feature2id_figer.txt resource/Wiki/label2id_figer.txt

echo "Preprocessing (creating dictionaries)"
mkdir ./data

echo "Wiki"
mkdir ./data/Wiki
python create_dicts.py resource/Wiki/word2id_figer.txt resource/Wiki/feature2id_figer.txt resource/Wiki/label2id_figer.txt  resource/glove.840B.300d.txt data/Wiki/dicts_figer.pkl

echo "Preprocessing (creating datasets)"

echo "Wiki"
python create_dataset.py data/Wiki/dicts_figer.pkl corpus/Wiki/train.txt data/Wiki/train_figer.pkl
python create_dataset.py data/Wiki/dicts_figer.pkl corpus/Wiki/dev.txt data/Wiki/dev_figer.pkl
python create_dataset.py data/Wiki/dicts_figer.pkl corpus/Wiki/test.txt data/Wiki/test_figer.pkl

