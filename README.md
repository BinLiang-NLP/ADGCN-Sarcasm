# Sarcasm_ADGCN

# Introduction
This repository was used in our paper:  

**Affective Dependency Graph for Sarcasm Detection**
<br>
Chen Lou<sup>\*</sup>, Bin Liang<sup>\*</sup>, Lin Gui, Yulan He, Yixue Dang, Ruifeng Xu. *Proceedings of SIGIR 2021*

<sup>\*</sup> The first two authors contribute equally to this work.

Please cite our paper and kindly give a star for this repository if you use this code.

## Requirements
- pytorch >= 0.4.0
- numpy >= 1.13.3
- sklearn
- python 3.6 / 3.7
- transformers

## Pretrained Models
Download glove.42B.300d.zip from [glove website](https://nlp.stanford.edu/projects/glove/) and unzip in project root path.

## Usage
* Install [SpaCy](https://spacy.io/) package and language models with
```bash
pip3 install spacy
```
and
```bash
python3 -m spacy download en
```

## Training
* Train with command, optional arguments could be found in [train.py](/train.py)
```bash
python3 train.py 
```





