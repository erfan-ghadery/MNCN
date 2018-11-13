# Multilingual Ngram-based Convolutional Network (MNCN)

This is the code for the paper "MNCN: A Multilingual Ngram-based Convolutional Network for Aspect Category Detection in Online Reviews".

## Installation

### PyTorch

The code is based on PyTorch. You can find installation instructions [here](http://pytorch.org/).

### Dependencies

The code is written in Python 3.6 and pytorch 0.4.1. Its dependencies are summarized in the file ```requirements.txt```. You can install these dependencies like this:

```
pip install -r requirements.txt
```

### Code reference

[MNCN](address)

### word embeddings

MUSE: You can find the pre-trained multilingual word embedding [here](https://github.com/facebookresearch/MUSE),
and place them in the 'embeddings' directory.

### Dataset

You can find the dataset in the semeval 2016 website [here](http://alt.qcri.org/semeval2016/task5/index.php?id=data-and-tools). Copy the dataset for each language in the directory 'data'

### Running
You can run the model by changing the train and test languages in main.py file and then running the following code:
```
python main.py
```
