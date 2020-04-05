# MimicProp

MimicProp(Mimicking Propagation) is a

PAPER_CITATION

## Abstract

Lexicon-based methods and word embeddings are the two widely used approaches for analyzing texts in social media. The choice of an approach can have a significant impact on the reliability of the text analysis. For example, lexicons provide manually curated, domain-specific attributes about a limited set of words, while word embeddings learn to encode some loose semantic interpretations for a much broader set of words. Text analysis can benefit from a representation that offers both the broad coverage of word embeddings and the do-main knowledge of lexicons. This paper presents MimicProp,a new graph-mode method that learns a lexicon-aligned word embedding. Our approach improves over prior graph-based methods in terms of its interpret ability (i.e., lexicon attribute scan be recovered) and generalizability (i.e., new words can be learned to incorporate lexicon knowledge). It also effectively improves the performance of downstream analysis applications, such as text classification.

## Dependency

Please install the required packages by ```pip3 install -r requirements.txt```

## Parameters

```m```: the number of connected nodes for lexicon words

```alpha```: the hyper-parameter balancing between lexicon/semantic in learning

```k```: the number of nodes to mimic 

```pre```: the location of the pretrained semantic embedding. this should be a pickled file (created by pickle package of python) of a python dictionary of: {WORD(str): EMBEDDING(1-d numpy array)}

```lex```: the location of the lexicon. this should be a pickled file (created by pickle package of python) of a python dictionary of: {WORD(str): SCORE(float)}

```voc```: the full vocabulary desired to run MimicProp. this should be a pickled file (created by pickle package of python) of a python dictionary of: {WORD(str): FREQUENCY(int)}

```name```: the name of the run. this is used in names of the stored files in caches and outputs. by default it is 'Untitled'

## Usage

### Run with your data

Please prepare the required data (see ```pre```, ```lex```, and ```voc``` above) in pickle format, and run the code with:
```python3 run_mimicprop.py --m YOUR_M --alpha YOUR_ALPHA --k YOUR_K --pre LOC_OF_PRE --lex LOC_OF_LEX --voc LOC_OF_VOC [--name NAME]```

### Run with our data

You can find our dataset at: UPLOAD_THE_DATA_TO_GOOGLE
Please download it to the root folder of this project, and run: 
```python3 run_mimicprop.py```

### Output

The output will be in 'output', with the name 'trained_*_final_emb.pkl'.

## Citation

If you make use of this code, please kindly cite our paper:

PAPER_CITATION
