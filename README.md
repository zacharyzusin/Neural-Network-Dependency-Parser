# Neural Network Dependency Parser

A neural network-based dependency parser implemented in PyTorch that predicts transitions for an arc-standard dependency parser. This project implements a simplified version of the parser described in Chen & Manning (2014).

## Overview

This dependency parser uses a feed-forward neural network to predict parser transitions based on the current state, including words on the stack and buffer. The parser implements three transitions:
- Shift
- Left Arc
- Right Arc

The neural network takes as input the representations of the top three words on the stack and the next three words in the buffer, and outputs both the transition type and dependency relation label.

## Prerequisites

- PyTorch
- NumPy

## Project Structure

- `conll_reader.py`: Data structures for dependency trees and functionality to read/write CoNLL-X format
- `get_vocab.py`: Extracts words and POS tags vocabulary from training data
- `extract_training_data.py`: Creates input/output matrices for neural network training
- `train_model.py`: Implements and trains the neural network model
- `decoder.py`: Uses the trained model to parse input sentences
- `evaluate.py`: Evaluates parser output against gold standard dependencies

## Data Format

The project uses the CoNLL-X format, where each word in a sentence is represented by a line with the following tab-separated fields:
1. Word ID (starting at 1)
2. Word form
3. Lemma
4. Universal POS tag
5. Corpus-specific POS tag
6. Features (unused)
7. Head word ID
8. Dependency relation
9. Deps (unused)
10. Misc annotations (unused)

## Model Architecture

The neural network consists of:
- An embedding layer (128 dimensions)
- A hidden layer with 128 units and ReLU activation
- An output layer with 91 units (45 dependency relations × 2 transitions + 1 shift)
- Log softmax activation on the output layer

## Usage

### 1. Generate Vocabulary
```bash
python get_vocab.py data/train.conll data/words.vocab data/pos.vocab
```

### 2. Extract Training Data
```bash
python extract_training_data.py data/train.conll data/input_train.npy data/target_train.npy
python extract_training_data.py data/dev.conll data/input_dev.npy data/target_dev.npy
```

### 3. Train the Model
```bash
python train_model.py data/input_train.npy data/target_train.npy data/model.pt
```

### 4. Parse Sentences
```bash
python decoder.py data/model.pt data/dev.conll
```

### 5. Evaluate Parser Performance
```bash
python evaluate.py data/model.pt data/dev.conll
```

## Reference

This implementation is based on:

Chen, D., & Manning, C. (2014). A Fast and Accurate Dependency Parser using Neural Networks. In Proceedings of the 2014 Conference on Empirical Methods in Natural Language Processing (EMNLP).
