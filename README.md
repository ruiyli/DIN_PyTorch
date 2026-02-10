# Deep Interest Network (DIN) - PyTorch Implementation

PyTorch implementation of **Deep Interest Network for Click-Through Rate Prediction** (DIN), published in KDD 2018 by Alibaba Group.

[![Paper](https://img.shields.io/badge/Paper-KDD%202018-blue)](https://arxiv.org/abs/1706.06978)
[![Python](https://img.shields.io/badge/Python-3.7+-brightgreen.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.0+-orange.svg)](https://pytorch.org/)

## 📄 Paper

**Deep Interest Network for Click-Through Rate Prediction**
Guorui Zhou, Chengru Song, Xiaoqiang Zhu, Ying Fan, Han Zhu, Xiao Ma, Yanghui Yan, Junqi Jin, Han Li, Kun Gai
*KDD 2018*

## 🎯 Overview

DIN (Deep Interest Network) is a click-through rate (CTR) prediction model designed for e-commerce recommendation systems. The key innovation is the **local activation unit** that uses attention mechanism to adaptively learn user interests from historical behaviors according to the candidate advertisement.

### Key Features

- ✨ **Attention Mechanism**: Dynamically activates relevant user historical behaviors based on candidate ads
- 🎲 **Dice Activation**: Data-adaptive activation function tailored for CTR prediction
- 🚀 **High Performance**: Significantly improves CTR prediction accuracy in production systems
- 📊 **Scalable**: Handles millions of users and items efficiently

## 🏗️ Model Architecture

```
Input Features
  ├─ User ID
  ├─ Candidate Item (ID + Category)
  └─ Historical Behaviors (Item Sequence + Category Sequence)
       ↓
  Embedding Layer
       ↓
  Attention Layer (Local Activation Unit)
       ├─ Computes attention scores between candidate and history
       └─ Outputs weighted user interest representation
       ↓
  Feature Concatenation
       [user_emb, item_emb, hist_sum, item×hist, attention_output]
       ↓
  MLP (with Dice activation)
       ↓
  Softmax → CTR Prediction
```

### Core Components

1. **Embedding Layer** ([embedding.py](embedding.py))
   - Converts sparse IDs to dense vectors
   - Xavier initialization

2. **Attention Layer** ([attention.py](attention.py))
   - **Local Activation Unit**: Computes attention scores
   - Feature engineering: `[query, behavior, query-behavior, query×behavior]`
   - Masked softmax for variable-length sequences

3. **Dice Activation** ([dice.py](dice.py))
   - Data-adaptive activation function
   - Formula: `f(x) = [α + (1-α)·sigmoid(normalize(x))] · x`

4. **FC Layer** ([fc.py](fc.py))
   - Configurable fully connected layer
   - Supports BatchNorm, Dropout, multiple activations (ReLU, PReLU, Dice)

## 📁 Project Structure

```
DIN/
├── README.md                  # This file
├── model.py                   # DIN model implementation
├── attention.py               # Attention mechanism (Local Activation Unit)
├── embedding.py               # Embedding layer
├── dice.py                    # Dice activation function
├── fc.py                      # Fully connected layer
├── trainer.py                 # Training and evaluation script
├── data_iterator.py           # Data loading and preprocessing
├── utils.py                   # Utility functions (AUC calculation, etc.)
└── shuffle.py                 # Data shuffling utilities
```

## 📊 Dataset

### Download Training Data

Due to the large file size, the training data is hosted on **Google Drive**:

**📥 [Download Dataset from Google Drive](https://drive.google.com/drive/folders/1bzJthjKVcksFGADdaIR42RPVZ30k3RVv?usp=sharing)**

### Expected Data Structure

After downloading, organize the data as follows:

```
RankingModels/
├── Codes/
│   └── DIN/
│       └── (this repository)
└── data/
    ├── local_train_splitByUser    # Training data
    ├── local_test_splitByUser     # Test data
    ├── uid_voc.pkl                # User ID vocabulary
    ├── mid_voc.pkl                # Item ID vocabulary
    ├── cat_voc.pkl                # Category vocabulary
    ├── item-info                  # Item metadata
    └── reviews-info               # Review information
```

### Data Format

Each line in the training/test file contains:
```
label \t user_id \t item_id \t category \t history_items \t history_categories
```

- **label**: 0 (no click) or 1 (click)
- **user_id**: User identifier
- **item_id**: Candidate item identifier
- **category**: Candidate item category
- **history_items**: User's historical item sequence (separated by `\x01`)
- **history_categories**: Corresponding category sequence (separated by `\x01`)

## 🛠️ Installation

### Requirements

```bash
pip install torch numpy scikit-learn
```

### Tested Environment

- Python 3.7+
- PyTorch 1.10+
- NumPy 1.21+
- CUDA 11.0+ (optional, for GPU training)

## 🚀 Usage

### Training

Train the DIN model with default parameters:

```bash
python trainer.py --mode train --model DIN --epochs 5
```

**Training Parameters:**

```bash
python trainer.py \
    --mode train \
    --model DIN \
    --epochs 5
```

| Argument | Default | Description |
|----------|---------|-------------|
| `--mode` | `train` | Mode: `train` or `test` |
| `--model` | `DIN` | Model type |
| `--epochs` | `5` | Number of training epochs |

### Testing

Evaluate a trained model:

```bash
python trainer.py --mode test --model DIN --model_path best_model/cpkt_noshuffDIN42
```

**Testing Parameters:**

```bash
python trainer.py \
    --mode test \
    --model DIN \
    --model_path path/to/checkpoint
```

### Model Configuration

Edit hyperparameters in [trainer.py](trainer.py):

```python
EMBEDDING_DIM = 18          # Embedding dimension
HIDDEN_DIM = [108, 200, 80, 2]  # MLP hidden dimensions
batch_size = 128            # Batch size
maxlen = 100                # Max sequence length
lr = 0.001                  # Learning rate
```

## 📈 Training Details

### Hyperparameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| Embedding Dim | 18 | Dimension of embedding vectors |
| Attention Hidden | [80, 40, 1] | Local activation unit architecture |
| MLP Hidden | [200, 80] | Deep network hidden layers |
| Batch Size | 128 | Training batch size |
| Max Seq Length | 100 | Maximum user history length |
| Learning Rate | 0.001 | Adam optimizer learning rate |
| Weight Decay | 0.0001 | L2 regularization |

### Training Process

- **Optimizer**: Adam (β₁=0.9, β₂=0.999)
- **Loss Function**: Binary Cross-Entropy
- **Evaluation Metrics**: AUC, Accuracy, Loss
- **Checkpointing**: Best model saved based on test AUC
- **Early Stopping**: Manual (monitor test AUC)

### Expected Output

During training, you'll see:

```
UID Embedding trainable params: 6516720
MID Embedding trainable params: 4415796
CAT Embedding trainable params: 19212
Attention Layer trainable params: 11041
MLP Layer trainable params: 38536
Total trainable params: 11001305
test_auc: 0.5039, test_loss: 0.3661, test_acc: 0.5009
epoch: 1/iter: 500, train loss: 0.3340, train acc: 0.5775
...
```

## 🔬 Model Components Explained

### 1. Attention Mechanism

The attention layer computes relevance scores between candidate items and historical behaviors:

```python
attention_score = LocalActivationUnit(candidate, history)
attention_weight = softmax(attention_score)
user_interest = Σ(attention_weight_i × history_i)
```

**Feature Engineering:**
- Raw features: `[query, behavior]`
- Difference: `query - behavior`
- Interaction: `query × behavior`
- Final input: `[query, behavior, query-behavior, query×behavior]`

### 2. Dice Activation

Adaptive activation function that adjusts based on data statistics:

```python
x_normed = (x - mean(x)) / std(x)
p(x) = sigmoid(x_normed)
output = [α + (1-α)·p(x)] · x
```

Where `α` is a learnable parameter per feature dimension.

### 3. Feature Concatenation

Final MLP input combines multiple representations:

```python
features = concat([
    user_embedding,              # 18-dim
    candidate_item_embedding,    # 36-dim (item + category)
    sum(history_embeddings),     # 36-dim (global interest)
    candidate × sum(history),    # 36-dim (global interaction)
    attention_output             # 36-dim (local interest)
])
# Total: 162 dimensions
```

## 📊 Results

The model is evaluated on:
- **AUC (Area Under ROC Curve)**: Primary metric
- **Accuracy**: Classification accuracy
- **Log Loss**: Cross-entropy loss

Checkpoints are saved in:
- `output/`: Training checkpoints (every 1000 iterations)
- `best_model/`: Best model based on test AUC

## 📚 Citation

If you use this code in your research, please cite the original paper:

```bibtex
@inproceedings{zhou2018din,
  title={Deep interest network for click-through rate prediction},
  author={Zhou, Guorui and Zhu, Xiaoqiang and Song, Chengru and Fan, Ying and Zhu, Han and Ma, Xiao and Yan, Yanghui and Jin, Junqi and Li, Han and Gai, Kun},
  booktitle={Proceedings of the 24th ACM SIGKDD International Conference on Knowledge Discovery \& Data Mining},
  pages={1059--1068},
  year={2018}
}
```

## 🙏 Acknowledgments

This implementation is inspired by and builds upon the excellent work from:

- **Original Paper**: [Deep Interest Network for Click-Through Rate Prediction (KDD 2018)](https://arxiv.org/abs/1706.06978)
- **Reference Implementation**: [yeyingdege/ctr-din-pytorch](https://github.com/yeyingdege/ctr-din-pytorch)

Special thanks to the authors of the reference repository for their valuable PyTorch implementation.

## 📝 License

This project is open-sourced for research and educational purposes. Please refer to the original paper and repository for commercial use considerations.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit issues or pull requests.

## 📧 Contact

For questions or issues, please open an issue on GitHub.

---

**⭐ If you find this implementation useful, please consider giving it a star!**
