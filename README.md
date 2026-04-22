# 🧠 Self-Pruning Neural Network (PyTorch)

A PyTorch implementation of a **self-pruning neural network** that learns to remove its own unnecessary connections during training using **differentiable gating and L1 regularization**.

---

## 🚀 Overview

Traditional pruning methods remove weights **after training**.
This project introduces a **learnable pruning mechanism** where the network **adapts its structure during training**.

Each connection in the network is controlled by a **trainable gate**:

* Important connections → remain active
* Unimportant connections → suppressed automatically

This results in a **sparser and more efficient model** without manual pruning.

---

## 🏗️ Architecture

The model is a simple feedforward neural network built using custom layers:

* `PrunableLinear` (custom layer)
* ReLU activations
* Fully connected architecture

### 🔑 Key Idea

Each weight has an associated **gate score**:

[
W_{effective} = W \times \sigma(G)
]

Where:

* (W): Weight matrix
* (G): Gate scores (learnable)
* (\sigma(G)): Sigmoid → produces values in [0,1]

---

## ✂️ Sparsity Mechanism

To encourage pruning, an **L1 regularization penalty** is applied on gate values:

[
Loss = ClassificationLoss + \lambda \times SparsityLoss
]

* ClassificationLoss → CrossEntropyLoss
* SparsityLoss → Sum of all gate values
* (\lambda) → Controls pruning strength

Higher (\lambda) ⇒ More sparsity, but possible accuracy drop.

---

## 📊 Results

| Lambda | Accuracy (%) | Sparsity (%) |
| ------ | ------------ | ------------ |
| 0.0    | 54.86        | 0.00         |
| 1e-05  | 57.10        | 1.72         |
| 0.0001 | 55.65        | 36.17        |
| 0.001  | 50.66        | 94.79        |
| 0.01   | 42.84        | 99.83        |

### 📌 Observations

* Small λ → minimal pruning
* Moderate λ → good balance
* High λ → extreme sparsity but lower accuracy

---

## 📈 Visualization

A histogram of gate values shows:

* Many gates pushed close to **zero (pruned)**
* Others remain **active (important connections)**

This confirms successful **automatic pruning during training**.

---

## ⚙️ Installation & Setup

### 1️⃣ Clone the repository

```bash
git clone https://github.com/your-username/self-pruning-network.git
cd self-pruning-network
```

### 2️⃣ Install dependencies

```bash
pip install torch torchvision matplotlib
```

### 3️⃣ Run the project

```bash
python main.py
```

---

## 📁 Project Structure

```
self-pruning-network/
├── main.py        # Training + evaluation script
├── report.md      # Detailed explanation and analysis
├── README.md      # Project documentation
```

---

## 🧪 Dataset

* **CIFAR-10**
* Automatically downloaded via `torchvision.datasets`

---

## 🧠 Key Concepts Demonstrated

* Differentiable pruning
* L1 regularization for sparsity
* Custom PyTorch layers
* Trade-off between accuracy and model compression
* End-to-end learning of sparse structures

---

## 📌 Key Takeaways

* Neural networks can **learn to prune themselves**
* L1 regularization effectively drives sparsity
* There is a **critical balance** between pruning and performance
* This approach eliminates the need for **post-training pruning**

---

## 🔮 Future Improvements

* Extend to CNN architectures
* Use structured pruning (neurons/channels)
* Deploy on edge devices
* Combine with quantization for further compression

---

## 👨‍💻 Author

**Jaibir Singh Chhabra**

---

## 📬 Submission Note

This project was developed as part of an **AI Engineering Internship Case Study**, focusing on practical implementation of self-pruning neural networks and sparse learning techniques.
