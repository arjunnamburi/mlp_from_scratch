# Neural network from scratch in NumPy

A fully connected neural network (multilayer perceptron) written using only NumPy, with backpropagation derived by hand. You choose the number of hidden layers and the number of neurons in each one when you run the notebook. I built it in July 2026, at the end of my first year, to understand what actually happens inside `model.fit()`.

## What it does

- **Configurable architecture.** The notebook asks for the number of hidden layers and the width of each one, then builds the network from a list of `Layer` objects. Input size is 30 and the output is a single sigmoid unit.
- **A `Layer` class** that stores its weights, biases and its last input, pre-activation and activation, so it has everything it needs for the backward pass. Each layer computes its own `grad_W` and `grad_b`, plus the gradient it passes back to the previous layer.
- **Hand-derived backpropagation.** Hidden layers use ReLU, and the output uses sigmoid. The gradients are worked out on paper using the chain rule and coded directly. There is no autograd.
- **Training.** Full-batch gradient descent on mean squared error, learning rate 0.05, 1000 epochs. He initialisation for the weights.
- **Gradient check.** Each weight and bias is nudged by ±1e-5, and the resulting slope of the loss is compared with the backprop gradient.

## Data

The [Wisconsin breast cancer dataset](https://scikit-learn.org/stable/datasets/toy_dataset.html#breast-cancer-dataset) from scikit-learn has 30 numeric features per tumour and a binary label (malignant or benign). I use the first 500 rows and split them into 320 train, 80 validation and 100 test. Features are standardised per feature, using training-set statistics only.

## Results

Network: 2 hidden layers of 8 neurons (same seed and settings for both rows).

| Version | Test accuracy |
|---|---|
| Always predict the majority class | 65% |
| First version (one global mean and std for all features) | 90% |
| Per-feature standardisation | 97% |

**Gradient check:** the relative error between the backprop and numerical gradients is below 4e-10 for every layer's weights and biases, so the hand-derived gradients are correct.

| Layer | W error | b error |
|---|---|---|
| 1 | 3.55e-10 | 2.71e-10 |
| 2 | 7.15e-11 | 1.06e-10 |
| 3 (output) | 1.83e-10 | 1.31e-10 |

### A bug I found

My first version normalised with `np.mean(X_train)`, which averages over the whole matrix. Every feature then got the same scaling. The features range from about 0.004 to about 900, so the large ones (like tumour area) dominated and the small ones were effectively invisible. Normalising each column separately (`axis=0`) raised test accuracy from 90% to 97%.

## Run it

Open the notebook in Google Colab or Jupyter and run all cells. When prompted, enter the number of hidden layers and the neurons per layer (for example `2`, then `8`, `8`, which gives the results above). Install the requirements with `pip install -r requirements.txt`. scikit-learn is used only to load and split the data.

## What I'd do next

- Use a different loss function, such as binary cross-entropy, so that the sigmoid's `p(1-p)` term is removed from the output gradient
- Try mini-batch gradient descent
- Tune hyperparameters such as the learning rate, and use early stopping to prevent overfitting
