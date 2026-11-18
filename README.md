# VGG11 on CIFAR-10 with Transfer Learning

**A didactic Jupyter notebook that implements VGG11 with batch normalization in PyTorch and fine-tunes ImageNet weights on CIFAR-10, reaching 92.41% test accuracy in five epochs.**

> **Context.** Built in May 2025 as a personal learning project, when I was publishing my first public repositories with coursework in machine learning and deep learning. The notebook is written as a walkthrough: it explains the VGG architecture layer by layer and shows how a pre-trained model is adapted to a smaller dataset.

## What it does

- Rebuilds VGG11 (configuration "A") from a compact layer list, with a `BatchNorm2d` after every convolution and a shared classifier head fed by `AdaptiveAvgPool2d(7)`.
- Loads the ImageNet-trained `vgg11_bn` from `torchvision`, swaps its 1000-class output layer for a 10-class one, and copies the weights into the custom model (128,812,810 trainable parameters).
- Converts CIFAR-10 from 32x32 to 224x224 and normalizes it with the ImageNet mean and standard deviation, adding rotation, horizontal flip and random crop on the training split.
- Implements a learning rate finder from scratch, then fine-tunes with Adam and discriminative learning rates: the convolutional features at one tenth of the rate used for the classifier.
- Trains for 5 epochs and then inspects the result with a confusion matrix, the most confident mistakes, PCA and t-SNE projections of the output logits, and visualizations of the first-layer filters.

## How it works

1. Define the `VGG` module: a feature extractor passed in as an argument, adaptive average pooling to 7x7 so any input of at least 32x32 fits the original classifier, and the standard three-layer fully connected head with dropout.
2. Describe each VGG variant as a list where integers are convolution widths and `'M'` is a 2x2 max pool, and turn it into an `nn.Sequential` with `get_vgg_layers`.
3. Load the pre-trained `torchvision` model, replace `classifier[-1]` with a new `nn.Linear(4096, 10)`, and transfer all weights with `load_state_dict`.
4. Split CIFAR-10 into 45,000 training and 5,000 validation images, build the transform pipelines, and load batches of 128.
5. Sweep the learning rate exponentially from 1e-7 to 10 over 100 iterations while recording the smoothed loss, then restart from the saved initial weights and train with the chosen rate of 5e-4.
6. Evaluate on the 10,000 test images and use the saved best-validation checkpoint for the analysis cells.

## Results

All numbers below are printed in the notebook outputs.

Parameter count:

```
The model has 128,812,810 trainable parameters
```

Fine-tuning log:

```
Epoch: 01 | Epoch Time: 8m 15s
	Train Loss: 0.450 | Train Acc: 84.57%
	 Val. Loss: 0.266 |  Val. Acc: 91.09%
Epoch: 02 | Epoch Time: 8m 13s
	Train Loss: 0.215 | Train Acc: 92.82%
	 Val. Loss: 0.238 |  Val. Acc: 92.81%
Epoch: 03 | Epoch Time: 8m 15s
	Train Loss: 0.148 | Train Acc: 95.00%
	 Val. Loss: 0.228 |  Val. Acc: 93.24%
Epoch: 04 | Epoch Time: 8m 14s
	Train Loss: 0.107 | Train Acc: 96.45%
	 Val. Loss: 0.232 |  Val. Acc: 93.20%
Epoch: 05 | Epoch Time: 8m 14s
	Train Loss: 0.082 | Train Acc: 97.25%
	 Val. Loss: 0.240 |  Val. Acc: 93.61%
```

Test set:

```
Test Loss: 0.242 | Test Acc: 92.41%
```

Validation loss is lowest at epoch 3 while training loss keeps falling, so the notebook keeps the checkpoint from the best validation epoch and evaluates that one on the test set.

## Run it

The repository has no requirements file; these are the libraries the notebook imports:

```
python3 -m venv .venv
source .venv/bin/activate
pip install torch torchvision scikit-learn matplotlib tqdm jupyter
jupyter notebook vgg11_cifar10.ipynb
```

Run the cells in order from top to bottom. What you need:

- A GPU. The notebook metadata records a Colab GPU runtime, and each training epoch took about 8 minutes there. CPU training is impractical at 224x224 with 128M parameters.
- Internet access on the first run: `torchvision` downloads the VGG11 batch-norm weights (about 140 MB) and `torchvision.datasets` downloads CIFAR-10 into `.data/`.
- Space for the CIFAR-10 dataset and two checkpoints, `init_params.pt` (written by the learning rate finder) and `tut4-model.pt` (best validation epoch).

Not verified: the exact library versions. The notebook was produced with Python 3.8.3 and uses the legacy `pretrained=True` argument of `torchvision.models`, which newer releases deprecate in favour of `weights=`.

## Repository layout

```
vgg11_cifar10.ipynb   the whole project: architecture, training, evaluation and analysis
```

## Stack

Python, PyTorch, torchvision, scikit-learn, matplotlib, tqdm, Jupyter (run on Google Colab).
