# Classifier Training

This project contains a script for training a classifier on a dataset using PyTorch and PyTorch Lightning. The script includes configurable hyperparameters and utilizes a custom convolutional neural network model for classification.

## Requirements

- nn_zoo
- wandb account and API key

## Installation

1. Install `nn-zoo`

    ```sh
    pip install nn-zoo
    ```

2. Copy over the `train.py` script and the `config.yaml` to your project directory.

3. Run the script with the desired hyperparameters.

    ```sh
    python train.py \
        --learning_rate 0.001 \
        --width 8 \
        --depth 3 \
        --dropout_prob 0.2 \
        --epochs 10
    ```

    Or initialize a `sweep` for hyperparameter tuning.

    ```sh
    wandb sweep --project<PROJECT_NAME> config.yaml
    ```

    ```sh
    wandb agent <SWEEP_ID>
    ```

## Usage

To train the MNIST classifier, run the following command with optional arguments to customize the training process:

```sh
python train_mnist.py [options]
```

### Options

- `--learning_rate` (float): Learning rate for the optimizer. Default: 1e-3
- `--width` (int): Width of the first layer of the model. Default: 4
- `--depth` (int): Depth of the convolutional stack. Default: 2
- `--dropout_prob` (float): Dropout probability for the classifier. Default: 0.1
- `--use_linear_norm` (bool): Use a linear layer for normalization. Default: False
- `--optim` (str): Optimizer to use for training. Default: "adam"
- `--batch_size` (int): Batch size for training. Default: 64
- `--epochs` (int): Number of training epochs. Default: 1
- `--num_workers` (int): Number of workers for the dataloader. Default: 2
- `--prefetch_factor` (int): Prefetch factor for the dataloader. Default: 2
- `--pin_memory` (bool): Pin memory for the dataloader. Default: False
- `--persistent_workers` (bool): Use persistent workers for the dataloader. Default: False

### Example Command

```sh
python train.py \
    --learning_rate 0.001 \
    --width 8 \
    --depth 3 \
    --dropout_prob 0.2 \
    --epochs 10
```

## Model Architecture

The classifier model is defined in the `Classifer` class, which consists of:

- A backbone composed of convolutional blocks for feature extraction.
- A classifier head for predicting the class labels.

### Model Summary

To view the model summary, the script uses the `torchinfo` library:

```python
from torchinfo import summary
summary(classifier_trainer.model, input_size=(args.batch_size, 1, 32, 32))
```

## Logging and Visualization

The training process is logged using Weights and Biases (Wandb). The logger is configured as follows:

```python
logger = WandbLogger(project="mnist-classifier", name="classifier", log_model=True)
logger.watch(classifier_trainer.model, log="all")
logger.log_hyperparams(vars(args))
```

## Script Structure

1. **Argument Parsing**: The script uses `argparse` to handle command-line arguments.
2. **Data Module**: The `MNISTDataModule` class from `nn_zoo` handles data loading and preprocessing.
3. **Model Initialization**: The `Classifer` class defines the CNN model.
4. **Training and Testing**: The `ClassifierTrainer` class manages the training and testing loops.
5. **Main Function**: The `main` function orchestrates the training process.
