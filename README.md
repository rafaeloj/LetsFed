# FedCIA: Federated Client Independable Aggregation

## Command Line Options

This section details the available command line options for the program.

### Options

- `-c`, `--clients`
  - **Description**: Specifies the number of clients to be used.
  - **Default**: `0`
  - **Type**: Integer

- `-e`, `--local-epochs`
  - **Description**: Sets the number of local epochs for training on each client.
  - **Default**: `1`
  - **Type**: Integer

- `-d`, `--dataset`
  - **Description**: Defines which dataset to use for training.
  - **Default**: `mnist`
  - **Type**: String
  - **Available**: `'mnist'`, `'cifar10'`, `'fashion_mnist'`, `'sasha/dog-food'`, `'zh-plus/tiny-imagenet'` and any Flower federated dataset

- `-r`, `--rounds`
  - **Description**: Determines the number of rounds of training.
  - **Default**: `100`
  - **Type**: Integer

- `-s`, `--strategy`
  - **Description**: Specifies the federated learning strategy to be used.
  - **Default**: `CIA`
  - **Type**: String

- `--no-iid`
  - **Description**: A flag to set the dataset distribution among clients to be non-i.i.d.
  - **Default**: `False`
  - **Action**: `store_true` (This option does not require a value, it's either present or not.)

- `--init-clients`
  - **Description**: Sets the number of initial clients that participate of federated proccess.
  - **Default**: `2`
  - **Type**: Integer

### Usage

Here is a simple example of how to use these options in the command line:

```bash
~$ python environment.py --clients 10 --local-epochs 5 --dataset MNIST --rounds 50 --strategy CIA --init-clients 3
```

