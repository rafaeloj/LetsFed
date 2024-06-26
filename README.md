### Build
```
  ~$ ./build.sh
```

## Run
```
  ~$ ./experiments.sh
```

## Mudar parâmetros
```
  config-debug.ini
```


~Caso mudar algum parâmetro atualize os nomes dos arquivos em `./experiments.sh`

### Options

- `-c`, `--clients`
  - **Description**: Specifies the number of clients to be used.
  - **Default**: `10`
  - **Type**: Integer

- `-e`, `--local-epochs`
  - **Description**: Sets the number of local epochs for training on each client.
  - **Default**: `1`
  - **Type**: Integer

- `-d`, `--dataset`
  - **Description**: Defines which dataset to use for training.
  - **Default**: `mnist`
  - **Type**: String
  - **Available**: `'mnist'`, `'cifar10'`, `'fashion_mnist'`, `'sasha/dog-food'`, `'zh-plus/tiny-imagenet'`, and any Flower federated dataset

- `-r`, `--rounds`
  - **Description**: Determines the number of rounds of training.
  - **Default**: `10`
  - **Type**: Integer

- `-s`, `--strategy`
  - **Description**: Specifies the federated learning strategy to be used.
  - **Default**: `CIA`
  - **Type**: String

- `--no-iid`
  - **Description**: A flag to set the dataset distribution among clients to be non-i.i.d.
  - **Default**: `True`
  - **Action**: `store_true` (This option does not require a value, it's either present or not.)

- `--init-clients`
  - **Description**: Sets the number of initial clients that participate in the federated process.
  - **Default**: `2`
  - **Type**: Integer