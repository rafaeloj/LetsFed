## Let’s Federate - Effective Communication Strategy for Dynamic Client Participation - ICMLA Conference

**Abstract:**
Federated Learning (FL) has emerged as a privacy-preserving powerful tool in decentralized Machine Learning (ML) environments. However, real-world scenarios often face bandwidth limitations that can be overwhelmed when all clients simultaneously perform training and communicate with the server in a federated system. Consequently, selection mechanisms are critical for identifying optimal subsets of clients to participate in the federation. Traditional selection methods, however, typically do not allow clients the autonomy to decide whether or not to contribute to the federation. Therefore, this paper proposes LetsFed, a client selection framework that respects client independence throughout the training process. The LetsFed framework differentiates between participating and non-participating clients, employing targeted selection mechanisms to address system challenges effectively. Empirical results demonstrate that LestFed can outperform, in dynamic client participation environments, literature solutions by up to 40%, reducing unnecessary data transmission by as much as 29%, while also enhancing the efficacy of the selection process.

Cite
```
@INPROCEEDINGS{10903243,
  author={Jarczewski, Rafael O. and Cerqueira, Eduardo and Bittencourt, Luiz F. and Loureiro, Antonio A. F. and Villas, Leandro A. and de Souza, Allan M.},
  booktitle={2024 International Conference on Machine Learning and Applications (ICMLA)},
  title={Let's Federate - Effective Communication Strategy for Dynamic Client Participation},
  year={2024},
  volume={},
  number={},
  pages={361-368},
  keywords={Training;Analytical models;Federated learning;Bandwidth;Robustness;Data communication;Servers;Game theory;Faces;Convergence;federated learning;client selection;dynamic client participation},
  doi={10.1109/ICMLA61862.2024.00055}
}
```

---

## 🎯 Overview

A modular and extensible **Federated Learning research framework** built on [Flower](https://flower.dev/), designed for experimenting with dynamic client participation scenarios. The framework implements multiple aggregation, selection, and training strategies following software engineering best practices with modern design patterns.

## ✨ Key Features

- **🏗️ Modular Architecture**: Built with Strategy Pattern, Factory Pattern, Builder Pattern, and Chain of Responsibility
- **🔌 Extensible Design**: Plugin-based driver system for composable client behaviors
- **🎛️ Multiple Strategy Support**:
  - **Aggregation**: FedAvg, MaxFL
  - **Client Selection**: Random, DEEV, PoC, Round Robin, LetsFed
  - **Training**: Normal, LetsFed, MaxFL, FedPer, QFFL
- **🐳 Containerized Deployment**: Docker and Docker Compose with GPU support
- **⚙️ YAML Configuration**: Type-safe configuration management with OmegaConf
- **📊 Comprehensive Logging**: Automatic metrics tracking for analysis
- **🎯 Type Safety**: Full type hints throughout the codebase

## 📁 Project Structure

```
LetsFed/
├── 🖥️  server/                        # Federated Learning Server
│   ├── strategies/
│   │   ├── fl_server.py              # Main server implementation (Strategy Pattern)
│   │   ├── factory.py                # ServerBuilder (Builder Pattern)
│   │   ├── aggregate_method/         # Aggregation strategies (Factory Pattern)
│   │   │   ├── base.py               # Base aggregation interface
│   │   │   ├── factory.py            # AggregationFactory
│   │   │   └── types/
│   │   │       ├── fedavg.py         # FedAvg aggregation
│   │   │       └── maxfl.py          # MaxFL aggregation
│   │   └── client_selection_method/  # Selection strategies (Factory Pattern)
│   │       ├── base.py               # Base selection interface
│   │       ├── factory.py            # ClientSelectionFactory
│   │       └── types/
│   │           ├── random.py         # Random selection
│   │           ├── deev.py           # DEEV selection
│   │           ├── poc.py            # Power of Choice
│   │           ├── round_robin.py    # Round Robin
│   │           └── letsfed.py        # LetsFed selection
│   ├── strategies_manager.py         # Server entrypoint
│   ├── Dockerfile                    # Server container (CPU)
│   └── Dockerfile.gpu                # Server container (GPU)
│
├── 📱 client/                         # Federated Learning Client
│   ├── strategies/
│   │   ├── fl_client.py              # Main client implementation (Strategy Pattern)
│   │   ├── factory.py                # ClientBuilder (Builder Pattern)
│   │   ├── training/                 # Training strategies (Factory Pattern)
│   │   │   ├── base.py               # Base training interface
│   │   │   ├── factory.py            # TrainingStrategyFactory
│   │   │   └── types/
│   │   │       ├── normal.py         # Standard FedAvg training
│   │   │       ├── letsfed.py        # LetsFed training
│   │   │       ├── maxfl.py          # MaxFL training
│   │   │       ├── fedper.py         # FedPer training
│   │   │       └── qffl.py           # QFFL training
│   │   └── drivers/                  # Modular behaviors (Chain of Responsibility)
│   │       ├── driver.py             # Base driver interface
│   │       ├── accuracy.py           # Accuracy-based decision driver
│   │       ├── curiosity.py          # Curiosity-driven participation
│   │       ├── maxfl_qk.py           # MaxFL quality metric
│   │       └── maxfl_pre_training.py # MaxFL initialization
│   ├── strategies_manager.py         # Client entrypoint
│   ├── Dockerfile                    # Client container (CPU)
│   └── Dockerfile.gpu                # Client container (GPU)
│
├── ⚙️  conf/                          # Configuration Management
│   ├── config.yaml                   # Main configuration file
│   ├── loader.py                     # Config loader with validation
│   └── structs.py                    # Type-safe config dataclasses
│
├── 📊 dataset_manager/                # Dataset Handling
│   ├── dataset_manager.py            # Dataset partitioning logic
│   ├── dataloader.py                 # Data loading utilities
│   └── structs.py                    # Data structures
│
├── 🧠 model/                          # Model Management
│   └── model_manager.py              # Model factory (CNN, DNN)
│
├── 🛠️  utils/                         # Utilities
│   ├── logger.py                     # Metrics logging system
│   ├── docker_compose_manager.py     # Dynamic compose generator
│   └── utils.py                      # Helper functions
│
├── 🌐 app/                            # Web Interface
│   ├── service.py                    # Flask web app
│   └── templates/                    # HTML templates
│
├── 📋 requirements-client.txt         # Client dependencies
├── 📋 requirements-server.txt         # Server dependencies
├── 🐳 docker-compose.yml              # Container orchestration (CPU)
├── 🐳 docker-compose.gpu.yml          # Container orchestration (GPU)
├── 🔨 build.sh                        # Build script
└── 🧪 test/                           # Testing and analysis
    └── teste.ipynb                   # Results analysis notebook
```

## 🚀 Quick Start

### Prerequisites

- **Docker** >= 20.10
- **Docker Compose** >= 2.0
- **(Optional)** NVIDIA Docker for GPU support
- **Python** >= 3.8 (for local development)

### Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd LetsFed
   ```

2. **Configure your experiment**

   Edit `conf/config.yaml`:
   ```yaml
   rounds: 10
   n_clients: 30

   server:
     aggregation_method:
       name: fedavg  # Options: fedavg, maxfl
     selection_method:
       name: letsfed  # Options: random, deev, poc, round_robin, letsfed
       perc_of_clients: 0.3

   client:
     training_strategy:
       name: letsfed  # Options: normal, letsfed, maxfl, fedper, qffl
     epochs: 5

   dataset:
     dataset: fashion_mnist  # Options: fashion_mnist, cifar10
   ```

3. **Build containers**
   ```bash
   # CPU version
   ./build.sh

   # GPU version
   docker-compose -f docker-compose.gpu.yml build
   ```

4. **Run the experiment**
   ```bash
   # CPU
   docker-compose up

   # GPU
   docker-compose -f docker-compose.gpu.yml up
   ```

5. **Analyze results**

   Logs are saved in the `logs/` directory. Use the provided Jupyter notebook:
   ```bash
   jupyter notebook test/teste.ipynb
   ```

### Alternative: Local Development

For development without Docker:

```bash
# Install dependencies
pip install -r requirements-server.txt
pip install -r requirements-client.txt

# Run server
python -m server.strategies_manager

# Run clients (in separate terminals)
CID=0 python -m client.strategies_manager
CID=1 python -m client.strategies_manager
```

## 🏗️ Architecture & Design Patterns

The framework implements a sophisticated architecture using multiple design patterns to ensure modularity, extensibility, and maintainability.

### Design Patterns Overview

```mermaid
graph TB
    subgraph "Factory Pattern"
        AF[AggregationFactory]
        CSF[ClientSelectionFactory]
        TSF[TrainingStrategyFactory]
        AF -->|creates| AGG[FedAvg/MaxFL]
        CSF -->|creates| SEL[Random/LetsFed/PoC]
        TSF -->|creates| TRN[Normal/LetsFed/MaxFL]
    end

    subgraph "Builder Pattern"
        SB[ServerBuilder]
        CB[ClientBuilder]
        SB -->|uses| AF
        SB -->|uses| CSF
        CB -->|uses| TSF
    end

    subgraph "Strategy Pattern + Dependency Injection"
        FLS[FLServer]
        FLC[FLClient]
        SB -->|builds| FLS
        CB -->|builds| FLC
        FLS -->|delegates to| AGG
        FLS -->|delegates to| SEL
        FLC -->|delegates to| TRN
    end

    subgraph "Chain of Responsibility"
        TRN -->|configures| DRV[Drivers]
        DRV -->|pipeline| D1[AccuracyDriver]
        DRV -->|pipeline| D2[CuriosityDriver]
        DRV -->|pipeline| D3[MaxFLQkDriver]
    end

    style Factory Pattern fill:#e1f5ff
    style Builder Pattern fill:#fff4e1
    style Strategy Pattern + Dependency Injection fill:#e8f5e9
    style Chain of Responsibility fill:#f3e5f5
```

### 1. Strategy Pattern + Dependency Injection

The core classes (`FLServer` and `FLClient`) are **generic implementations** that delegate behavior to injected strategies:

```mermaid
classDiagram
    class FLServer {
        +ClientSelectionMethod selection
        +AggregateMethod aggregation
        +configure_fit()
        +aggregate_fit()
    }

    class FLClient {
        +TrainingStrategy training_strategy
        +List~Driver~ drivers
        +fit()
        +evaluate()
    }

    class ClientSelectionMethod {
        <<interface>>
        +select()*
    }

    class AggregateMethod {
        <<interface>>
        +agg_fit()*
    }

    class TrainingStrategy {
        <<interface>>
        +fit()*
        +evaluate()*
    }

    FLServer --> ClientSelectionMethod
    FLServer --> AggregateMethod
    FLClient --> TrainingStrategy

    ClientSelectionMethod <|-- Random
    ClientSelectionMethod <|-- LetsFed
    ClientSelectionMethod <|-- PoC

    AggregateMethod <|-- FedAvg
    AggregateMethod <|-- MaxFL

    TrainingStrategy <|-- Normal
    TrainingStrategy <|-- LetsFedTraining
    TrainingStrategy <|-- MaxFLTraining
```

**Key Points:**
- `FLServer` and `FLClient` are **single, generic classes** (not interfaces)
- Different behaviors are achieved through **strategy composition**
- Strategies are created by **Factory Pattern** and **injected via Builder Pattern**

### 2. Factory Pattern

Used for creating different strategy implementations:

```mermaid
graph LR
    subgraph Factories
        AF[AggregationFactory]
        CSF[ClientSelectionFactory]
        TSF[TrainingStrategyFactory]
    end

    subgraph Config
        C[config.yaml]
    end

    subgraph Implementations
        FA[FedAvg]
        MX[MaxFL]
        RN[Random]
        LF[LetsFed]
        PC[PoC]
        NR[Normal]
        LT[LetsFedTraining]
        MT[MaxFLTraining]
    end

    C -->|aggregation: fedavg| AF
    C -->|selection: letsfed| CSF
    C -->|training: letsfed| TSF

    AF -.->|creates| FA
    AF -.->|creates| MX
    CSF -.->|creates| RN
    CSF -.->|creates| LF
    CSF -.->|creates| PC
    TSF -.->|creates| NR
    TSF -.->|creates| LT
    TSF -.->|creates| MT
```

### 3. Builder Pattern

Simplifies construction of `FLServer` and `FLClient` with all dependencies:

```mermaid
sequenceDiagram
    participant Config
    participant ServerBuilder
    participant Factories
    participant FLServer

    Config->>ServerBuilder: create(config)
    ServerBuilder->>Factories: create strategies
    Factories-->>ServerBuilder: strategy instances
    ServerBuilder->>FLServer: new(strategies)
    FLServer-->>ServerBuilder: configured server
    ServerBuilder-->>Config: ready server
```

### 4. Chain of Responsibility (Drivers)

Training strategies can compose behaviors through a **pipeline of drivers**:

```mermaid
graph LR
    subgraph TrainingStrategy
        TS[LetsFedTraining]
    end

    subgraph Driver Pipeline
        D1[AccuracyDriver]
        D2[CuriosityDriver]
        D3[MaxFLQkDriver]
    end

    subgraph FLClient
        FC[Client State]
    end

    TS -->|configures| D1
    TS -->|configures| D2
    TS -->|configures| D3

    D1 -->|modifies| FC
    D2 -->|modifies| FC
    D3 -->|modifies| FC

    FC -.->|client.willing| D1
    FC -.->|client.state| D2
    FC -.->|client.qk| D3
```

**Example:**
- **LetsFedTraining** uses `[AccuracyDriver, CuriosityDriver]`
- **MaxFLTraining** uses `[MaxFLQkDriver]`
- **LetsFedTraining** can also use `MaxFLQkDriver` for additional metrics!

**Benefits:**
- ✅ **Reusability**: Share drivers across strategies
- ✅ **Modularity**: Each driver has a single responsibility
- ✅ **Composability**: Mix and match drivers freely

### Federated Learning Flow

```mermaid
sequenceDiagram
    participant Server
    participant ClientManager
    participant Client1
    participant Client2

    Note over Server: Round N starts

    Server->>Server: configure_fit()
    Server->>Server: selection.select()
    Server->>ClientManager: sample_clients()
    ClientManager-->>Server: [Client1, Client2]

    par Training Phase
        Server->>Client1: fit(parameters)
        Server->>Client2: fit(parameters)

        Client1->>Client1: apply_drivers()
        Client1->>Client1: training_strategy.fit()
        Client1-->>Server: updated_weights_1

        Client2->>Client2: apply_drivers()
        Client2->>Client2: training_strategy.fit()
        Client2-->>Server: updated_weights_2
    end

    Server->>Server: aggregate_fit()
    Server->>Server: aggregation.agg_fit()

    Note over Server: Evaluation Phase

    par Evaluation Phase
        Server->>Client1: evaluate(parameters)
        Server->>Client2: evaluate(parameters)

        Client1->>Client1: training_strategy.evaluate()
        Client1-->>Server: metrics_1

        Client2->>Client2: training_strategy.evaluate()
        Client2-->>Server: metrics_2
    end

    Server->>Server: aggregate_evaluate()
    Note over Server: Round N ends
```

## 📊 Available Strategies

### Server-Side Strategies

#### Aggregation Methods

| Strategy | Description | Use Case |
|----------|-------------|----------|
| **FedAvg** | Weighted average of client models | Standard federated learning |
| **MaxFL** | Utility-maximizing aggregation | Resource-constrained environments |

**Configuration:**
```yaml
server:
  aggregation_method:
    name: fedavg  # or maxfl
```

#### Client Selection Methods

| Strategy | Description | Key Feature |
|----------|-------------|-------------|
| **Random** | Random client selection | Baseline, simple |
| **Round Robin** | Sequential selection | Fair participation |
| **PoC** | Power of Choice | Quality-based selection |
| **DEEV** | Diversity-based selection | Reduces staleness |
| **LetsFed** | Voluntary participation | Respects client autonomy |

**Configuration:**
```yaml
server:
  selection_method:
    name: letsfed
    perc_of_clients: 0.3  # Select 30% of clients
```

### Client-Side Strategies

#### Training Strategies

| Strategy | Description | Special Features |
|----------|-------------|------------------|
| **Normal** | Standard FedAvg training | Baseline implementation |
| **LetsFed** | Voluntary participation training | Uses AccuracyDriver, CuriosityDriver |
| **MaxFL** | MaxFL training approach | Uses MaxFLQkDriver |
| **FedPer** | Personalized federated learning | Separate global/local layers |
| **QFFL** | Fair federated learning | Fairness-aware training |

**Configuration:**
```yaml
client:
  training_strategy:
    name: letsfed
    threshold: 1.0  # Strategy-specific parameter
```

#### Driver System (Modular Behaviors)

Drivers are **composable components** that add specific behaviors to training strategies:

| Driver | Purpose | Modifies |
|--------|---------|----------|
| **AccuracyDriver** | Decides if client wants to participate | `client.willing` |
| **CuriosityDriver** | Manages exploration/exploitation states | `client.state`, `client.curiosity` |
| **MaxFLQkDriver** | Calculates client quality metric | `client.qk` |
| **MaxFLPreTrainingDriver** | Initializes MaxFL threshold | `client.maxfl_threshold` |

**Example: Combining Drivers**

```python
class LetsFedTraining(TrainingStrategy):
    def _get_drivers(self):
        return [
            AccuracyDriver(),      # Decide participation
            CuriosityDriver(),     # Manage exploration
            MaxFLQkDriver(),       # Add quality metric (reused from MaxFL!)
        ]
```

This demonstrates the power of the **Chain of Responsibility pattern**: drivers from different strategies can be freely combined!

## 🔧 Extending the Framework

The framework is designed for easy extensibility. Here's how to add new components:

### Adding a New Training Strategy

1. **Create the implementation** in `client/strategies/training/types/`:

```python
from ..base import TrainingStrategy
from ...drivers.driver import Driver

class MyCustomTraining(TrainingStrategy):
    def init(self, client):
        # Initialize client-specific parameters
        client.add_drivers(self._get_drivers())

    def _get_drivers(self) -> list[Driver]:
        # Return list of drivers to use
        return [AccuracyDriver(), MyCustomDriver()]

    def fit(self, client, parameters, config):
        # Implement training logic
        client.model.set_weights(parameters)
        client.apply_drivers(parameters, config)
        # ... training code ...
        return updated_weights, num_examples, metrics

    def evaluate(self, client, parameters, config):
        # Implement evaluation logic
        # ...
        return loss, num_examples, metrics
```

2. **Register in factory** (`client/strategies/training/factory.py`):

```python
class TrainingStrategyFactory:
    @staticmethod
    def create(config: Environment) -> TrainingStrategy:
        strategies = {
            "normal": NormalClient,
            "letsfed": LetsFedClient,
            "my_custom": MyCustomTraining,  # Add here
        }
        return strategies[config.client.training_strategy.name]()
```

3. **Use in configuration**:

```yaml
client:
  training_strategy:
    name: my_custom
```

### Adding a New Driver

1. **Create the driver** in `client/strategies/drivers/`:

```python
from .driver import Driver
from flwr.common import Config, NDArrays

class MyCustomDriver(Driver):
    def run(self, client, parameters: NDArrays, config: Config) -> None:
        # Implement driver logic
        # Modify client state/attributes
        client.my_metric = self._calculate_metric(client)
```

2. **Use in any training strategy**:

```python
def _get_drivers(self):
    return [MyCustomDriver(), AccuracyDriver()]
```

### Adding a New Selection Method

1. **Create implementation** in `server/strategies/client_selection_method/types/`:

```python
from ..base import ClientSelectionMethod

class MyCustomSelection(ClientSelectionMethod):
    def select(self, server, client_manager, num_clients):
        # Implement selection logic
        # ...
        return selected_clients
```

2. **Register in factory** and update config as shown above.

### Adding a New Aggregation Method

Similar process in `server/strategies/aggregate_method/types/`.

### Architecture Benefits

```mermaid
graph TB
    subgraph "Easy to Extend"
        NT[New Training Strategy]
        ND[New Driver]
        NS[New Selection Method]
        NA[New Aggregation Method]
    end

    subgraph "No Changes Needed"
        FC[FLClient]
        FS[FLServer]
        CF[Config System]
        DK[Docker Setup]
    end

    NT -.->|registers in| Factory1[TrainingStrategyFactory]
    ND -.->|used by| NT
    NS -.->|registers in| Factory2[ClientSelectionFactory]
    NA -.->|registers in| Factory3[AggregationFactory]

    Factory1 -->|no change| FC
    Factory2 -->|no change| FS
    Factory3 -->|no change| FS

    style "Easy to Extend" fill:#c8e6c9
    style "No Changes Needed" fill:#fff9c4
```

**Key Principle**: You extend behavior by **adding new classes**, not by **modifying existing ones** (Open/Closed Principle).

## 📈 Logging and Analysis

### Automatic Metrics Collection

The framework automatically logs metrics to the `logs/` directory:

```
logs/
├── c-data-0.csv        # Client 0 training/evaluation metrics
├── c-data-1.csv        # Client 1 training/evaluation metrics
├── ...
├── c-data-N.csv        # Client N training/evaluation metrics
└── server-data.csv     # Server-side aggregated metrics
```

### Logged Metrics

#### Client Metrics
- Round number
- Training accuracy/loss (`g_fit_acc`, `g_fit_loss`)
- Evaluation accuracy/loss (`g_eval_acc`, `g_eval_loss`)
- Selection status (`selected`)
- Participation state (`participating_state`, `desired_state`)
- Strategy-specific metrics (e.g., `qk` for MaxFL, `willing` for LetsFed)

#### Server Metrics
- Aggregated loss/accuracy per round
- Number of clients selected
- Selection method used
- Aggregation method used

### Analysis

Use the provided Jupyter notebook for visualization and analysis:

```bash
jupyter notebook test/teste.ipynb
```

The notebook includes:
- Training convergence plots
- Client participation patterns
- Comparison between strategies
- Statistical analysis

## 🐳 Docker Configuration

### Container Architecture

```mermaid
graph TB
    subgraph "Docker Network: fl_network"
        Server[Server Container<br/>rfl_server:9999]
        C0[Client 0<br/>CID=0]
        C1[Client 1<br/>CID=1]
        CN[Client N<br/>CID=N]
    end

    subgraph "Volumes"
        Logs[./logs:/logs]
        Data[./data:/data]
        Config[./conf:/conf]
    end

    Server ---|gRPC| C0
    Server ---|gRPC| C1
    Server ---|gRPC| CN

    Server -.->|mount| Logs
    Server -.->|mount| Config
    C0 -.->|mount| Logs
    C0 -.->|mount| Data
    C1 -.->|mount| Data
    CN -.->|mount| Data
```

### GPU Support

For experiments with GPU acceleration:

```bash
# Build GPU images
docker-compose -f docker-compose.gpu.yml build

# Run with GPU
docker-compose -f docker-compose.gpu.yml up
```

**Requirements:**
- NVIDIA GPU
- NVIDIA Docker runtime
- CUDA-compatible drivers

### Scaling Clients

Modify the number of clients in `docker-compose.yml` or use the manager:

```bash
python utils/docker_compose_manager.py --n-clients 50
docker-compose up
```

## ⚙️ Configuration Reference

### Complete Configuration Example

```yaml
# General settings
rounds: 10                    # Number of federated learning rounds
n_clients: 30                 # Total number of clients
init_clients: 1.0             # Fraction of clients to initialize
gpu: false                    # Enable GPU acceleration
log_path: logs                # Path for log files

# Server configuration
server:
  ip: 0.0.0.0
  port: 9999

  # Aggregation strategy
  aggregation_method:
    name: fedavg              # Options: fedavg, maxfl
    # MaxFL parameters (if name=maxfl)
    # epsilon: 10
    # learning_rate: 0.01

  # Client selection strategy
  selection_method:
    name: letsfed             # Options: random, deev, poc, round_robin, letsfed
    perc_of_clients: 0.3      # Percentage of clients to select per round
    # DEEV parameters (if name=deev)
    # decay: 0.95
    # LetsFed parameters (if name=letsfed)
    # participating_method: random
    # non_participating_method: poc

# Client configuration
client:
  epochs: 5                   # Local training epochs
  learning_rate: 0.0001       # Learning rate

  # Training strategy
  training_strategy:
    name: letsfed             # Options: normal, letsfed, maxfl, fedper, qffl
    # LetsFed parameters (if name=letsfed)
    # threshold: 1.0

  participating: true         # Initial participation state

# Dataset configuration
dataset:
  dataset: fashion_mnist      # Options: fashion_mnist, cifar10, mnist
  path: logs                  # Dataset storage path
  batch_size: 32              # Batch size for training

# Model configuration
model:
  model: cnn                  # Options: cnn, dnn
  # Model-specific parameters...
```

### Environment Variables

Override configuration with environment variables:

```bash
# Client ID (required for clients)
export CID=0

# State variables (LetsFed)
export IDLE_STATE=0
export EXPLORING_STATE=1

# Run client
python -m client.strategies_manager
```

## 🎓 Research & Publications

This framework was developed as part of research on Federated Learning with dynamic client participation.

### Citation

If you use this framework in your research, please cite:

```bibtex
@INPROCEEDINGS{10903243,
  author={Jarczewski, Rafael O. and Cerqueira, Eduardo and Bittencourt, Luiz F. and Loureiro, Antonio A. F. and Villas, Leandro A. and de Souza, Allan M.},
  booktitle={2024 International Conference on Machine Learning and Applications (ICMLA)},
  title={Let's Federate - Effective Communication Strategy for Dynamic Client Participation},
  year={2024},
  pages={361-368},
  doi={10.1109/ICMLA61862.2024.00055}
}
```

## 🛠️ Development

### Code Quality Tools

The project uses several tools to maintain code quality:

```bash
# Linting and formatting
ruff check .
ruff format .

# Pre-commit hooks
pre-commit install
pre-commit run --all-files
```

### Project Philosophy

This framework follows key software engineering principles:

- **🎯 SOLID Principles**
  - Single Responsibility: Each class has one clear purpose
  - Open/Closed: Extend via new classes, not modifications
  - Liskov Substitution: Strategies are interchangeable
  - Interface Segregation: Focused, minimal interfaces
  - Dependency Inversion: Depend on abstractions, not concretions

- **🎨 Design Patterns**
  - Strategy Pattern for algorithm selection
  - Factory Pattern for object creation
  - Builder Pattern for complex construction
  - Chain of Responsibility for driver pipeline

- **📦 Modular Design**
  - Clear separation of concerns
  - Composable components
  - Minimal coupling, high cohesion

### Testing

```bash
# Run unit tests (if available)
pytest tests/

# Run experiment with test configuration
docker-compose -f docker-compose.test.yml up
```

## 🤝 Contributing

Contributions are welcome! To contribute:

1. **Fork the repository**
2. **Create a feature branch**: `git checkout -b feature/amazing-feature`
3. **Follow the coding style**: Use `ruff` for formatting
4. **Add tests** if applicable
5. **Update documentation**: Keep README and docstrings current
6. **Commit changes**: `git commit -m 'Add amazing feature'`
7. **Push to branch**: `git push origin feature/amazing-feature`
8. **Open a Pull Request**

### Contribution Guidelines

- Follow existing design patterns
- Maintain type hints throughout
- Write clear docstrings
- Keep commits atomic and well-described
- Update relevant documentation

## � Additional Resources

### Federated Learning
- [Flower Documentation](https://flower.dev/docs/) - FL framework used
- [Federated Learning Book](https://www.federated-learning.com/) - Comprehensive guide
- [FedAvg Paper](https://arxiv.org/abs/1602.05629) - Original federated averaging paper

### Design Patterns
- [Refactoring Guru](https://refactoring.guru/design-patterns) - Pattern catalog
- [Python Design Patterns](https://python-patterns.guide/) - Python-specific patterns

### Docker
- [Docker Best Practices](https://docs.docker.com/develop/dev-best-practices/)
- [Docker Compose](https://docs.docker.com/compose/)

## 🐛 Troubleshooting

### Common Issues

**Issue**: Clients can't connect to server
```bash
# Check network connectivity
docker network ls
docker network inspect letsfed_fl_network

# Verify server is running
docker logs letsfed-rfl_server-1
```

**Issue**: Out of memory errors
```bash
# Reduce number of clients or batch size in config.yaml
n_clients: 10  # Reduce from 30
dataset:
  batch_size: 16  # Reduce from 32
```

**Issue**: GPU not detected
```bash
# Verify NVIDIA Docker runtime
docker run --rm --gpus all nvidia/cuda:11.0-base nvidia-smi

# Check docker-compose.gpu.yml has proper GPU configuration
```

**Issue**: Import errors in containers
```bash
# Rebuild containers with --no-cache
docker-compose build --no-cache
```

## 📊 Performance Benchmarks

Typical performance metrics on standard hardware:

| Metric | Value |
|--------|-------|
| **Build Time** | ~3-5 minutes |
| **Container Size** | ~2.0 GB |
| **Startup Time** | ~10-15 seconds |
| **Round Duration** (30 clients) | ~2-3 minutes |
| **Memory Usage** (per client) | ~500 MB |

*Tested on: Intel i7, 16GB RAM, Docker 20.10*

## 🔒 Security Considerations

This is a **research framework** and should not be used in production without proper security review:

- No authentication/authorization implemented
- Network communication is not encrypted by default
- Data is not encrypted at rest
- No input validation for malicious data

For production use, consider:
- Adding TLS/SSL for communication
- Implementing client authentication
- Encrypting datasets
- Adding differential privacy mechanisms

## 📝 License

[Add your license information here - e.g., MIT, Apache 2.0, GPL]

## 🙏 Acknowledgments

This research was supported by:
- [Add funding sources]
- [Add institutional support]
- [Add collaborators]

Special thanks to the [Flower](https://flower.dev/) team for providing an excellent federated learning framework.

## 📞 Contact

- **Author**: Rafael O. Jarczewski
- **Institution**: [Add institution]
- **Email**: [Add email]
- **GitHub Issues**: For bug reports and feature requests

---

**⚠️ Note**: This is a research framework under active development. Some features may be experimental or incomplete. See issues for known limitations and planned improvements.

**🚀 Happy Federating!**
