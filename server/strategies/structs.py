from dataclasses import MISSING, dataclass

from .aggregate_method.structs import AggregationMethodConfig
from .client_selection_method.structs import SelectionMethodConfig
from .parameters_strategy.structs import ParametersStrategyConfig


@dataclass
class ServerConfig:
    """Server configuration including network and FL parameters."""

    ip: str = MISSING
    port: int = MISSING

    aggregation_method: AggregationMethodConfig = MISSING
    selection_method: SelectionMethodConfig = MISSING
    parameters_strategy: ParametersStrategyConfig = MISSING
