import logging
import os
import sys
from typing import Optional


class Logger:
    """
    Custom Logger class that combines standard Python logging with CSV metrics logging.

    This logger provides:
    - Standard logging methods (info, warning, error, debug, critical) for application logs
    - Custom log_metrics() method for writing experiment metrics to CSV files
    - Singleton pattern to ensure consistent logging across the application

    Each module should create its own logger instance using Logger(__name__) to
    maintain proper hierarchical logging with module names.
    """

    # Class-level storage for logger instances (one per module name)
    _loggers: dict[str, "Logger"] = {}

    # Shared configuration across all loggers
    _metrics_folder: str = "./logs"
    _logging_configured: bool = False

    def __new__(cls, name: str = __name__) -> "Logger":
        """
        Singleton factory: returns existing logger for the given name or creates a new one.

        Args:
            name: Logger name (typically __name__ from calling module)

        Returns:
            Logger instance for the given name
        """
        if name not in cls._loggers:
            instance = super(Logger, cls).__new__(cls)
            cls._loggers[name] = instance
        return cls._loggers[name]

    def __init__(self, name: str = __name__, metrics_folder: str = "") -> None:
        """
        Initialize the logger with both standard logging and metrics capabilities.

        Args:
            name: Logger name (typically __name__ from calling module)
            metrics_folder: Optional subfolder for metrics files (relative to ./logs)
        """
        # Avoid re-initialization of existing logger instances
        if hasattr(self, "_initialized") and self._initialized:
            return

        self.name = name

        # Configure standard Python logger
        self._logger = logging.getLogger(name)

        # Configure logging format and handlers only once
        if not Logger._logging_configured:
            self._configure_logging()
            Logger._logging_configured = True

        # Set metrics folder
        if metrics_folder:
            Logger._metrics_folder = f"./logs{metrics_folder}"

        self._initialized = True

    def _configure_logging(self) -> None:
        """Configure the root logger with handlers and formatters."""
        # Get root logger to configure globally
        root_logger = logging.getLogger()
        root_logger.setLevel(logging.INFO)

        # Avoid duplicate handlers
        if root_logger.handlers:
            return

        # Create console handler with formatting
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)

        # Create formatter
        formatter = logging.Formatter(
            fmt="%(asctime)s - %(name)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
        )
        console_handler.setFormatter(formatter)

        # Add handler to root logger
        root_logger.addHandler(console_handler)

        # Also configure this logger
        self._logger.setLevel(logging.INFO)

    # Standard logging methods delegated to Python's logging module

    def debug(self, msg: str, *args, **kwargs) -> None:  # noqa: ANN002
        """Log a debug message."""
        self._logger.debug(msg, *args, **kwargs)

    def info(self, msg: str, *args, **kwargs) -> None:  # noqa: ANN002
        """Log an info message."""
        self._logger.info(msg, *args, **kwargs)

    def warning(self, msg: str, *args, **kwargs) -> None:  # noqa: ANN002
        """Log a warning message."""
        self._logger.warning(msg, *args, **kwargs)

    def error(self, msg: str, *args, **kwargs) -> None:  # noqa: ANN002
        """Log an error message."""
        self._logger.error(msg, *args, **kwargs)

    def critical(self, msg: str, *args, **kwargs) -> None:  # noqa: ANN002
        """Log a critical message."""
        self._logger.critical(msg, *args, **kwargs)

    def exception(self, msg: str, *args, **kwargs) -> None:  # noqa: ANN002
        """Log an exception with traceback."""
        self._logger.exception(msg, *args, **kwargs)

    # Metrics logging methods (CSV files)

    def _write_metrics(self, filename: str, data: list, header: Optional[list[str]] = None) -> None:
        """
        Internal method to write metrics to a CSV file.

        Args:
            filename: Name of the CSV file
            data: List of data entries to log
            header: Optional list of header names
        """
        file_path = f"{Logger._metrics_folder}{filename}"

        if header is not None:
            with open(file_path, "w") as file:
                file.write(f"{','.join(header)}\n")
                file.write(f"{','.join([str(d) for d in data])}\n")
            return

        with open(file_path, "a") as file:
            file.write(f"{','.join([str(d) for d in data])}\n")

    def log_metrics(self, filename: str, data: dict) -> None:
        """
        Log experiment metrics to a CSV file.

        This method writes metrics (accuracy, loss, etc.) to CSV files for later analysis.
        Creates the file with headers if it doesn't exist, otherwise appends data.

        Args:
            filename: Name of the CSV file (e.g., "/s-data.csv", "/c-data.csv")
            data: Dictionary of metric names and values to log

        Example:
            logger.log_metrics("/s-data.csv", {
                "round": 1,
                "accuracy": 0.95,
                "loss": 0.05
            })
        """
        header = list(data.keys())
        values = list(data.values())

        # Validation
        if len(header) != len(values):
            self.error("Metrics logging error: header and data length mismatch")
            self.error(f"header ({len(header)}): {header}")
            self.error(f"values ({len(values)}): {values}")
            raise ValueError("Header and data length mismatch in metrics logging")

        file_path = f"{Logger._metrics_folder}{filename}"

        # Create directory and file with header if it doesn't exist
        if not os.path.exists(file_path):
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            self._write_metrics(filename, values, header=header)
            self.debug(f"Created new metrics file: {file_path}")
            return

        # Append to existing file
        self._write_metrics(filename=filename, data=values)

    def set_level(self, level: int) -> None:
        """
        Set the logging level.

        Args:
            level: Logging level (e.g., logging.DEBUG, logging.INFO)
        """
        self._logger.setLevel(level)
