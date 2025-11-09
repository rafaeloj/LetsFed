import os


class Logger:
    """
    Logger class for logging events to a file.
    Implements the Singleton pattern to ensure only one instance exists.
    """

    _instance = None
    _initialized = False

    def __new__(cls, logger_foulder: str = "") -> "Logger":
        if cls._instance is None:
            cls._instance = super(Logger, cls).__new__(cls)
        return cls._instance

    def __init__(self, logger_foulder: str = "") -> None:
        if not self._initialized:
            self.logger_foulder = f"./logs{logger_foulder}"
            self._initialized = True

    def _log(self, filename: str, data: list, header: list[str] = None) -> None:
        """
        Append data to a CSV file, creating it with a header if it doesn't exist.

        Args:
            filename: Name of the CSV file
            data: List of data entries to log
            header: Optional list of header names
        """
        file_path = f"{self.logger_foulder}{filename}"
        if header is not None:
            with open(file_path, "w") as file:
                file.write(f"{','.join(header)}\n")
                file.write(f"{','.join([f'{d}' for d in data])}\n")
            return

        with open(file_path, "a") as file:
            file.write(f"{','.join([f'{d}' for d in data])}\n")

    def log(self, filename: str, data: dict) -> None:
        """
        Log data to a CSV file, creating it with a header if it doesn't exist.

        Args:
            filename: Name of the CSV file
            data: Dictionary of data entries to log
        """
        header = data.keys()
        data = data.values()
        if isinstance(header, list) and len(header) != len(data.values()):
            print("Log aggregate problem")
            print(f"header ({len(header)}): {header}")
            print(f"Data: (len({data})){data}")
            exit(992)

        file_path = f"{self.logger_foulder}{filename}"
        if not os.path.exists(file_path):
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            self._log(filename, data, header=header)
            return

        self._log(filename=filename, data=data)
