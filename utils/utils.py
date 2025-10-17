class Utils:
    """
    Utility functions for the project.
    """

    @staticmethod
    def is_select_by_server(cid: str, server_selection: list[str]) -> bool:
        """
        Check if a client ID is selected by the server.

        Args:
            cid: Client ID to check
            server_selection: List of client IDs selected by the server

        Returns:
            True if the client ID is in the server selection list, False otherwise.
        """
        return str(cid) in server_selection
