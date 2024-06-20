from typing import List

def is_select_by_server(cid: str, server_selection: List[str]):
    return cid in server_selection