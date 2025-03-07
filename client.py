from typing import Optional
from master import GFSMaster

class GFSClient:
    def __init__(self, master: GFSMaster):
        self.master = master

    def write_file(self, filename: str, data: str) -> bool:
        """Write data to a file."""
        chunk_locations = self.master.open_file(filename, data)
        return bool(chunk_locations)

    def read_file(self, filename: str) -> Optional[str]:
        """Read data from a file."""
        chunk_locations = self.master.open_file(filename)
        if not chunk_locations:
            return None
        chunk_server = self.master.get_chunk_servers()[chunk_locations[0]]
        chunk_id = self.master.metadata[filename][0]
        chunk = chunk_server.get_chunk(chunk_id)
        return chunk.data if chunk else None
