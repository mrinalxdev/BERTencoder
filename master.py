from typing import Dict, List, Optional
import random
from chunkserver import ChunkServer, Chunk

class GFSMaster:
    def __init__(self):
        self.metadata: Dict[str, List[str]] = {}  # filename -> list of chunk_ids
        self.chunk_locations: Dict[str, List[str]] = {}  # chunk_id -> list of server_ids
        self.chunk_servers: Dict[str, ChunkServer] = {
            "cs1": ChunkServer("cs1"),
            "cs2": ChunkServer("cs2"),
            "cs3": ChunkServer("cs3")
        }

    def open_file(self, filename: str, data: str = None) -> List[str]:
        """Handle client file open request, return chunk locations."""
        if filename not in self.metadata:
            if data:  # Write operation
                chunk_id = f"chunk_{random.randint(1000, 9999)}"
                self.metadata[filename] = [chunk_id]
                self.chunk_locations[chunk_id] = ["cs1"]  # Default to cs1
                self.chunk_servers["cs1"].store_chunk(Chunk(chunk_id, data, len(data)))
                self._replicate_chunk(chunk_id)
            else:  # Read operation for non-existent file
                return []
        return self.chunk_locations.get(self.metadata[filename][0], [])

    def _replicate_chunk(self, chunk_id: str):
        """Replicate chunk across chunk servers for fault tolerance."""
        primary_server = self.chunk_servers[self.chunk_locations[chunk_id][0]]
        for server_id, server in self.chunk_servers.items():
            if server_id not in self.chunk_locations[chunk_id]:
                primary_server.replicate_to(chunk_id, server)
                self.chunk_locations[chunk_id].append(server_id)

    def get_chunk_servers(self) -> Dict[str, ChunkServer]:
        """Return all chunk servers."""
        return self.chunk_servers
