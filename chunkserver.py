from typing import Dict, List, Optional
from dataclasses import dataclass

@dataclass
class Chunk:
    chunk_id: str
    data: str
    size: int

class ChunkServer:
    def __init__(self, server_id: str):
        self.server_id = server_id
        self.chunks: Dict[str, Chunk] = {}
        self.replicas: Dict[str, List[str]] = {}  # Maps chunk_id to list of server_ids

    def store_chunk(self, chunk: Chunk) -> bool:
        """Store a chunk on this server."""
        self.chunks[chunk.chunk_id] = chunk
        return True

    def get_chunk(self, chunk_id: str) -> Optional[Chunk]:
        """Retrieve a chunk by ID."""
        return self.chunks.get(chunk_id)

    def replicate_to(self, chunk_id: str, target_server: 'ChunkServer') -> bool:
        """Replicate a chunk to another server."""
        if chunk_id not in self.chunks:
            return False
        target_server.store_chunk(self.chunks[chunk_id])
        if chunk_id not in self.replicas:
            self.replicas[chunk_id] = []
        self.replicas[chunk_id].append(target_server.server_id)
        return True

    def delete_chunk(self, chunk_id: str) -> bool:
        """Delete a chunk from this server."""
        if chunk_id in self.chunks:
            del self.chunks[chunk_id]
            return True
        return False
