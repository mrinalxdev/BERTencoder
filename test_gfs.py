import unittest
from master import GFSMaster
from client import GFSClient
from chunkserver import ChunkServer, Chunk

class TestGFSSimulation(unittest.TestCase):
    def setUp(self):
        self.master = GFSMaster()
        self.client = GFSClient(self.master)

    def test_write_file(self):
        result = self.client.write_file("test1.txt", "Test data")
        self.assertTrue(result)
        self.assertIn("test1.txt", self.master.metadata)
        chunk_id = self.master.metadata["test1.txt"][0]
        self.assertIn(chunk_id, self.master.chunk_locations)

    def test_read_file(self):
        self.client.write_file("test2.txt", "Read me")
        data = self.client.read_file("test2.txt")
        self.assertEqual(data, "Read me")

    def test_read_non_existent_file(self):
        data = self.client.read_file("nonexistent.txt")
        self.assertIsNone(data)

    def test_chunk_replication(self):
        self.client.write_file("test3.txt", "Replicate me")
        chunk_id = self.master.metadata["test3.txt"][0]
        locations = self.master.chunk_locations[chunk_id]
        self.assertEqual(len(locations), 3)  # Should replicate to all 3 servers

        for server_id in locations:
            chunk = self.master.chunk_servers[server_id].get_chunk(chunk_id)
            self.assertIsNotNone(chunk)
            self.assertEqual(chunk.data, "Replicate me")

    def test_chunk_server_storage(self):
        chunk = Chunk("chunk_test", "Test data", 9)
        cs = ChunkServer("test_cs")
        cs.store_chunk(chunk)
        retrieved = cs.get_chunk("chunk_test")
        self.assertEqual(retrieved.data, "Test data")
        self.assertTrue(cs.delete_chunk("chunk_test"))
        self.assertIsNone(cs.get_chunk("chunk_test"))

if __name__ == "__main__":
    unittest.main()
