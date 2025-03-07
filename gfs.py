"""Usage:
    python gfs.py --help          # Show help message
    python gfs.py write <file> <data>  # Write data to a file
    python gfs.py read <file>     # Read data from a file
    python gfs.py list            # List all files in the system
"""

import argparse
import sys
from typing import Optional
from master import GFSMaster
from client import GFSClient

class GFSExample:
    def __init__(self):
        """Initialize the GFS example with a master and client."""
        self.master = GFSMaster()
        self.client = GFSClient(self.master)

    def write_file(self, filename: str, data: str) -> bool:
        """Write data to a file and demonstrate replication.

        Args:
            filename: Name of the file to write
            data: Data to write to the file
        Returns:
            bool: True if write was successful, False otherwise
        """
        try:
            success = self.client.write_file(filename, data)
            if success:
                print(f"Successfully wrote '{data}' to '{filename}'")
                self._print_chunk_locations(filename)
            else:
                print(f"Failed to write to '{filename}'")
            return success
        except Exception as e:
            print(f"Error writing to '{filename}': {e}")
            return False

    def read_file(self, filename: str) -> Optional[str]:
        """Read data from a file and show where it’s stored.

        Args:
            filename: Name of the file to read
        Returns:
            str or None: The file content if found, None otherwise
        """
        try:
            data = self.client.read_file(filename)
            if data is not None:
                print(f"Read from '{filename}': '{data}'")
                self._print_chunk_locations(filename)
            else:
                print(f"File '{filename}' not found")
            return data
        except Exception as e:
            print(f"Error reading '{filename}': {e}")
            return None

    def list_files(self) -> None:
        """List all files in the system with their chunk locations."""
        try:
            files = self.master.metadata.keys()
            if not files:
                print("No files in the system")
                return
            print("Files in GFS:")
            for filename in files:
                print(f"- {filename}")
                self._print_chunk_locations(filename)
        except Exception as e:
            print(f"Error listing files: {e}")

    def _print_chunk_locations(self, filename: str) -> None:
        """Helper to print chunk locations for a file."""
        if filename in self.master.metadata:
            chunk_id = self.master.metadata[filename][0]
            locations = self.master.chunk_locations.get(chunk_id, [])
            print(f"  Chunk '{chunk_id}' stored on: {', '.join(locations)}")

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="GFS Simulation Example")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    write_parser = subparsers.add_parser("write", help="Write data to a file")
    write_parser.add_argument("filename", help="Name of the file")
    write_parser.add_argument("data", help="Data to write")
    read_parser = subparsers.add_parser("read", help="Read data from a file")
    read_parser.add_argument("filename", help="Name of the file")
    list_parser = subparsers.add_parser("list", help="List all files")

    return parser.parse_args()

def main():
    """Main entry point for the GFS example."""
    args = parse_args()
    example = GFSExample()

    if args.command == "write":
        example.write_file(args.filename, args.data)
    elif args.command == "read":
        example.read_file(args.filename)
    elif args.command == "list":
        example.list_files()
    else:
        print("Running default demo...")
        example.write_file("demo.txt", "Hello, GFS!")
        example.read_file("demo.txt")
        example.write_file("example.txt", "Another file")
        example.list_files()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nExiting...")
        sys.exit(0)
    except Exception as e:
        print(f"Unexpected error: {e}")
        sys.exit(1)
