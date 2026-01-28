"""
Test MongoDB connection
"""

import sys
from pathlib import Path

# Add parent directory to path
parent_dir = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(parent_dir))

from src.database.mongodb_client import get_mongo_client

def main():
    print("Testing MongoDB connection...")
    client = get_mongo_client()
    
    if client.db is not None:
        print("✅ Connection successful!")
        print(f"Database: {client.db.name}")
        
        # List collections
        collections = client.db.list_collection_names()
        print(f"Collections: {collections}")
    else:
        print("❌ Connection failed!")

if __name__ == "__main__":
    main()
