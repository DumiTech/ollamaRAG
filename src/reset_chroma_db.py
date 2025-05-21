import chromadb
import os
import shutil

from chromadb.config import Settings


CHROMA_PATH = os.path.abspath("../chroma_db/") # ChromaDB directory data


def reset_chroma_db(persist_directory=CHROMA_PATH):
    """Completely resets ChromaDB by deleting all files and recreating the directory"""
    try:
        # First reset through client
        client = chromadb.PersistentClient(
            path=persist_directory,
            settings=Settings(allow_reset=True)
        )
        client.reset()
        
        # Then delete the entire directory
        if os.path.exists(persist_directory):
            shutil.rmtree(persist_directory)
            print(f"Deleted all ChromaDB files.")
        
        # Recreate empty directory structure
        os.makedirs(persist_directory, exist_ok=True)
        
        print("Successfully reset ChromaDB")
        
    except Exception as e:
        print(f"Reset failed: {str(e)}")
        raise


confirm = input("WARNING: This will DELETE ALL DATA. Continue? (y/n): ")
if confirm.lower() == 'y':
    reset_chroma_db()
else:
    print("Reset cancelled")
