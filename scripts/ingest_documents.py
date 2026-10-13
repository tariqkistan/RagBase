import asyncio
from pathlib import Path

from dotenv import load_dotenv

from ragbase.database import embed_chunks, setup_database, store_chunks
from ragbase.ingestion import load_and_chunk

load_dotenv()


async def main():
    await setup_database()

    documents_dir = Path("documents")
    if not documents_dir.exists():
        print(f"Documents directory '{documents_dir}' not found.")
        return

    total_chunks = 0
    for filepath in documents_dir.iterdir():
        if filepath.is_file():
            print(f"Processing {filepath.name}...")
            chunks = load_and_chunk(str(filepath))
            chunks_with_embeddings = await embed_chunks(chunks)
            count = await store_chunks(chunks_with_embeddings)
            total_chunks += count
            print(f"  Stored {count} chunks")

    print(f"\nTotal chunks stored: {total_chunks}")


if __name__ == "__main__":
    asyncio.run(main())
