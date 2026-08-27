"""Embed each chunk (via a local OpenAI-compatible embeddings endpoint,
not local to this machine) and load it straight into a Chroma collection.

The raw vector is never written back to the chunks jsonl - it's only
meaningful to a vector index, so it goes directly into Chroma. Stored
alongside it is the chunk's return_text, the text a hit is answered with,
which leaves the store self-contained: serving needs it and nothing else. A chunk is skipped only if its chunk_id
already exists in the collection AND its text is unchanged (compared via
a stored hash) - a new chunk_id gets added, and an existing chunk_id whose
text changed (edited explanation, a chunk_type's field recipe changed,
etc.) gets re-embedded and upserted in place.
"""

import hashlib
import json
import sys
import time
from pathlib import Path

import chromadb
import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "config"))
from config import (
    API_KEY,
    CHROMA_DISTANCE_METRIC,
    EMBEDDING_BASE_URL,
    EMBEDDING_MODEL,
    VERIFY_SSL,
    chroma_collection_name,
    chroma_dir,
    chunk_jsonl_paths,
    chunked_text_dir,
)

if VERIFY_SSL is False:
    requests.packages.urllib3.disable_warnings(requests.packages.urllib3.exceptions.InsecureRequestWarning)

SAVE_EVERY = 20
MAX_RETRIES = 3


def read_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def chunk_id(chunk):
    """Unique per stored vector, so a chunk that was split does not
    overwrite itself piece by piece.

    The part is only appended past the first, which leaves the id of an
    unsplit chunk exactly what it was - lowering CHUNK_MAX_CHARS re-embeds
    what it actually divided and nothing else."""
    part = chunk.get("part") or 0
    suffix = f"::{part}" if part else ""
    return f"{chunk['record_id']}::{chunk['chunk_type']}{suffix}"


def text_hash(chunk):
    """Covers both texts, so editing a return_fields recipe reloads the
    chunk even though what it is embedded on has not moved. Hashing only
    the embedded text would leave the stored answer stale - the more
    confusing failure, since retrieval would still look right."""
    payload = chunk["embedded_text"] + "\x00" + chunk.get("return_text", "")
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def call_embedding(text, api_key):
    response = requests.post(
        f"{EMBEDDING_BASE_URL}/embeddings",
        headers={"Authorization": f"Bearer {api_key}"},
        json={"model": EMBEDDING_MODEL, "input": text},
        timeout=60,
        verify=VERIFY_SSL,
    )
    if not response.ok:
        raise requests.HTTPError(
            f"{response.status_code} {response.reason} for {response.url}: {response.text[:500]}",
            response=response,
        )
    return response.json()["data"][0]["embedding"]


def call_embedding_with_retry(chunk, api_key):
    for attempt in range(MAX_RETRIES):
        try:
            return call_embedding(chunk["embedded_text"], api_key)
        except Exception as e:
            if attempt == MAX_RETRIES - 1:
                print(f"  FAILED {chunk['record_id']} ({chunk['chunk_type']}): {e}")
                return None
            time.sleep(2 ** attempt)


def fetch_existing_hashes(collection, ids):
    existing_hashes = {}
    for i in range(0, len(ids), 200):
        batch = collection.get(ids=ids[i:i + 200], include=["metadatas"])
        for id_, metadata in zip(batch["ids"], batch["metadatas"]):
            existing_hashes[id_] = metadata.get("text_hash")
    return existing_hashes


def find_changed_or_new_chunks(collection, chunks):
    existing_hashes = fetch_existing_hashes(collection, [chunk_id(c) for c in chunks])
    return [c for c in chunks if existing_hashes.get(chunk_id(c)) != text_hash(c)]


def load_chunks(collection, chunks, api_key):
    pending = find_changed_or_new_chunks(collection, chunks)
    print(f"{len(chunks)} chunks, {len(pending)} new or changed, need embedding + loading")

    batch_ids, batch_embeddings, batch_documents, batch_metadatas = [], [], [], []

    def flush():
        if batch_ids:
            collection.upsert(ids=batch_ids, embeddings=batch_embeddings, documents=batch_documents, metadatas=batch_metadatas)
            batch_ids.clear()
            batch_embeddings.clear()
            batch_documents.clear()
            batch_metadatas.clear()

    for i, chunk in enumerate(pending):
        embedding = call_embedding_with_retry(chunk, api_key)
        if embedding:
            batch_ids.append(chunk_id(chunk))
            batch_embeddings.append(embedding)
            # The document is what a hit is answered with, not what it was
            # found by: the store then holds everything a search needs.
            batch_documents.append(chunk.get("return_text") or chunk["embedded_text"])
            batch_metadatas.append({
                "record_id": chunk["record_id"],
                "chunk_type": chunk["chunk_type"],
                "text_hash": text_hash(chunk),
            })
            print(f"  [{i + 1}/{len(pending)}] {chunk['record_id']} ({chunk['chunk_type']})")

        if (i + 1) % SAVE_EVERY == 0:
            flush()

    flush()


def main():

    chunk_paths = chunk_jsonl_paths()
    if not chunk_paths:
        print(f"No chunks under {chunked_text_dir()} - run parse_chunks.py first")
        return

    chunks = []
    for path in chunk_paths:
        file_chunks = read_jsonl(path)
        chunks.extend(file_chunks)
        print(f"Read {len(file_chunks)} chunks from {path.name}")

    client = chromadb.PersistentClient(path=str(chroma_dir()))
    collection = client.get_or_create_collection(
        name=chroma_collection_name(),
        metadata={"hnsw:space": CHROMA_DISTANCE_METRIC},
    )

    load_chunks(collection, chunks, API_KEY)
    print(f"Done. Collection '{collection.name}' now has {collection.count()} chunks.")


if __name__ == "__main__":
    main()
