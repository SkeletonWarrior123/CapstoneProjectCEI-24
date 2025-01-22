import os
import pandas as pd
import asyncio
from pinecone import Pinecone, ServerlessSpec
from dotenv import load_dotenv
from langchain_pinecone import PineconeEmbeddings

# Load environment variables
load_dotenv()

# Initialize Pinecone
pinecone_api_key = os.getenv("PINECONE_API_KEY")
pc = Pinecone(api_key=pinecone_api_key)

# Define Pinecone index name and embedding model
PINECONE_INDEX_NAME = "car-data-index"
EMBEDDING_MODEL = "multilingual-e5-large"

async def generate_embedding(text, embeddings):
    return embeddings.embed_query(text)

def process_csv_and_store_data(csv_path):
    if not os.path.exists(csv_path):
        print("CSV file not found!")
        return

    chunk_size = 1000
    chunks = pd.read_csv(csv_path, chunksize=chunk_size)
    embeddings = PineconeEmbeddings(model=EMBEDDING_MODEL)

    # Create the index if it doesn't exist
    if PINECONE_INDEX_NAME not in pc.list_indexes().names():
        pc.create_index(
            name=PINECONE_INDEX_NAME,
            dimension=1024,  # Ensure this matches the embedding dimension
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )

    index = pc.Index(PINECONE_INDEX_NAME)

    total_rows = 0
    uploaded_ids = []

    for chunk in chunks:
        chunk = chunk.dropna()  # Drop rows with missing data
        texts = chunk.apply(lambda row: (
            f"Model: {row['model']}, Year: {row['year']}, Price: {row['price']}, "
            f"Transmission: {row['transmission']}, Mileage: {row['mileage']}, Fuel Type: {row['fuelType']}, "
            f"Tax: {row['tax']}, MPG: {row['mpg']}, Engine Size: {row['engineSize']}, Manufacturer: {row['Manufacturer']}"
        ), axis=1).tolist()

        vector_data = []

        async def process_texts():
            for i, text in enumerate(texts):
                try:
                    embedding = await generate_embedding(text, embeddings)
                    vector_id = f"row-{total_rows + i}"
                    vector_data.append({
                        "id": vector_id,
                        "values": embedding,
                        "metadata": {
                            "row_index": total_rows + i,
                            "text": text[:512]  # Truncate if too long
                        }
                    })
                    uploaded_ids.append(vector_id)
                except Exception as e:
                    print(f"Skipping row due to embedding error: {e}")
                    continue

        asyncio.run(process_texts())

        # Batch upload vectors to Pinecone
        batch_size = 50
        for i in range(0, len(vector_data), batch_size):
            batch = vector_data[i:i + batch_size]
            try:
                index.upsert(vectors=batch)
                print(f"Upserted {len(batch)} vectors")
            except Exception as e:
                print(f"Batch upload failed: {e}")

        total_rows += len(chunk)

    print(f"Total rows processed: {total_rows}")
    print(f"Uploaded indexes: {uploaded_ids}")

if __name__ == "__main__":
    # Use raw string or double backslashes to avoid escape sequence issues
    csv_path = r"C:\carvilla-v1.0\cardataset.csv"
    process_csv_and_store_data(csv_path)
