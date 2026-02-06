import chromadb
from sentence_transformers import SentenceTransformer
from datetime import datetime
from typing import List, Union, Optional



class MemoryStore:
    def __init__(self):
        self.client = chromadb.PersistentClient(
            path="./chroma_db"
        )

        self.collection = self.client.get_or_create_collection(
            name="linkedin_memory"
        )

        self.embedder = SentenceTransformer("all-MiniLM-L6-v2")

    def add_memory(
        self,
        user_id: str,
        text: Optional[Union[str, List[str]]],
        memory_type: str = "generic"
    ):
        """
        Stores memory in vector DB.
        Summarize the following LinkedIn post for long-term memory storage.

        Rules:
        - 3–5 bullet points
        - Capture the core topic and key ideas
        - Preserve tone and intent
        - No emojis
        - No hashtags
        - No CTA
        - No meta commentary
        - Do NOT say "this post discusses"
        - If content is empty or meaningless, return EMPTY
        """

        if not text:
            return  # Nothing to store

        # Normalize to list
        if isinstance(text, str):
            texts = [text]
        else:
            texts = text

        for item in texts:
            if not item.strip():
                continue

            embedding = self.embedder.encode(item).tolist()

            self.collection.add(
                documents=[item],
                embeddings=[embedding],
                metadatas=[{
                    "user_id": user_id,
                    "memory_type": memory_type,
                    "timestamp": str(datetime.now())
                }],
                ids=[f"{user_id}_{datetime.now().timestamp()}"]
            )


    def get_relevant_memory(
        self,
        user_id: str,
        query: str,
        k: int = 7
    ) -> List[str]:

        query_embedding = self.embedder.encode(query).tolist()

        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=k,
            where={"user_id": user_id}
        )

        if results and results.get("documents"):
            return results["documents"][0]

        return []

    

    ## new on eadded
    def clear_user_memory(self, user_id: str):
        self.collection.delete(where={"user_id": user_id})




memory_store = MemoryStore()
