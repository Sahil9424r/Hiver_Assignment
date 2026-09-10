"""
ChromaDB Vector Store for indexing and retrieving historical brand resolutions.
Uses local dense embeddings (all-MiniLM-L6-v2) for semantic search.
"""

import json
import logging
from typing import Dict, List, Optional
import chromadb
from chromadb.config import Settings

from src.config import CHROMA_DIR, HISTORICAL_RESOLUTIONS_PATH

logger = logging.getLogger(__name__)

COLLECTION_NAME = "apple_support_resolutions"


class ChromaResolutionStore:
    """
    Manages local ChromaDB vector database for semantic retrieval of historical support resolutions.
    """

    def __init__(self, persist_dir: Optional[str] = None):
        self.persist_dir = str(persist_dir or CHROMA_DIR)
        self.client = chromadb.PersistentClient(
            path=self.persist_dir,
            settings=Settings(anonymized_telemetry=False)
        )
        self.collection = self.client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"description": "Historical AppleSupport conversation resolutions for RAG grounding"}
        )

    def count(self) -> int:
        """Returns the number of indexed documents."""
        return self.collection.count()

    def index_resolutions(self, resolutions: List[Dict], force_reindex: bool = False) -> int:
        """
        Indexes a list of resolution dictionaries into ChromaDB.
        Each dictionary should contain 'id', 'customer_message', and 'support_reply'.
        """
        current_count = self.collection.count()
        if current_count > 0 and not force_reindex:
            logger.info(f"ChromaDB collection already contains {current_count} documents. Skipping indexing.")
            return current_count

        if force_reindex and current_count > 0:
            logger.info("Force reindex requested. Deleting existing collection entries...")
            self.client.delete_collection(COLLECTION_NAME)
            self.collection = self.client.get_or_create_collection(name=COLLECTION_NAME)

        ids = []
        documents = []
        metadatas = []

        for item in resolutions:
            item_id = str(item.get("id"))
            cust_text = item.get("customer_message", "").strip()
            supp_text = item.get("support_reply", "").strip()

            if not cust_text or not supp_text:
                continue

            ids.append(item_id)
            # The document to embed is the customer query and symptom description
            documents.append(cust_text)
            metadatas.append({
                "conversation_id": item_id,
                "support_reply": supp_text,
                "intent": item.get("intent", "general")
            })

        batch_size = 100
        total = len(ids)
        for i in range(0, total, batch_size):
            end = min(i + batch_size, total)
            self.collection.add(
                ids=ids[i:end],
                documents=documents[i:end],
                metadatas=metadatas[i:end]
            )

        logger.info(f"Successfully indexed {self.collection.count()} resolutions into ChromaDB at {self.persist_dir}.")
        return self.collection.count()

    def search_similar_resolutions(
        self,
        query: str,
        top_k: int = 3,
        filter_intent: Optional[str] = None
    ) -> List[Dict]:
        """
        Performs semantic vector search for historical resolutions matching an incoming query.
        Returns a list of matching dicts with 'similarity_score', 'matched_customer_query', and 'support_reply'.
        """
        if self.collection.count() == 0:
            logger.warning("ChromaDB collection is empty! Indexing historical resolutions now...")
            if HISTORICAL_RESOLUTIONS_PATH.exists():
                with open(HISTORICAL_RESOLUTIONS_PATH, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    self.index_resolutions(data)
            else:
                return []

        where_filter = {"intent": filter_intent} if filter_intent else None

        results = self.collection.query(
            query_texts=[query],
            n_results=min(top_k, self.collection.count()),
            where=where_filter,
            include=["documents", "metadatas", "distances"]
        )

        matches = []
        docs = results.get("documents", [[]])[0]
        metas = results.get("metadatas", [[]])[0]
        dists = results.get("distances", [[]])[0]

        for doc, meta, dist in zip(docs, metas, dists):
            # Chroma returns L2 or cosine distance; convert to similarity score in [0, 1]
            similarity = max(0.0, min(1.0, 1.0 - (dist / 2.0)))
            matches.append({
                "conversation_id": meta.get("conversation_id", ""),
                "matched_customer_query": doc,
                "support_reply": meta.get("support_reply", ""),
                "intent": meta.get("intent", "general"),
                "similarity_score": round(similarity, 4),
                "distance": round(dist, 4)
            })

        return matches


# Global instance
_store_instance = None


def get_vector_store() -> ChromaResolutionStore:
    """Singleton getter for ChromaResolutionStore."""
    global _store_instance
    if _store_instance is None:
        _store_instance = ChromaResolutionStore()
    return _store_instance


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    store = get_vector_store()
    if HISTORICAL_RESOLUTIONS_PATH.exists():
        with open(HISTORICAL_RESOLUTIONS_PATH, "r", encoding="utf-8") as f:
            resolutions = json.load(f)
        store.index_resolutions(resolutions)

    print("Store count:", store.count())
    query = "My battery is draining extremely fast after updating to iOS 11"
    print(f"\nQuerying: '{query}'")
    hits = store.search_similar_resolutions(query, top_k=2)
    for h in hits:
        print("Score:", h["similarity_score"])
        print("Matched Query:", h["matched_customer_query"])
        print("Support Reply:", h["support_reply"])
        print("-" * 40)
