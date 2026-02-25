"""
Production-hardened FAISS vector store retrieval service.

Thread-safe with cached embedding dimensions, input validation, and proper logging.

Requires: pip install faiss-cpu langchain-community langchain-core langchain-text-splitters
    (or: pip install agentsoul[retrieval])
"""

import logging
import os
import shutil
import threading
from typing import Dict, List, Optional, Tuple
from uuid import uuid4

try:
    import faiss
    from langchain_community.docstore.in_memory import InMemoryDocstore
    from langchain_community.vectorstores import FAISS
    from langchain_core.documents import Document
    from langchain_text_splitters import CharacterTextSplitter
except ImportError:
    raise ImportError(
        "FAISS and LangChain packages are required for FAISSService. "
        "Install them with: pip install faiss-cpu langchain-community langchain-core langchain-text-splitters  "
        "or: pip install agentsoul[retrieval]"
    )

logger = logging.getLogger(__name__)


class FAISSService:
    """FAISS vector store retrieval service.

    Thread-safe similarity search over a FAISS index backed by LangChain.

    Args:
        embedding: A LangChain-compatible embedding model (must have ``embed_query``).
        index_path: Directory path for the FAISS index on disk.
        md_data_paths: Optional list of markdown file paths to build an index from.
        documents: Optional pre-built list of LangChain Documents (overrides md_data_paths).
        replace_existing: If True, rebuild the index even if one exists on disk.
    """

    def __init__(
        self,
        embedding,
        index_path: str,
        md_data_paths: Optional[List[str]] = None,
        documents: Optional[List[Document]] = None,
        replace_existing: bool = False,
    ):
        self.embedding = embedding
        self.index_path = index_path
        self._lock = threading.Lock()
        self._embedding_dim: Optional[int] = None

        self.vector_store, _ = self._build_or_load_index(
            md_paths=md_data_paths or [],
            path=index_path,
            documents=documents,
            replace_existing=replace_existing,
        )

    def _get_embedding_dimension(self) -> int:
        """Return the embedding dimension, caching the result."""
        if self._embedding_dim is None:
            self._embedding_dim = len(self.embedding.embed_query("dimension probe"))
        return self._embedding_dim

    def load_and_split_markdowns(self, docs: List[str]) -> List[Document]:
        """Load markdown files and split on ``---`` separators.

        Args:
            docs: List of file paths to markdown documents.

        Returns:
            List of LangChain Document objects with metadata.
        """
        logger.info("Loading %d markdown files for indexing...", len(docs))
        documents: List[Document] = []

        for path in docs:
            if not os.path.isfile(path):
                logger.warning("Markdown file not found, skipping: %s", path)
                continue

            with open(path, "r", encoding="utf-8") as f:
                markdown_document = f.read()

            # chunk_size=1 forces splits only on separator, not on length
            text_splitter = CharacterTextSplitter(
                separator="\n---\n",
                chunk_size=1,
                chunk_overlap=0,
            )
            splits = text_splitter.split_text(markdown_document)

            documents.extend([
                Document(
                    page_content=split,
                    metadata={"chunk_order": i, "source": path},
                )
                for i, split in enumerate(splits)
            ])

        uuids = [str(uuid4()) for _ in range(len(documents))]
        for i, uid in enumerate(uuids):
            documents[i].metadata["doc_id"] = uid

        logger.info("Split into %d document chunks.", len(documents))
        return documents

    def _build_or_load_index(
        self,
        md_paths: List[str],
        path: str,
        documents: Optional[List[Document]] = None,
        replace_existing: bool = False,
    ) -> Tuple:
        """Build a new FAISS index or load an existing one from disk."""
        if not os.path.exists(path) or replace_existing:
            if documents is None:
                documents = self.load_and_split_markdowns(md_paths)

            if not documents:
                logger.warning("No documents to index. Creating empty vector store.")
                dim = self._get_embedding_dimension()
                index = faiss.IndexFlatL2(dim)
                vector_store = FAISS(
                    embedding_function=self.embedding,
                    index=index,
                    docstore=InMemoryDocstore(),
                    index_to_docstore_id={},
                )
                return vector_store, []

            if replace_existing and os.path.exists(path):
                logger.info("Replacing existing FAISS index at %s", path)
                shutil.rmtree(path, ignore_errors=True)

            uuids = [doc.metadata["doc_id"] for doc in documents]
            dim = self._get_embedding_dimension()
            index = faiss.IndexFlatL2(dim)
            vector_store = FAISS(
                embedding_function=self.embedding,
                index=index,
                docstore=InMemoryDocstore(),
                index_to_docstore_id={},
            )
            vector_store.add_documents(documents=documents, ids=uuids)
            os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
            vector_store.save_local(path)
            logger.info("Built and saved FAISS index with %d docs at %s", len(documents), path)
        else:
            vector_store = FAISS.load_local(
                path, self.embedding, allow_dangerous_deserialization=True
            )
            docstore_ids = list(vector_store.index_to_docstore_id.values())
            documents = [vector_store.docstore.search(i) for i in docstore_ids]
            logger.info("Loaded FAISS index with %d docs from %s", len(documents), path)

        return vector_store, documents

    def search_with_ranks(
        self,
        queries: list,
        top_k: int = 20,
        rel_filter: Optional[list] = None,
    ) -> Tuple[List[Dict], List]:
        """Search the FAISS vector store and return ranked results.

        Args:
            queries: List of query strings.
            top_k: Number of top results to return.
            rel_filter: Ignored for FAISS (accepted for API compatibility
                with graph-based callers).

        Returns:
            Tuple of (results_list, relations_list).
            relations_list is always empty for FAISS.
        """
        if rel_filter:
            logger.debug(
                "rel_filter=%s ignored by FAISS search (no graph relations available)",
                rel_filter,
            )

        if not queries or not isinstance(queries, list):
            logger.warning("Invalid queries input: %s", type(queries))
            return [], []

        out_data: List[Dict] = []
        rag_set: set = set()

        with self._lock:
            for j, query in enumerate(queries):
                if not isinstance(query, str) or not query.strip():
                    continue
                try:
                    hits = self.vector_store.similarity_search_with_score(query, k=top_k)
                except Exception as e:
                    logger.warning(
                        "similarity_search_with_score failed for query %d, falling back: %s", j, e
                    )
                    try:
                        docs = self.vector_store.similarity_search(query, k=top_k)
                        hits = [(d, 0.0) for d in docs]
                    except Exception as e2:
                        logger.error("Fallback search also failed for query %d: %s", j, e2)
                        continue

                for d, dist in hits:
                    doc_id = d.metadata.get("doc_id")
                    if doc_id is None:
                        doc_id = f"{d.metadata.get('source', '?')}::{d.metadata.get('chunk_order', '?')}"
                    if doc_id not in rag_set:
                        rag_set.add(doc_id)
                        score = dist.item() if hasattr(dist, "item") else float(dist)
                        out_data.append({
                            "doc_id": doc_id,
                            "text": d.page_content,
                            "metadata": d.metadata,
                            "score": score,
                        })

        out_data = sorted(out_data, key=lambda x: x["score"])
        return out_data[:top_k], []
