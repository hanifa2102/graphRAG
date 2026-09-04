"""GraphRAG lifecycle management: build, persist, load, inspect, and query."""

import asyncio
import copy
import os
import pickle
from collections import defaultdict
from pathlib import Path
from typing import Any, Literal

from dotenv import load_dotenv
from llama_index.core import Document, PropertyGraphIndex, Settings
from llama_index.core.graph_stores.types import (
    KG_NODES_KEY,
    KG_RELATIONS_KEY,
)
from llama_index.core.llms.llm import LLM
from llama_index.llms.openai import OpenAI

from .graph_rag_engine import GraphRAGExtractor, GraphRAGStore
from .graph_rag_schema import GraphRAGSchema
from .graph_rag_services import GraphRAGService

LLMProvider = Literal["openai", "qwen"]


class GraphRAGManager:
    """Class 4: build, persist, reload, inspect, visualize, and query the graph."""

    CHECKPOINT_VERSION = 1

    def __init__(
        self,
        provider: LLMProvider = "openai",
        extraction_model: str = "gpt-4o-mini",
        query_model: str = "gpt-4o",
        qwen_base_url: str = "http://192.168.10.45:4000/v1",
        qwen_context_window: int = 32768,
        max_paths_per_chunk: int = 20,
        num_workers: int = 4,
        max_cluster_size: int = 10,
        request_timeout: float = 180.0,
        request_max_retries: int = 5,
    ) -> None:
        load_dotenv()
        if provider not in ("openai", "qwen"):
            raise ValueError("provider must be either 'openai' or 'qwen'")

        self.provider = provider
        self.qwen_base_url = qwen_base_url
        self.qwen_context_window = qwen_context_window
        self.llm_request_options = {
            "timeout": request_timeout,
            "max_retries": request_max_retries,
        }
        self.extraction_llm = self._new_llm(extraction_model)
        self.query_llm = self._new_llm(query_model)
        Settings.llm = self.extraction_llm

        self.max_paths_per_chunk = max_paths_per_chunk
        self.num_workers = num_workers
        self.max_cluster_size = max_cluster_size
        self.documents: list[Document] = []
        self.graph_store: GraphRAGStore | None = None
        self.index: PropertyGraphIndex | None = None
        self.extractor = self._new_extractor()

    def _new_llm(self, model: str) -> LLM:
        if self.provider == "openai":
            return OpenAI(
                model=model,
                temperature=0,
                **self.llm_request_options,
            )

        api_key = os.getenv("LITELLM_API_KEY")
        if not api_key:
            raise ValueError(
                "LITELLM_API_KEY is required when provider='qwen'"
            )
        try:
            from llama_index.llms.openai_like import OpenAILike
        except ImportError as exc:
            raise ImportError(
                "Qwen support requires llama-index-llms-openai-like. "
                "Install the dependencies from requirements.txt."
            ) from exc

        return OpenAILike(
            model=model,
            api_base=self.qwen_base_url,
            api_key=api_key,
            temperature=0,
            context_window=self.qwen_context_window,
            is_chat_model=True,
            is_function_calling_model=False,
            should_use_structured_outputs=True,
            additional_kwargs={
                "extra_body": {
                    "chat_template_kwargs": {"enable_thinking": False}
                }
            },
            **self.llm_request_options,
        )

    def _new_extractor(self) -> GraphRAGExtractor:
        return GraphRAGExtractor(
            llm=self.extraction_llm,
            extract_prompt=GraphRAGSchema.extraction_prompt(),
            max_paths_per_chunk=self.max_paths_per_chunk,
            num_workers=self.num_workers,
        )

    def load_documents(
        self,
        csv_file: str | Path,
        max_articles: int | None = None,
    ) -> list[Document]:
        self.documents = GraphRAGService.load_documents(csv_file, max_articles)
        return self.documents

    def build_knowledge_graph(
        self,
        documents: list[Document] | None = None,
        build_communities: bool = True,
        show_progress: bool = True,
    ) -> GraphRAGStore:
        """Run extraction, construct the graph, and optionally summarize communities."""

        if documents is not None:
            self.documents = documents
        if not self.documents:
            raise ValueError("No documents loaded. Call load_documents() first.")

        self.extractor = self._new_extractor()
        self.graph_store = GraphRAGStore()
        print("Building knowledge graph; LLM extraction may take several minutes...")
        self.index = PropertyGraphIndex(
            nodes=self.documents,
            kg_extractors=[self.extractor],
            property_graph_store=self.graph_store,
            embed_kg_nodes=False,
            show_progress=show_progress,
        )
        if build_communities:
            self.graph_store.build_communities(
                summary_llm=self.extraction_llm,
                max_cluster_size=self.max_cluster_size,
            )
        return self.graph_store

    def save_knowledge_graph(self, checkpoint_file: str | Path) -> Path:
        """Pickle graph state and summaries, excluding live LLM clients."""

        graph_store = self._require_graph_store()
        checkpoint_path = Path(checkpoint_file)
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": self.CHECKPOINT_VERSION,
            "graph": graph_store.graph,
            "community_summaries": dict(
                graph_store.get_community_summaries()
            ),
        }
        temporary_path = checkpoint_path.with_suffix(
            checkpoint_path.suffix + ".tmp"
        )
        with temporary_path.open("wb") as output:
            pickle.dump(payload, output, protocol=pickle.HIGHEST_PROTOCOL)
        temporary_path.replace(checkpoint_path)
        print(f"Knowledge graph saved to '{checkpoint_path}'")
        return checkpoint_path

    def load_knowledge_graph(self, checkpoint_file: str | Path) -> GraphRAGStore:
        """Load a trusted checkpoint created by ``save_knowledge_graph``."""

        checkpoint_path = Path(checkpoint_file)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        with checkpoint_path.open("rb") as source:
            payload: dict[str, Any] = pickle.load(source)

        if payload.get("version") != self.CHECKPOINT_VERSION:
            raise ValueError(
                "Unsupported checkpoint version: "
                f"{payload.get('version')!r}"
            )
        if "graph" not in payload or "community_summaries" not in payload:
            raise ValueError("Checkpoint is missing required graph state")

        graph_store = GraphRAGStore()
        graph_store.graph = payload["graph"]
        graph_store.community_summaries = dict(
            payload["community_summaries"]
        )
        self.graph_store = graph_store
        self.index = None
        print(f"Knowledge graph loaded from '{checkpoint_path}'")
        print(
            f"Entities: {self.entity_count} | Relations: {self.relation_count} | "
            f"Communities: {len(graph_store.get_community_summaries())}"
        )
        return graph_store

    def build_and_store(
        self,
        csv_file: str | Path,
        checkpoint_file: str | Path,
        max_articles: int | None = None,
        show_progress: bool = True,
    ) -> GraphRAGStore:
        """Convenience method for the complete expensive build workflow."""

        self.load_documents(csv_file, max_articles)
        graph_store = self.build_knowledge_graph(show_progress=show_progress)
        self.save_knowledge_graph(checkpoint_file)
        return graph_store

    async def atest_extraction(self, document_index: int = 3) -> dict[str, Any]:
        """Extract and print one article without mutating the stored document."""

        if not self.documents:
            raise ValueError("No documents loaded. Call load_documents() first.")
        if not 0 <= document_index < len(self.documents):
            raise IndexError(
                f"document_index must be between 0 and {len(self.documents) - 1}"
            )

        sample = self.documents[document_index]
        print("=== Title ===")
        print(sample.metadata.get("title", "untitled"))
        print("\n=== Raw text ===")
        print(sample.text)

        extracted = await self.extractor._aextract(copy.deepcopy(sample))
        entities = extracted.metadata.get(KG_NODES_KEY, [])
        relationships = extracted.metadata.get(KG_RELATIONS_KEY, [])

        print("\n=== Extracted entities ===")
        for entity in entities:
            print(f"  [{entity.label}] {entity.name}")
            print(f"    {entity.properties.get('entity_description', '')}")

        print("\n=== Extracted relationships ===")
        names_by_id = {entity.id: entity.name for entity in entities}
        for relationship in relationships:
            source = names_by_id.get(
                relationship.source_id, relationship.source_id
            )
            target = names_by_id.get(
                relationship.target_id, relationship.target_id
            )
            print(f"  {source} --[{relationship.label}]--> {target}")

        return {
            "document": sample,
            "entities": entities,
            "relationships": relationships,
        }

    def test_extraction(self, document_index: int = 3) -> dict[str, Any]:
        """Synchronous wrapper for scripts; use ``atest_extraction`` in Jupyter."""

        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(self.atest_extraction(document_index))
        raise RuntimeError(
            "An event loop is already running. In Jupyter use: "
            "await manager.atest_extraction(document_index)"
        )

    def print_unique_entities(self) -> dict[str, list[str]]:
        """Print and return unique entity names grouped by ontology type."""

        graph_store = self._require_graph_store()
        grouped: defaultdict[str, set[str]] = defaultdict(set)
        for node in graph_store.get_entity_nodes():
            grouped[node.label].add(node.name)

        result = {
            entity_type: sorted(names)
            for entity_type, names in sorted(grouped.items())
        }
        for entity_type, names in result.items():
            print(f"\n{entity_type} ({len(names)})")
            for name in names:
                print(f"  {name}")
        return result

    def inspect_entity(self, entity_name: str) -> dict[str, Any] | None:
        """Print an entity, its source article, and all incident relations."""

        graph_store = self._require_graph_store()
        entity = next(
            (
                node
                for node in graph_store.get_entity_nodes()
                if node.name == entity_name
            ),
            None,
        )
        if entity is None:
            print(f"Entity not found: {entity_name}")
            return None

        print(f"Node: {entity.name!r}  label={entity.label!r}")
        print(f"Source: {entity.properties.get('source', 'N/A')}")
        print(f"Title: {entity.properties.get('title', 'N/A')}")

        title = entity.properties.get("title")
        article = next(
            (
                document
                for document in self.documents
                if document.metadata.get("title") == title
            ),
            None,
        )
        if article:
            print("\n=== Article text ===")
            print(article.text)

        relations = [
            relation
            for relation in graph_store.graph.relations.values()
            if relation.source_id == entity.id or relation.target_id == entity.id
        ]
        names_by_id = {
            node.id: node.name
            for node in graph_store.get_entity_nodes()
        }
        print(f"\nRelations ({len(relations)}):")
        for relation in relations:
            source = names_by_id.get(relation.source_id, relation.source_id)
            target = names_by_id.get(relation.target_id, relation.target_id)
            print(f"  {source} --[{relation.label}]--> {target}")
        return {"entity": entity, "article": article, "relations": relations}

    def visualize(
        self,
        graph_data_file: str | Path,
        template_file: str | Path,
        output_file: str | Path,
    ) -> Path:
        return GraphRAGService.visualize_graph(
            graph_store=self._require_graph_store(),
            graph_data_file=graph_data_file,
            template_file=template_file,
            output_file=output_file,
        )

    def query(self, questions: str | list[str]) -> dict[str, str]:
        return GraphRAGService.query_system(
            graph_store=self._require_graph_store(),
            community_llm=self.extraction_llm,
            query_llm=self.query_llm,
            questions=questions,
        )

    def _require_graph_store(self) -> GraphRAGStore:
        if self.graph_store is None:
            raise ValueError("No graph is available. Build or load one first.")
        return self.graph_store

    @property
    def entity_count(self) -> int:
        return len(self._require_graph_store().get_entity_nodes())

    @property
    def relation_count(self) -> int:
        return len(self._require_graph_store().graph.relations)
