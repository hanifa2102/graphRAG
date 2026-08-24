"""GraphRAG extraction, graph storage, community detection, and querying."""

import asyncio
from collections.abc import Sequence
from typing import Any

import networkx as nx
from graspologic.partition import hierarchical_leiden
from llama_index.core import Settings
from llama_index.core.async_utils import run_jobs
from llama_index.core.graph_stores import SimplePropertyGraphStore
from llama_index.core.graph_stores.types import (
    EntityNode,
    KG_NODES_KEY,
    KG_RELATIONS_KEY,
    Relation,
)
from llama_index.core.llms.llm import LLM
from llama_index.core.prompts import PromptTemplate
from llama_index.core.query_engine import CustomQueryEngine
from llama_index.core.schema import BaseNode, TransformComponent
from pydantic import Field, field_validator

from .graph_rag_schema import ExtractionResult, GraphRAGSchema


class GraphRAGExtractor(TransformComponent):
    """Extract ontology-constrained entities and relationships with descriptions."""

    llm: LLM = Field(default_factory=lambda: Settings.llm)
    extract_prompt: PromptTemplate = Field(
        default_factory=lambda: PromptTemplate(GraphRAGSchema.extraction_prompt())
    )
    num_workers: int = 4
    max_paths_per_chunk: int = 20

    @field_validator("extract_prompt", mode="before")
    @classmethod
    def coerce_to_prompt_template(cls, value: Any) -> PromptTemplate:
        return PromptTemplate(value) if isinstance(value, str) else value

    def __call__(
        self,
        nodes: Sequence[BaseNode],
        show_progress: bool = False,
        **kwargs: Any,
    ) -> list[BaseNode]:
        return asyncio.run(self.acall(nodes, show_progress=show_progress, **kwargs))

    async def _aextract(self, node: BaseNode) -> BaseNode:
        text = node.get_content(metadata_mode="llm")
        try:
            result = await self.llm.astructured_predict(
                ExtractionResult,
                self.extract_prompt,
                text=text,
                max_knowledge_triplets=self.max_paths_per_chunk,
            )
            entities = result.entities
            relationships = result.relationships
        except Exception as exc:
            node_label = node.metadata.get("title", node.node_id)
            raise RuntimeError(
                f"Extraction failed for {node_label!r}; graph construction was "
                "stopped to avoid saving an incomplete checkpoint."
            ) from exc

        existing_nodes = node.metadata.pop(KG_NODES_KEY, [])
        existing_relations = node.metadata.pop(KG_RELATIONS_KEY, [])
        base_metadata = node.metadata.copy()

        existing_nodes += [
            EntityNode(
                name=entity.name,
                label=entity.type,
                properties={
                    **base_metadata,
                    "entity_description": entity.description,
                },
            )
            for entity in entities
        ]

        entity_lookup = {entity.name: entity.type for entity in entities}
        for relationship in relationships:
            source_node = EntityNode(
                name=relationship.source,
                label=entity_lookup.get(relationship.source, "ENTITY"),
                properties=base_metadata,
            )
            target_node = EntityNode(
                name=relationship.target,
                label=entity_lookup.get(relationship.target, "ENTITY"),
                properties=base_metadata,
            )
            if relationship.source not in entity_lookup:
                existing_nodes.append(source_node)
            if relationship.target not in entity_lookup:
                existing_nodes.append(target_node)
            existing_relations.append(
                Relation(
                    label=relationship.relation,
                    source_id=source_node.id,
                    target_id=target_node.id,
                    properties={
                        **base_metadata,
                        "relationship_description": relationship.description,
                    },
                )
            )

        node.metadata[KG_NODES_KEY] = existing_nodes
        node.metadata[KG_RELATIONS_KEY] = existing_relations
        return node

    async def acall(
        self,
        nodes: Sequence[BaseNode],
        show_progress: bool = False,
        **kwargs: Any,
    ) -> list[BaseNode]:
        """Process nodes concurrently, bounded by ``num_workers``."""

        jobs = [self._aextract(node) for node in nodes]
        return await run_jobs(
            jobs,
            workers=self.num_workers,
            show_progress=show_progress,
            desc="Extracting triplets",
        )


class GraphRAGStore(SimplePropertyGraphStore):
    """Property graph store with Leiden communities and LLM summaries."""

    community_summaries: dict[int, str] = {}

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.community_summaries = {}

    def build_communities(
        self,
        summary_llm: LLM,
        max_cluster_size: int = 10,
    ) -> dict[int, str]:
        """Detect communities, generate their summaries, and return them."""

        print("Running community detection...")
        nx_graph = self.to_networkx()
        if not nx_graph.nodes:
            print("Graph is empty; no communities to detect.")
            self.community_summaries = {}
            return self.community_summaries

        print(
            f"Graph has {nx_graph.number_of_nodes()} nodes, "
            f"{nx_graph.number_of_edges()} edges"
        )
        clusters = hierarchical_leiden(
            nx_graph,
            max_cluster_size=max_cluster_size,
        )
        print(f"Found {len({cluster.cluster for cluster in clusters})} communities")

        community_info = self._collect_community_info(nx_graph, clusters)
        self.community_summaries = {}
        self._generate_summaries(community_info, summary_llm)
        print(f"Generated {len(self.community_summaries)} community summaries")
        return self.community_summaries

    def to_networkx(self) -> nx.Graph:
        """Convert the entity portion of the property graph to NetworkX."""

        nx_graph = nx.Graph()
        for node in self.get_entity_nodes():
            nx_graph.add_node(node.id)

        for relation in self.graph.relations.values():
            if relation.source_id in nx_graph and relation.target_id in nx_graph:
                nx_graph.add_edge(
                    relation.source_id,
                    relation.target_id,
                    relationship=relation.label,
                    description=relation.properties.get(
                        "relationship_description", ""
                    ),
                )
        return nx_graph

    # Compatibility with the method name used in the reference notebook.
    def _to_networkx(self) -> nx.Graph:
        return self.to_networkx()

    def get_entity_nodes(self) -> list[EntityNode]:
        """Return all typed entity nodes, excluding document/chunk nodes."""

        return [
            node
            for node in self.graph.nodes.values()
            if isinstance(node, EntityNode)
        ]

    def _collect_community_info(
        self,
        nx_graph: nx.Graph,
        clusters: Sequence[Any],
    ) -> dict[int, dict[str, list[Any]]]:
        community_mapping = {item.node: item.cluster for item in clusters}
        node_details = {
            node.id: {
                "name": node.name,
                "type": node.label,
                "description": node.properties.get("entity_description", ""),
            }
            for node in self.get_entity_nodes()
        }

        community_info: dict[int, dict[str, list[Any]]] = {}
        for item in clusters:
            community_id, node_id = item.cluster, item.node
            community_info.setdefault(
                community_id,
                {"entities": [], "relationships": []},
            )
            if node_id in node_details:
                community_info[community_id]["entities"].append(
                    node_details[node_id]
                )

            for neighbor in nx_graph.neighbors(node_id):
                if community_mapping.get(neighbor) != community_id:
                    continue
                edge = nx_graph.get_edge_data(node_id, neighbor) or {}
                relation = edge.get("relationship", "RELATED")
                description = edge.get("description", "")
                source_name = node_details.get(node_id, {}).get("name", node_id)
                target_name = node_details.get(neighbor, {}).get("name", neighbor)
                entry = f"{source_name} --[{relation}]--> {target_name}"
                if description:
                    entry += f" ({description})"
                community_info[community_id]["relationships"].append(entry)
        return community_info

    def _generate_summaries(
        self,
        community_info: dict[int, dict[str, list[Any]]],
        summary_llm: LLM,
    ) -> None:
        for community_id, data in community_info.items():
            if not data["relationships"] and not data["entities"]:
                continue
            entities_text = "\n".join(
                f"- {entity['name']} ({entity['type']}): "
                f"{entity['description']}"
                for entity in data["entities"]
                if entity.get("name")
            )
            relationships_text = "\n".join(
                sorted(set(data["relationships"]))
            )
            prompt = f"""You are analysing a cluster of entities from news articles about
AI copyright, governance, and intellectual property.

Entities in this cluster:
{entities_text}

Relationships:
{relationships_text}

Write a concise briefing (3-5 sentences) that:
1. Identifies the main organizations, people, legal cases, or topics in this cluster
2. Explains how they are connected and why, including legal or regulatory context
3. Highlights any disputes, lawsuits, policy positions, or tensions
4. Notes anything particularly relevant to AI copyright or governance

Briefing:"""
            response = summary_llm.complete(prompt)
            self.community_summaries[community_id] = response.text
            print(f"  Community {community_id}: {response.text[:100]}...")

    def get_community_summaries(self) -> dict[int, str]:
        return self.community_summaries


class GraphRAGQueryEngine(CustomQueryEngine):
    """Answer from individual communities, then synthesize the relevant answers."""

    graph_store: GraphRAGStore
    llm: LLM
    community_llm: LLM

    def custom_query(self, query_str: str) -> str:
        summaries = self.graph_store.get_community_summaries()
        if not summaries:
            return (
                "No community summaries found. Build communities or load a "
                "completed checkpoint first."
            )

        community_answers = [
            self._answer_from_community(summary, query_str)
            for summary in summaries.values()
        ]
        relevant_answers = [answer for answer in community_answers if answer.strip()]
        if not relevant_answers:
            return (
                "I don't have enough information in the knowledge graph to "
                "answer that question."
            )
        return self._aggregate(relevant_answers, query_str)

    def _answer_from_community(self, summary: str, query: str) -> str:
        prompt = (
            f"Community summary:\n{summary}\n\n"
            f"Question: {query}\n\n"
            "If this summary contains information relevant to the question, "
            "answer it. If not relevant, reply exactly: "
            "'No relevant information.'\n\nAnswer:"
        )
        text = self.community_llm.complete(prompt).text.strip()
        return "" if "no relevant information" in text.lower() else text

    def _aggregate(self, answers: list[str], query: str) -> str:
        combined = "\n\n---\n\n".join(answers)
        prompt = (
            "You have received answers from multiple knowledge graph communities "
            f"about this question:\n\nQuestion: {query}\n\n"
            f"Community answers:\n{combined}\n\n"
            "Synthesize these into a single, clear, well-structured final answer. "
            "Remove redundancy, keep all important details, and ensure the answer "
            "directly addresses the question.\n\nFinal Answer:"
        )
        return self.llm.complete(prompt).text
