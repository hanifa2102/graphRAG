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


# Community-detection weights for a Candidate Causal Graph.
# These weights affect clustering only; they do not change the extracted graph semantics.
RELATION_WEIGHTS: dict[str, float] = {
    "CAUSES": 4.0,
    "MAY_CAUSE": 3.5,
    "INDICATES": 3.0,
    "DETECTS": 2.5,
    "TRIGGERS": 2.5,
    "MITIGATES": 2.0,
    "CLEARS": 2.0,
    "RESPONDS_TO": 1.8,
    "CONTROLS": 1.5,
    "MONITORS": 1.5,
    "CONFIGURES": 1.4,
    "HAS_STATE": 1.0,
    "SENDS": 1.0,
    "RECEIVES": 1.0,
    "CONNECTED_TO": 0.8,
    "PART_OF": 0.7,
    "REQUIRES": 0.7,
    "HAS_STEP": 0.6,
    "ACTS_ON": 0.6,
    "PRECEDES": 0.5,
    "TESTS": 0.5,
    "PREVENTS": 1.5,
}


class GraphRAGExtractor(TransformComponent):
    """Extract ontology-constrained entities and relationships with descriptions."""

    llm: LLM = Field(default_factory=lambda: Settings.llm)
    extract_prompt: PromptTemplate = Field(
        default_factory=lambda: PromptTemplate(GraphRAGSchema.extraction_prompt())
    )
    num_workers: int = 4
    max_paths_per_chunk: int = 20
    rejected_relationships: list[dict[str, Any]] = Field(default_factory=list, exclude=True)

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
        rejected_relationships = []

        for relationship in relationships:
            # Do not silently create out-of-ontology ENTITY nodes when the model emits
            # an endpoint that it failed to declare in the entity list. Precision is
            # preferred over graph completion for the Candidate Causal Graph.
            missing = [
                endpoint
                for endpoint in (relationship.source, relationship.target)
                if endpoint not in entity_lookup
            ]
            if missing:
                rejected_relationships.append({
                    "source": relationship.source,
                    "target": relationship.target,
                    "label": relationship.relation,
                    "description": relationship.description,
                    "missing_endpoints": list(dict.fromkeys(missing)),
                    "reason": "endpoint_not_declared",
                    "chunk_id": node.node_id,
                    "provenance": node.metadata.get("provenance", []),
                    "excerpt": node.get_content(metadata_mode="none"),
                })
                continue

            source_node = EntityNode(
                name=relationship.source,
                label=entity_lookup[relationship.source],
                properties=base_metadata,
            )
            target_node = EntityNode(
                name=relationship.target,
                label=entity_lookup[relationship.target],
                properties=base_metadata,
            )
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

        if rejected_relationships:
            self.rejected_relationships.extend(rejected_relationships)
            node_label = node.metadata.get("title", node.node_id)
            print(
                f"Rejected {len(rejected_relationships)} relationship(s) in {node_label!r} "
                "because one or more endpoints were not declared as entities."
            )
            for item in rejected_relationships[:5]:
                print(f"  {item['source']} --[{item['label']}]--> {item['target']}; "
                      f"missing={item['missing_endpoints']}")

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
        self.community_members = {}
        self.community_primary = {}
        self.community_membership_origin = "unavailable"
        self.rejected_relationships = []

    def upsert_relations(self, relations: list[Relation]) -> None:
        """Merge source occurrences when a triple appears in multiple chunks/pages."""
        by_triple = {
            (item.source_id, item.label, item.target_id): item
            for item in self.graph.relations.values()
        }
        for relation in relations:
            key = (relation.source_id, relation.label, relation.target_id)
            previous = by_triple.get(key)
            occurrences = []
            for item in (previous, relation):
                if item is not None:
                    for provenance in item.properties.get("provenance", []):
                        if provenance not in occurrences:
                            occurrences.append(provenance)
            relation.properties["provenance"] = occurrences
            super().upsert_relations([relation])
            if previous is not None:
                previous.properties["provenance"] = occurrences
            by_triple[key] = previous if previous is not None else relation

    def build_communities(
        self,
        summary_llm: LLM,
        max_cluster_size: int = 18,
    ) -> dict[int, str]:
        """Detect final communities, generate their summaries, and return them."""

        print("Running community detection...")
        nx_graph = self.to_networkx()
        if not nx_graph.nodes:
            print("Graph is empty; no communities to detect.")
            self.community_summaries = {}
            self.community_members = {}
            self.community_primary = {}
            return self.community_summaries

        print(
            f"Graph has {nx_graph.number_of_nodes()} nodes, "
            f"{nx_graph.number_of_edges()} edges"
        )
        hierarchy = hierarchical_leiden(
            nx_graph,
            max_cluster_size=max_cluster_size,
            resolution=1.0,
            random_seed=42,
            weight_attribute="weight",
        )

        # hierarchical_leiden returns a state log across hierarchy levels.
        # For RAG summaries and primary visualization grouping, use final membership only.
        final_clusters = [item for item in hierarchy if item.is_final_cluster]
        print(
            f"Found {len({cluster.cluster for cluster in final_clusters})} final communities "
            f"({len({cluster.cluster for cluster in hierarchy})} community IDs across hierarchy)"
        )

        self._record_community_members(final_clusters, "built-final")
        community_info = self._collect_community_info(nx_graph, final_clusters)
        self.community_summaries = {}
        self._generate_summaries(community_info, summary_llm)
        print(f"Generated {len(self.community_summaries)} community summaries")
        return self.community_summaries

    def _record_community_members(self, clusters: Sequence[Any], origin: str) -> None:
        self.community_members = {}
        self.community_primary = {}
        for item in clusters:
            members = self.community_members.setdefault(item.cluster, [])
            if item.node not in members:
                members.append(item.node)
            # This also works when callers pass only final clusters.
            if item.is_final_cluster:
                self.community_primary[item.node] = item.cluster
        self.community_membership_origin = origin

    def ensure_community_members(self) -> None:
        """Recover final visualization groups for old checkpoints."""
        if self.community_members or not self.community_summaries:
            return
        graph = self.to_networkx()
        if graph.number_of_edges():
            hierarchy = hierarchical_leiden(
                graph,
                max_cluster_size=18,
                resolution=1.0,
                random_seed=42,
                weight_attribute="weight",
            )
            final_clusters = [item for item in hierarchy if item.is_final_cluster]
            self._record_community_members(final_clusters, "recovered-final")

    def to_networkx(self) -> nx.Graph:
        """Convert the entity portion of the property graph to weighted NetworkX."""

        nx_graph = nx.Graph()
        for node in self.get_entity_nodes():
            nx_graph.add_node(node.id)

        for relation in self.graph.relations.values():
            if relation.source_id not in nx_graph or relation.target_id not in nx_graph:
                continue

            weight = RELATION_WEIGHTS.get(relation.label, 1.0)
            if nx_graph.has_edge(relation.source_id, relation.target_id):
                # Multiple semantic relations between the same pair should strengthen
                # their connection rather than overwrite the first relation's weight.
                edge = nx_graph[relation.source_id][relation.target_id]
                edge["weight"] = float(edge.get("weight", 0.0)) + weight
                labels = edge.setdefault("relationships", [])
                if relation.label not in labels:
                    labels.append(relation.label)
                descriptions = edge.setdefault("descriptions", [])
                description = relation.properties.get("relationship_description", "")
                if description and description not in descriptions:
                    descriptions.append(description)
            else:
                description = relation.properties.get("relationship_description", "")
                nx_graph.add_edge(
                    relation.source_id,
                    relation.target_id,
                    weight=weight,
                    relationship=relation.label,
                    relationships=[relation.label],
                    description=description,
                    descriptions=[description] if description else [],
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
                relations = edge.get("relationships") or [
                    edge.get("relationship", "RELATED")
                ]
                descriptions = edge.get("descriptions") or [
                    edge.get("description", "")
                ]
                source_name = node_details.get(node_id, {}).get("name", node_id)
                target_name = node_details.get(neighbor, {}).get("name", neighbor)
                for index, relation in enumerate(relations):
                    description = descriptions[index] if index < len(descriptions) else ""
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
            prompt = GraphRAGSchema.render_prompt(
                "community_summary",
                entities_text=entities_text,
                relationships_text=relationships_text,
            )
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
        prompt = GraphRAGSchema.render_prompt(
            "community_answer", summary=summary, query=query,
        )
        text = self.community_llm.complete(prompt).text.strip()
        return "" if "no relevant information" in text.lower() else text

    def _aggregate(self, answers: list[str], query: str) -> str:
        combined = "\n\n---\n\n".join(answers)
        prompt = GraphRAGSchema.render_prompt(
            "aggregation", combined=combined, query=query,
        )
        return self.llm.complete(prompt).text
