"""Application services for data loading, graph export, visualization, and queries."""

import json
from pathlib import Path
from typing import TYPE_CHECKING, Iterable

import pandas as pd
from llama_index.core import Document
from .graph_rag_engine import GraphRAGQueryEngine, GraphRAGStore

if TYPE_CHECKING:
    from llama_index.core.llms.llm import LLM


class GraphRAGService:
    """Class 3: stateless application services shared by the CLI and notebook."""

    @staticmethod
    def load_documents(
        csv_file: str | Path,
        max_articles: int | None = None,
    ) -> list[Document]:
        """Read article CSV data and return LlamaIndex ``Document`` objects."""

        csv_path = Path(csv_file)
        if not csv_path.exists():
            raise FileNotFoundError(f"Dataset not found: {csv_path}")

        dataframe = pd.read_csv(csv_path)
        if "full_text" not in dataframe.columns:
            raise ValueError("Dataset must contain a 'full_text' column")
        if max_articles is not None:
            if max_articles < 1:
                raise ValueError("max_articles must be at least 1")
            dataframe = dataframe.head(max_articles)

        documents = [
            Document(
                text=str(row["full_text"]),
                metadata={
                    "title": str(row.get("title", "")),
                    "source": str(row.get("source", "")),
                    "date": str(row.get("date", "")),
                },
            )
            for _, row in dataframe.iterrows()
        ]
        print(f"Created {len(documents)} LlamaIndex documents")
        return documents

    @staticmethod
    def export_graph_data(
        graph_store: GraphRAGStore,
        output_file: str | Path = "graph_data.json",
    ) -> dict:
        """Export graph nodes, edges, and community count as D3-compatible JSON."""

        output_path = Path(output_file)
        nx_graph = graph_store.to_networkx()
        node_metadata = {
            node.id: {
                "id": node.id,
                "label": node.name,
                "type": node.label or "OTHER",
                "description": node.properties.get("entity_description", ""),
            }
            for node in graph_store.get_entity_nodes()
        }
        nodes_data = [
            node_metadata[node_id]
            for node_id in nx_graph.nodes()
            if node_id in node_metadata
        ]
        links_data = [
            {
                "source": source,
                "target": target,
                "label": data.get("relationship", ""),
                "description": data.get("description", ""),
            }
            for source, target, data in nx_graph.edges(data=True)
        ]
        graph_data = {
            "nodes": nodes_data,
            "links": links_data,
            "communities": len(graph_store.get_community_summaries()),
        }

        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(graph_data, indent=2),
            encoding="utf-8",
        )
        print(f"Graph data exported to '{output_path}'")
        print(
            f"Nodes: {len(nodes_data)} | Edges: {len(links_data)} | "
            f"Communities: {graph_data['communities']}"
        )
        return graph_data

    @classmethod
    def visualize_graph(
        cls,
        graph_store: GraphRAGStore | None = None,
        graph_data_file: str | Path = "graph_data.json",
        template_file: str | Path = "graph_template.html",
        output_file: str | Path = "ai_copyright_graph.html",
    ) -> Path:
        """Render graph data into the repository's interactive HTML template."""

        graph_data_path = Path(graph_data_file)
        if graph_store is not None:
            graph_data = cls.export_graph_data(graph_store, graph_data_path)
        else:
            if not graph_data_path.exists():
                raise FileNotFoundError(
                    f"Graph data not found: {graph_data_path}. Export it first."
                )
            graph_data = json.loads(graph_data_path.read_text(encoding="utf-8"))

        template_path = Path(template_file)
        if not template_path.exists():
            raise FileNotFoundError(f"Graph template not found: {template_path}")
        template = template_path.read_text(encoding="utf-8")
        placeholder = "GRAPH_DATA_PLACEHOLDER"
        if placeholder not in template:
            raise ValueError(
                f"Template '{template_path}' does not contain {placeholder}"
            )

        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            template.replace(placeholder, json.dumps(graph_data)),
            encoding="utf-8",
        )
        print(f"Visualization saved to '{output_path}'")
        return output_path

    @staticmethod
    def query_system(
        graph_store: GraphRAGStore,
        community_llm: "LLM",
        query_llm: "LLM",
        questions: str | Iterable[str],
    ) -> dict[str, str]:
        """Run one or more questions against the community-summary query engine."""

        question_list = [questions] if isinstance(questions, str) else list(questions)
        query_engine = GraphRAGQueryEngine(
            graph_store=graph_store,
            community_llm=community_llm,
            llm=query_llm,
        )
        answers = {}
        for question in question_list:
            print(f"Query: {question}")
            print("=" * 70)
            answer = query_engine.custom_query(question)
            answers[question] = answer
            print(answer)
        return answers
