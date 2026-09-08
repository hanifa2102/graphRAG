"""Application services for data loading, graph export, visualization, and queries."""

import json
import re
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Iterable

import pandas as pd
from llama_index.core import Document
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import TextNode
from pypdf import PdfReader
from pypdf.errors import PdfReadError

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
    def parse_page_selection(
        pages: str | Sequence[int] | None,
        total_pages: int,
    ) -> list[int]:
        """Return validated, 1-based PDF page numbers from a page selection."""

        if total_pages < 1:
            raise ValueError("The PDF does not contain any pages")
        if pages is None:
            return list(range(1, total_pages + 1))

        if isinstance(pages, str):
            page_spec = pages.strip()
            if page_spec.startswith("[") != page_spec.endswith("]"):
                raise ValueError(
                    "Page selection must have both opening and closing brackets"
                )
            if page_spec.startswith("["):
                page_spec = page_spec[1:-1]
            page_spec = re.sub(r"\s+", "", page_spec)
            page_spec = page_spec.translate(
                str.maketrans({"–": "-", "—": "-", "−": "-"})
            )
            if not page_spec:
                raise ValueError("Page selection cannot be empty")

            selected_pages: list[int] = []
            for item in page_spec.split(","):
                match = re.fullmatch(r"(\d+)(?:-(\d+))?", item)
                if match is None:
                    raise ValueError(
                        f"Invalid page selection item {item!r}; use values such "
                        "as '5' or '7-12'"
                    )
                start = int(match.group(1))
                end = int(match.group(2) or start)
                if start > end:
                    raise ValueError(
                        f"Invalid descending page range {item!r}"
                    )
                selected_pages.extend(range(start, end + 1))
        else:
            selected_pages = []
            for page_number in pages:
                if isinstance(page_number, bool) or not isinstance(page_number, int):
                    raise TypeError("Page numbers must be integers")
                selected_pages.append(page_number)
            if not selected_pages:
                raise ValueError("Page selection cannot be empty")

        invalid_pages = sorted(
            {page for page in selected_pages if not 1 <= page <= total_pages}
        )
        if invalid_pages:
            raise ValueError(
                f"Requested page(s) {invalid_pages} outside PDF page range "
                f"1-{total_pages}"
            )

        # Overlapping ranges should not cause a page to be processed twice.
        return list(dict.fromkeys(selected_pages))

    @classmethod
    def load_pdf_documents(
        cls,
        pdf_file: str | Path,
        pages: str | Sequence[int] | None = None,
        chunk_size: int = 1000,
        chunk_overlap: int = 100,
    ) -> list[TextNode]:
        """Extract selected text-PDF pages into sentence-aware token chunks."""

        pdf_path = Path(pdf_file)
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")
        if not pdf_path.is_file():
            raise ValueError(f"PDF path is not a file: {pdf_path}")
        if pdf_path.suffix.lower() != ".pdf":
            raise ValueError(f"Expected a .pdf file: {pdf_path}")
        if chunk_size < 1:
            raise ValueError("chunk_size must be at least 1")
        if not 0 <= chunk_overlap < chunk_size:
            raise ValueError(
                "chunk_overlap must be at least 0 and smaller than chunk_size"
            )

        try:
            reader = PdfReader(str(pdf_path))
        except (PdfReadError, OSError) as exc:
            raise ValueError(f"Could not read PDF: {pdf_path}") from exc
        if reader.is_encrypted:
            raise ValueError(f"Encrypted PDFs are not supported: {pdf_path}")

        selected_pages = cls.parse_page_selection(pages, len(reader.pages))
        pdf_title = pdf_path.stem
        if reader.metadata and reader.metadata.title:
            pdf_title = str(reader.metadata.title)

        page_documents: list[Document] = []
        empty_pages: list[int] = []
        for page_number in selected_pages:
            text = reader.pages[page_number - 1].extract_text() or ""
            text = cls._clean_pdf_text(text)
            if not text:
                empty_pages.append(page_number)
                continue
            page_documents.append(
                Document(
                    id_=f"{pdf_path.resolve()}#page={page_number}",
                    text=text,
                    metadata={
                        "title": pdf_title,
                        "source": str(pdf_path),
                        "file_name": pdf_path.name,
                        "page_number": page_number,
                    },
                )
            )

        if not page_documents:
            raise ValueError(
                "No extractable text found on the selected PDF pages. "
                "This loader supports text-based PDFs only."
            )

        splitter = SentenceSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            include_metadata=True,
            include_prev_next_rel=True,
        )
        chunks = splitter.get_nodes_from_documents(page_documents)
        chunks_per_page: dict[int, int] = {}
        for chunk_index, chunk in enumerate(chunks, start=1):
            page_number = int(chunk.metadata["page_number"])
            page_chunk_index = chunks_per_page.get(page_number, 0) + 1
            chunks_per_page[page_number] = page_chunk_index
            chunk.metadata["chunk_index"] = chunk_index
            chunk.metadata["page_chunk_index"] = page_chunk_index

        if empty_pages:
            print(f"Skipped PDF pages with no extractable text: {empty_pages}")
        print(
            f"Loaded {len(page_documents)} page(s) from '{pdf_path}' and "
            f"created {len(chunks)} chunks"
        )
        return chunks

    @staticmethod
    def _clean_pdf_text(text: str) -> str:
        """Normalize common text-PDF line-break artifacts."""

        text = text.replace("\r\n", "\n").replace("\r", "\n")
        text = re.sub(r"(?<=\w)-\n(?=\w)", "", text)
        text = re.sub(r"[ \t]+\n", "\n", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()

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
