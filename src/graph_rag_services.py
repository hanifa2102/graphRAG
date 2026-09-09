"""Application services for data loading, graph export, visualization, and queries."""

import json
import re
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Iterable

from llama_index.core import Document
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import TextNode
from pypdf import PdfReader
from pypdf.errors import PdfReadError

from .graph_rag_engine import GraphRAGQueryEngine, GraphRAGStore
from .graph_rag_schema import GraphRAGSchema

OUTPUT_DIR = Path(__file__).resolve().parents[1] / "output"

def output_path(file: str | Path | None, suffix: str) -> Path:
    """Keep a single artifact of each kind per active profile, regardless of CWD."""
    prefix = GraphRAGSchema.ACTIVE_ONTOLOGY.lower().replace(" ", "_")
    expected = OUTPUT_DIR / f"{prefix}_{suffix}"
    if file is not None and Path(file).resolve() != expected.resolve():
        raise ValueError(f"Use the profile output path: {expected}")
    return expected


if TYPE_CHECKING:
    from llama_index.core.llms.llm import LLM


class GraphRAGService:
    """Class 3: stateless application services shared by the CLI and notebook."""

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
    def list_pdf_sections(cls, pdf_file: str | Path) -> list[dict]:
        """List bookmark sections with stable outline IDs and physical page ranges.

        IDs such as ``1.2`` describe bookmark order, not printed section numbers.
        Parent ranges include descendants. Boundaries are page-based, so sections
        starting on the same page share that page. PDFs without bookmarks return [].
        """
        return cls._pdf_sections(cls._open_pdf(pdf_file))

    @staticmethod
    def _open_pdf(pdf_file: str | Path) -> PdfReader:
        pdf_path = Path(pdf_file)
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")
        if not pdf_path.is_file():
            raise ValueError(f"PDF path is not a file: {pdf_path}")
        if pdf_path.suffix.lower() != ".pdf":
            raise ValueError(f"Expected a .pdf file: {pdf_path}")
        try:
            reader = PdfReader(str(pdf_path))
        except (PdfReadError, OSError) as exc:
            raise ValueError(f"Could not read PDF: {pdf_path}") from exc
        if reader.is_encrypted:
            raise ValueError(f"Encrypted PDFs are not supported: {pdf_path}")
        return reader

    @staticmethod
    def _pdf_sections(reader: PdfReader) -> list[dict]:
        sections: list[dict] = []

        def visit(items, prefix="", level=1):
            index = 0
            section_id = prefix
            for item in items:
                if isinstance(item, list):
                    visit(item, section_id, level + 1)
                    continue
                index += 1
                section_id = f"{prefix}.{index}" if prefix else str(index)
                page = reader.get_destination_page_number(item)
                if page is None or not 0 <= page < len(reader.pages):
                    raise ValueError(f"PDF bookmark {item.title!r} has no valid page")
                sections.append(dict(
                    id=section_id, title=str(item.title), level=level,
                    start_page=page + 1, end_page=len(reader.pages),
                ))

        visit(reader.outline)
        if any(a["start_page"] > b["start_page"] for a, b in zip(sections, sections[1:])):
            raise ValueError("PDF bookmarks are not in page order; use pages instead")
        for index, section in enumerate(sections):
            for following in sections[index + 1:]:
                if following["level"] <= section["level"]:
                    section["end_page"] = max(section["start_page"], following["start_page"] - 1)
                    break
        return sections

    @staticmethod
    def _select_pdf_sections(selection: str | Sequence[str], available: list[dict]) -> list[dict]:
        if not available:
            raise ValueError("PDF has no bookmarks for section selection; use pages instead")
        selectors = [selection] if isinstance(selection, str) else list(selection)
        if not selectors:
            raise ValueError("Section selection cannot be empty")
        selected = []
        for selector in selectors:
            if not isinstance(selector, str):
                raise TypeError("Sections must be outline ID or title strings")
            selector = selector.strip()
            matches = [s for s in available if s["id"] == selector]
            if not matches:
                matches = [s for s in available if s["title"].casefold() == selector.casefold()]
            if not matches:
                raise ValueError(f"Unknown PDF section {selector!r}; use list_pdf_sections()")
            if len(matches) > 1:
                raise ValueError(f"Ambiguous PDF section {selector!r}; select its outline ID")
            if matches[0] not in selected:
                selected.append(matches[0])
        return selected

    @classmethod
    def _pdf_section_spans(
        cls,
        reader: PdfReader,
        section_ranges: Sequence[tuple[str, str]] | tuple[str, str],
        page_texts: dict[int, str],
    ) -> list[tuple[int, int, int, str, str]]:
        """Resolve heading pairs to page-local character spans before chunking."""
        if isinstance(section_ranges, tuple) and len(section_ranges) == 2 and all(
            isinstance(value, str) for value in section_ranges
        ):
            section_ranges = [section_ranges]
        if not section_ranges:
            raise ValueError("Section ranges cannot be empty")
        bookmarks = cls._pdf_sections(reader)

        def locate(title):
            matches = [s for s in bookmarks if s["title"].casefold() == title.casefold()]
            if len(matches) > 1:
                raise ValueError(f"Ambiguous heading {title!r} in PDF bookmarks")
            candidates = ([matches[0]["start_page"]] if matches
                          else range(1, len(reader.pages) + 1))
            pattern = re.compile(
                r"^[ \t]*" + r"[ \t]+".join(re.escape(word) for word in title.split())
                + r"[ \t]*$", re.IGNORECASE | re.MULTILINE,
            )
            locations = []
            for page in candidates:
                if page not in page_texts:
                    page_texts[page] = cls._clean_pdf_text(reader.pages[page - 1].extract_text() or "")
                locations.extend((page, match.start()) for match in pattern.finditer(page_texts[page]))
            if len(locations) != 1:
                raise ValueError(f"Expected one standalone heading {title!r}; found {len(locations)}")
            return locations[0]

        spans = []
        for pair in section_ranges:
            if (not isinstance(pair, (tuple, list)) or len(pair) != 2
                    or any(not isinstance(title, str) or not title.strip() for title in pair)):
                raise ValueError("Each section range must be a (title, next_title) pair of nonempty strings")
            title, next_title = (value.strip() for value in pair)
            start, stop = locate(title), locate(next_title)
            if stop <= start:
                raise ValueError(f"End heading {next_title!r} must follow {title!r}")
            for page in range(start[0], stop[0] + 1):
                if page not in page_texts:
                    page_texts[page] = cls._clean_pdf_text(reader.pages[page - 1].extract_text() or "")
                left = start[1] if page == start[0] else 0
                right = stop[1] if page == stop[0] else len(page_texts[page])
                if page_texts[page][left:right].strip():
                    span = (page, left, right, title, next_title)
                    if span not in spans:
                        spans.append(span)
        return spans

    @classmethod
    def load_pdf_documents(
        cls,
        pdf_file: str | Path,
        pages: str | Sequence[int] | None = None,
        chunk_size: int = 1000,
        chunk_overlap: int = 100,
        *,
        sections: str | Sequence[str] | None = None,
        section_ranges: Sequence[tuple[str, str]] | tuple[str, str] | None = None,
    ) -> list[TextNode]:
        """Chunk pages or bookmark sections, retaining physical page provenance.

        ``sections`` accepts one outline ID/title or a list of IDs/titles from
        ``list_pdf_sections``. It includes descendants and cannot accompany pages.
        Section selection loads whole pages, not exact heading-to-heading text.
        ``section_ranges=[(title, next_title)]`` instead includes the start heading
        and stops before the end heading, including text on its page. Headings
        must occupy standalone lines. Bookmarks narrow the search when available.
        Page headers/footers inside the range are retained. Selection modes are
        mutually exclusive. Section text is joined across pages before token
        splitting; each chunk has one provenance entry per contributing page.
        Page-only loading continues to split each page separately.
        """

        pdf_path = Path(pdf_file)
        if sum(value is not None for value in (pages, sections, section_ranges)) > 1:
            raise ValueError("Select either pages or sections or section_ranges, not multiple modes")
        if chunk_size < 1:
            raise ValueError("chunk_size must be at least 1")
        if not 0 <= chunk_overlap < chunk_size:
            raise ValueError(
                "chunk_overlap must be at least 0 and smaller than chunk_size"
            )

        reader = cls._open_pdf(pdf_path)
        selected_sections = []
        if sections is not None:
            selected_sections = cls._select_pdf_sections(sections, cls._pdf_sections(reader))
            pages = sorted({
                page for section in selected_sections
                for page in range(section["start_page"], section["end_page"] + 1)
            })
        page_texts = {}
        spans = (cls._pdf_section_spans(reader, section_ranges, page_texts)
                 if section_ranges is not None else None)
        selected_pages = (sorted({span[0] for span in spans}) if spans is not None
                          else cls.parse_page_selection(pages, len(reader.pages)))
        pdf_title = pdf_path.stem
        if reader.metadata and reader.metadata.title:
            pdf_title = str(reader.metadata.title)

        page_documents: list[Document] = []
        empty_pages: list[int] = []
        section_mode = sections is not None or section_ranges is not None
        if spans is None:
            spans = []
            for page_number in selected_pages:
                page_texts[page_number] = cls._clean_pdf_text(
                    reader.pages[page_number - 1].extract_text() or ""
                )
                spans.append((page_number, 0, len(page_texts[page_number]), None, None))

        # One document per heading range or bookmark section, rather than per page.
        # Merge overlapping bookmark selections so shared pages are read only once.
        groups = {}
        bookmark_groups = []
        for section in sorted(selected_sections, key=lambda item: item["start_page"]):
            if bookmark_groups and section["start_page"] <= bookmark_groups[-1][1]:
                bookmark_groups[-1][1] = max(bookmark_groups[-1][1], section["end_page"])
            else:
                bookmark_groups.append([section["start_page"], section["end_page"]])
        for span in spans:
            page_number, left, right, title, next_title = span
            if not page_texts[page_number][left:right].strip():
                empty_pages.append(page_number)
                continue
            if section_ranges is not None:
                key = (title, next_title)
            elif selected_sections:
                key = next(index for index, (start, end) in enumerate(bookmark_groups)
                           if start <= page_number <= end)
            else:
                key = page_number
            groups.setdefault(key, []).append(span)

        document_segments = {}
        document_texts = {}
        for group_index, group in enumerate(groups.values()):
            first_page, _, _, title, next_title = group[0]
            document_id = f"{pdf_path.resolve()}#page={first_page}"
            if section_mode:
                document_id += f"&section={group_index}"
            metadata = {
                "title": pdf_title, "source": str(pdf_path),
                "file_name": pdf_path.name, "page_number": first_page,
            }
            if title is not None:
                metadata["section_title"] = title
                metadata["next_section_title"] = next_title
            parts = []
            segments = []
            offset = 0
            for page_number, left, right, _, _ in group:
                if parts:
                    parts.append("\n")
                    offset += 1
                text = page_texts[page_number][left:right]
                segments.append((offset, offset + len(text), page_number, left))
                parts.append(text)
                offset += len(text)
            text = "".join(parts)
            document_segments[document_id] = segments
            document_texts[document_id] = text
            page_documents.append(Document(id_=document_id, text=text, metadata=metadata))

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
            chunk.metadata["chunk_index"] = chunk_index
            start, end = chunk.start_char_idx, chunk.end_char_idx
            if start is None or end is None:
                raise ValueError("PDF chunk has no character offsets for source provenance")
            provenance = []
            for seg_start, seg_end, page_number, page_offset in document_segments[chunk.ref_doc_id]:
                left, right = max(start, seg_start), min(end, seg_end)
                if left >= right:
                    continue
                excerpt = document_texts[chunk.ref_doc_id][left:right]
                if not excerpt.strip():
                    continue
                page_start = page_offset + left - seg_start
                page_end = page_offset + right - seg_start
                page_text = page_texts[page_number]
                page_chunk_index = chunks_per_page.get(page_number, 0) + 1
                chunks_per_page[page_number] = page_chunk_index
                location = {
                    "source": str(pdf_path), "file_name": pdf_path.name,
                    "page_number": page_number, "page_chunk_index": page_chunk_index,
                    "scope": "source_chunk", "excerpt": excerpt,
                    "line_start": page_text.count("\n", 0, page_start) + 1,
                    "line_end": page_text.count("\n", 0, max(page_start, page_end - 1)) + 1,
                }
                if selected_sections:
                    location["sections"] = [
                        {"id": section["id"], "title": section["title"]}
                        for section in selected_sections
                        if section["start_page"] <= page_number <= section["end_page"]
                    ]
                if section_ranges is not None:
                    location["section_title"] = chunk.metadata["section_title"]
                    location["next_section_title"] = chunk.metadata["next_section_title"]
                provenance.append(location)
            chunk.metadata["provenance"] = provenance
            # Singular fields identify the first contributing page for compatibility.
            chunk.metadata["page_number"] = provenance[0]["page_number"]
            chunk.metadata["page_chunk_index"] = provenance[0]["page_chunk_index"]
            if section_mode:
                chunk.metadata["page_numbers"] = [item["page_number"] for item in provenance]
            if selected_sections:
                chunk.metadata["sections"] = [
                    {"id": section["id"], "title": section["title"]}
                    for section in selected_sections
                    if any(section["start_page"] <= item["page_number"] <= section["end_page"]
                           for item in provenance)
                ]
            # Do not duplicate the source excerpt in the model prompt.
            chunk.excluded_llm_metadata_keys.append("provenance")
            chunk.excluded_embed_metadata_keys.append("provenance")

        if empty_pages:
            print(f"Skipped PDF pages with no extractable text: {empty_pages}")
        print(
            f"Loaded {len({span[0] for group in groups.values() for span in group})} page(s) from '{pdf_path}' and "
            f"created {len(chunks)} chunks"
        )
        return chunks

    @staticmethod
    def _clean_pdf_text(text: str) -> str:
        """Normalize whitespace while preserving extracted PDF line numbers."""

        text = text.replace("\r\n", "\n").replace("\r", "\n")
        text = re.sub(r"[ \t]+\n", "\n", text)
        return text if text.strip() else ""

    @staticmethod
    def get_graph_data(graph_store: GraphRAGStore) -> dict:
        """Build visualization data in memory without writing generated files."""

        graph_store.ensure_community_members()
        nx_graph = graph_store.to_networkx()
        node_metadata = {
            node.id: {
                "id": node.id,
                "label": node.name,
                "type": node.label or "OTHER",
                "description": node.properties.get("entity_description", ""),
                "community": graph_store.community_primary.get(node.id),
                "community_ids": [
                    community for community, members in graph_store.community_members.items()
                    if node.id in members
                ],
            }
            for node in graph_store.get_entity_nodes()
        }
        nodes_data = [
            node_metadata[node_id]
            for node_id in nx_graph.nodes()
            if node_id in node_metadata
        ]
        # Export the directed property relations, not the lossy community graph.
        links_data = [
            {
                "source": relation.source_id,
                "target": relation.target_id,
                "label": relation.label,
                "description": relation.properties.get("relationship_description", ""),
                "provenance": relation.properties.get("provenance", []),
            }
            for relation in graph_store.graph.relations.values()
            if relation.source_id in node_metadata and relation.target_id in node_metadata
        ]
        graph_data = {
            "profile": GraphRAGSchema.ACTIVE_ONTOLOGY,
            "nodes": nodes_data,
            "links": links_data,
            "communities": len(graph_store.community_members),
            "community_membership_origin": graph_store.community_membership_origin,
            "community_details": [
                {
                    "id": community,
                    "members": members,
                    "summary": graph_store.community_summaries.get(community, "")
                    if graph_store.community_membership_origin == "built" else "",
                }
                for community, members in sorted(graph_store.community_members.items())
            ],
        }

        return graph_data

    @classmethod
    def export_graph_data(
        cls,
        graph_store: GraphRAGStore,
        output_file: str | Path | None = None,
    ) -> dict:
        """Export the graph to the profile's canonical JSON output."""
        destination = output_path(output_file, "graph_data.json")
        graph_data = cls.get_graph_data(graph_store)
        nodes_data = graph_data["nodes"]
        links_data = graph_data["links"]
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            json.dumps(graph_data, indent=2),
            encoding="utf-8",
        )
        print(f"Graph data exported to '{destination}'")
        print(
            f"Nodes: {len(nodes_data)} | Edges: {len(links_data)} | "
            f"Communities: {graph_data['communities']}"
        )
        return graph_data

    @classmethod
    def visualize_graph(
        cls,
        graph_store: GraphRAGStore | None = None,
        graph_data_file: str | Path | None = None,
        template_file: str | Path = Path(__file__).resolve().parents[1] / "graph_template.html",
        output_file: str | Path | None = None,
    ) -> Path:
        """Render graph data into the repository's interactive HTML template."""

        graph_data_path = output_path(graph_data_file, "graph_data.json")
        destination = output_path(output_file, "graph.html")
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

        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            template.replace(placeholder, json.dumps(graph_data).replace("<", "\\u003c")),
            encoding="utf-8",
        )
        print(f"Visualization saved to '{destination}'")
        return destination

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
