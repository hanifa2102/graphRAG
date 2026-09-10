# CausalRAG — PDF knowledge graphs

Build and query a knowledge graph from text-based PDFs using the `Xelsis` or
`AI news` ontology and prompt profile. The notebook runs PDF loading, extraction,
Leiden community detection, summarization, visualization, and queries.

## Setup

Use Python 3.10+ and install `requirements.txt` in your virtual environment:

```bash
pip install -r requirements.txt
```

Configure the selected provider in `.env`: `LITELLM_API_KEY` for local Qwen through
LiteLLM, or `OPENAI_API_KEY` for OpenAI. The profile's `llm` settings in `profiles.yaml` select the provider and models.

## Run Streamlit

From `causalRAG/`, activate the virtual environment and choose the profile:

```bash
source .venv/bin/activate
streamlit run app.py -- xelsis
# Or, in a separate server:
streamlit run app.py --server.port 8502 -- ai_news
```

`app.py` is the single entry point. The argument after `--` selects the profile;
`xelsis`, `ai_news`, and quoted `"AI news"` are accepted. Run each profile in a
separate process because its ontology is selected at import time. Streamlit's
own flags go before `--`. An equivalent command that guarantees the project
Python environment is `.venv/bin/python -m streamlit run app.py -- xelsis`.

The app automatically loads `output/<profile>_graph_store.pkl`. If no checkpoint
exists, it explains how to build one in the notebook. Browsing a graph requires no
LLM credentials or requests, and does not write JSON/HTML files. **Reload graph**
refreshes a rebuilt checkpoint. Loading failures display the actual error,
checkpoint path, and Python executable; a rebuild is not automatically required.

- **Graph explorer** preserves community/type colors, community and entity-type
  filters, member lists and summaries, search, node selection and highlighting,
  node dragging, pan/zoom/reset, layout sliders, edge-label controls, statistics,
  and compact provenance on edge hover in the graph's left panel.
- **Query the System** accepts a typed question and runs the same community-answer
  and aggregation engine as the notebook. Example questions and a per-session
  history are included. **Ask** is disabled when summaries are absent. Queries
  use the complete saved summaries, independent of visualization filters.

All profile settings now live together in `profiles.yaml`, shared with the notebook:
PDF path/pages/chunking, output prefix, questions, and the `llm` options. Restart
the Streamlit server after editing the YAML. The default
uses the notebook's Qwen model and LiteLLM endpoint. For OpenAI, set the profile's
`llm.provider`, `llm.extraction_model`, and `llm.query_model` together. Existing
`.env` credentials remain server-side; the app never sends them to the graph.

Streamlit is the main UI. `graph_template.html` remains the internal D3 renderer
inside a sandboxed custom component so its existing interactions are preserved.
The app renders it in memory; standalone HTML export remains optional. Generated
JSON/checkpoints continue to use only the profile's canonical `output/` files.
Question history lives in the browser session's server-side state, not output files.

## Run the notebook

Open `graphrag_ai_copyright_refactored.ipynb` from `causalRAG/` or its parent.
Select `PROFILE` in Section 1 before importing `src`; edit its settings in `profiles.yaml`:

| Profile | PDF input | Default pages |
|---|---|---|
| `Xelsis` | `data/Xelsis.pdf` | 5, 7–8, 14–16, 18–21, 28–30 |
| `AI news` | `data/ai_news.pdf` | All pages |

Supply a text-based AI news PDF at the configured path before running that
profile. CSV loading is removed. Scanned/image-only PDFs require OCR before
loading; empty pages are skipped and reported. `PDF_PAGES` accepts 1-based page
selections such as `"5, 7-8"`, a list of page numbers, or `None` for every page.
Each entry in `PROFILES` groups `input_file`, `pages`, `pdf_chunk_size`,
`chunk_overlap`, `force_rebuild`, output prefix, and questions. Selecting `PROFILE`
loads those settings together; page-based loading keeps chunks within pages.

The Python service and manager also accept bookmark-based section selection:

```python
from src.graph_rag_services import GraphRAGService

sections = GraphRAGService.list_pdf_sections("data/Xelsis.pdf")
for section in sections:
    print(section)  # id, title, level, start_page, end_page

chunks = GraphRAGService.load_pdf_documents("data/Xelsis.pdf", sections=["1"])
# Also supported by manager.load_pdf_documents() and manager.build_pdf_and_store().
```

Use IDs from the listing (e.g. `"1.2"`) or exact, case-insensitive bookmark titles.
IDs reflect bookmark order, not printed section numbers. Pass multiple sections
as a list; commas in titles are preserved. Parent sections include their nested
subsections, and overlapping selections load each page once. Choose either
`pages` or `sections` in a call. Selected section IDs/titles are retained in chunk
metadata and provenance.

Bookmark selections start at the bookmark page and end before the next bookmark at
the same or higher level (or at the end of the PDF). Selection loads whole pages;
bookmarks starting on the same page share that page. It does not trim text at
headings or infer sections from a printed table of contents. PDFs without
bookmarks can use `pages` or explicit heading ranges; `list_pdf_sections()` returns
an empty list for them.

For exact heading boundaries, pass `(title, next_title)` pairs as `section_ranges`:

```python
chunks = GraphRAGService.load_pdf_documents(
    "data/152E-GEN-8002(doc)-C.pdf",
    section_ranges=[(
        "5.1.1 SYSTEM DESCRIPTION",
        "5.1.2 FUNCTIONAL DESCRIPTION OF AUTOMATIC SLIDING DOOR",
    )],
    chunk_overlap=0,
)
print("\n".join(chunk.text for chunk in chunks))
```

The start heading is included; the next heading is excluded. This includes any
continuation on the next heading's page. A single tuple or a list of tuples is
accepted, including several ranges on the same page. Heading matching ignores
case and differences in spaces/tabs, and requires a standalone extracted line.
Bookmarks narrow the search when available; otherwise the loader searches page
text and rejects missing or ambiguous headings. Headers/footers within a range
are retained. Section text is joined across pages before chunking, so a section
that fits the token budget becomes one chunk even when it spans several pages.
Larger sections split using `chunk_size` and `chunk_overlap`. Separate heading
ranges remain separate documents. Bookmark selections also join across pages;
overlapping bookmark selections are grouped to avoid duplicating shared pages.

Each section chunk has `page_numbers` and a `provenance` entry for every
contributing page, with its own excerpt and original line range. The singular
`page_number` and `page_chunk_index` fields refer to the first contributing page.
Page-based loading still chunks each page separately. Use only one of
`pages`, `sections`, or `section_ranges`. Both manager loading methods also accept
`section_ranges`.

Restart the kernel and run from the top after switching profiles or editing YAML.
Set `FORCE_REBUILD = True` after changing inputs, pages, chunk settings, provider,
ontology, or prompts. Otherwise the selected profile's checkpoint is reused.

## Outputs

`output/<profile>_graph_data.json` includes a `rejected_relationships` audit list.
When the extraction model references an endpoint it did not declare as an entity,
the rejected relationship is recorded with its source/target, label, description,
`missing_endpoints`, reason, chunk ID, source excerpt, and page/line provenance.
These entries are excluded from graph nodes and links but survive checkpoint
save/reload. Rebuild extraction to capture new audit records; previously printed
rejections cannot be recovered from older checkpoints.

All generated artifacts reside in `causalRAG/output/`, independent of the working
directory. Each profile has exactly one checkpoint, JSON export, and HTML file:

| Profile | Checkpoint | JSON | Visualization |
|---|---|---|---|
| Xelsis | `xelsis_graph_store.pkl` | `xelsis_graph_data.json` | `xelsis_graph.html` |
| AI news | `ai_news_graph_store.pkl` | `ai_news_graph_data.json` | `ai_news_graph.html` |

Rebuilding overwrites that profile's files; provider-specific variants are no
longer created. Service/manager methods default to these paths and reject custom
output paths. `graph_template.html` is the source template, not generated output.
Checkpoints contain graph state and community summaries, excluding live LLM
clients. Load only trusted checkpoints.

## Provenance and visualization

Each extracted triple stores a `provenance` list with its PDF filename/source,
1-based physical PDF page, page-local chunk number, extracted-text line range,
and source chunk excerpt. Line ranges identify the **source chunk**, not an exact
supporting sentence, and refer to pypdf's extracted text rather than printed line
numbers. Sections are not guessed. Repeated triples merge all distinct source
occurrences. Provenance survives checkpoint save/load and JSON/HTML export.

Open the Streamlit Graph explorer (or optional HTML export) and hover over an edge. The left sidebar shows the triple,
description, and all source locations/excerpts. The last hovered edge remains
visible so you can scroll and read it. Different predicates and reverse relations
between the same nodes remain distinct in JSON and appear as separate curves.
Nodes are colored by their most specific community by default. Use **Communities**
in the sidebar to isolate a group, see its member nodes, internal-edge count, and
summary. Switch **Color nodes by** to **Entity type** to restore type colors.
Parent communities may overlap; unassigned nodes appear in gray. **Show all**
clears both community and type filters.

Community memberships are persisted with new checkpoints. Older checkpoints
recover groups from graph topology without LLM calls. Their original summary IDs
cannot be matched reliably to recovered groups, so the panel asks you to rebuild
communities for matching summaries. Existing query summaries remain available.

The graph also supports node search, type filters, node selection, and layout
controls. D3 and fonts are loaded from CDNs.

Legacy graphs built without provenance need a PDF rebuild to capture all source
occurrences and line ranges. Existing CSV-derived graphs cannot acquire PDF page
numbers without being rebuilt from a PDF.

## Ontology and prompts

`ontology.yaml` defines named profiles under `ontologies`, each with `entity_types`
and `relation_types` mapped to descriptions. The notebook sets `GRAPH_RAG_PROFILE`
to override the YAML `active_ontology` default for the process. The same profile
selects all four templates in `prompt.yaml`. Allowed labels constrain the Pydantic
extraction models; missing profiles and invalid configuration fail early.

| Prompt | Required placeholders |
|---|---|
| `extraction` | `{entity_types}`, `{relation_types}`, `{max_knowledge_triplets}`, `{text}` |
| `community_summary` | `{entities_text}`, `{relationships_text}` |
| `community_answer` | `{summary}`, `{query}` |
| `aggregation` | `{combined}`, `{query}` |

Use `{{` and `}}` for literal braces. Preserve `No relevant information.` in the
community-answer prompt; the query engine uses it to discard irrelevant answers.

## Python API

```python
from src import GraphRAGManager

manager = GraphRAGManager(provider="openai")
manager.load_pdf_documents("data/Xelsis.pdf", pages="5, 7-8")
manager.build_knowledge_graph()
manager.save_knowledge_graph()
manager.visualize()
# Later, with the same active profile:
manager.load_knowledge_graph()
```

Set `GRAPH_RAG_PROFILE` before importing `src` to use a profile other than the
YAML default. `build_pdf_and_store()` also provides a combined load/build/save
workflow.

Run offline validation (mocked LLM, no API requests):

```bash
python -m unittest discover -s tests -v
node tests/test_visualization.js
```
