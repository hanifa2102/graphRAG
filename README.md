# GraphRAG — AI Copyright & Governance Knowledge Graph

An end-to-end GraphRAG pipeline that scrapes web content about AI intellectual property and copyright, extracts entities and relationships using LLMs, builds a queryable knowledge graph, and renders it as an interactive visualization.

## Overview

This project combines web scraping, LLM-powered entity extraction, graph analysis, and interactive visualization to build a domain-specific Retrieval-Augmented Generation (RAG) system focused on AI copyright and governance topics.

## Tech Stack

| Layer | Tools |
|-------|-------|
| LLM / RAG | LlamaIndex, OpenAI, or local Qwen through LiteLLM |
| Web scraping | SerpAPI, Trafilatura, YouTube Transcript API |
| Graph analysis | NetworkX, Graspologic (Louvain community detection) |
| Visualization | D3.js v7, vis-network 9.1.2 |
| Data | Pandas, Pydantic |


## Project Structure

```
graphrag/
├── scrape_info.ipynb              # Web scraping & text enrichment
├── graphrag_ai_copyright.ipynb    # GraphRAG pipeline, queries & visualization
├── ai_copyright_dataset.csv       # Scraped articles/videos (810 rows)
├── graph_data.json                # Extracted knowledge graph (nodes + edges)
├── ai_copyright_graph.html        # Interactive visualization (generated output)
├── graph_template.html            # HTML/D3.js template for visualization
├── .env                           # API keys (not committed)
└── lib/
    ├── bindings/utils.js          # Graph interaction utilities
    ├── vis-9.1.2/                 # vis-network library
    └── tom-select/                # Dropdown UI component
```

## Prerequisites

- Python 3.10+
- A [SerpAPI](https://serpapi.com/) key (Google Search)
- An OpenAI API key, a LiteLLM API key, or both depending on the selected provider
- API key variables added in the `.env` file

## Setup

1. **Clone the repo and create a virtual environment:**
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   ```

2. **Install dependencies:**
   ```bash
   pip install llama-index llama-index-llms-openai \
               llama-index-llms-openai-like graspologic \
               pandas nest-asyncio python-dotenv \
               google-search-results trafilatura youtube_transcript_api requests
   ```

3. **Configure API keys** — create a `.env` file in the project root:
   ```
   SERPAPI_KEY=your_serpapi_key_here
   OPENAI_API_KEY=your_openai_api_key_here
   LITELLM_API_KEY=your_litellm_api_key_here
   ```

## Usage

### Configure the ontology

`ontology.yaml` contains named profiles under `ontologies`, each defining
`entity_types` and `relation_types` as output labels mapped to descriptions.
`Xelsis` covers machine operation, maintenance, and troubleshooting. `AI news`
preserves the original seven entity types and eight relationship types for the
AI copyright dataset. Set `active_ontology: Xelsis` (the default) or
`active_ontology: AI news` to select the vocabulary used for extraction and
validation. Only the selected profile's labels are accepted.

In `graphrag_ai_copyright_refactored.ipynb`, select `PROFILE = "Xelsis"` or
`PROFILE = "AI news"` in Section 1 before importing `src`. This sets
`GRAPH_RAG_PROFILE` for the Python process, overriding the YAML default without
editing it. The notebook loads the Xelsis PDF's selected pages or the AI news
CSV respectively. PDF page/chunk settings and the CSV article limit are in the
same cell. Restart the kernel and run from the top after switching profiles.

Notebook output filenames include both the profile and LLM provider, for example
`xelsis_qwen_graph_store.pkl` and `ai_news_qwen_graph_store.pkl`. Set
`FORCE_REBUILD = True` after changing inputs, selected pages, chunk settings,
ontology, or prompts. Existing checkpoints otherwise bypass extraction.

`prompt.yaml` stores the matching `AI news` and `Xelsis` profiles under `prompts`.
The selected ontology profile selects all four LLM templates: `extraction`,
`community_summary`, `community_answer`, and `aggregation`. There is no separate
prompt selector, so prompts and ontology stay aligned. To add a domain, add a
profile with the same name to both YAML files.

Edit prompt wording freely, but preserve each template's placeholders:

| Template | Required placeholders |
|---|---|
| `extraction` | `{entity_types}`, `{relation_types}`, `{max_knowledge_triplets}`, `{text}` |
| `community_summary` | `{entities_text}`, `{relationships_text}` |
| `community_answer` | `{summary}`, `{query}` |
| `aggregation` | `{combined}`, `{query}` |

Use `{{` and `}}` for literal braces, such as JSON examples. Keep the exact
`No relevant information.` response instruction in `community_answer`, because
the query engine uses it to discard irrelevant answers. Missing prompt profiles,
empty templates, and missing or unknown placeholders fail at import time.

`src/graph_rag_schema.py` loads this file relative to the project directory,
independently of the working directory. Both the extraction prompt and Pydantic
structured-output constraints use these definitions. Missing or invalid
configuration fails at import time instead of falling back to hardcoded labels.

After editing the YAML, restart Python or the Jupyter kernel and rerun the
notebook cells. Rebuild the graph with a new checkpoint filename: existing
checkpoints retain the ontology and summaries used when they were built.
This configures the vocabulary; it does not add new graph properties, PDF
notebook wiring, or change the current undirected community graph conversion.

Run the offline schema checks with:
```bash
python -m unittest discover -s tests -v
```

### Step 1 — Scrape content (`scrape_info.ipynb`)

Searches Google for AI copyright/IP articles and YouTube videos, then enriches them with full article text (via Trafilatura) and video transcripts (via YouTube Transcript API).
You can swap out the search queries with any other search terms relevant to your project.

Output: `ai_copyright_dataset.csv`

### Step 2 — Build the knowledge graph (`graphrag_ai_copyright.ipynb`)

Loads the scraped dataset and runs the full GraphRAG pipeline. Set `USE_QWEN`
in the configuration cell to `True` for local Qwen or `False` for OpenAI.

- **Entity extraction** — uses the selected provider to extract entities and relationships under the defined ontology.
- **Relationship extraction** — maps typed relationships between entities using Pydantic-validated schemas
- **Community detection** — runs Louvain clustering to group related entities into thematic communities
- **Community summarization** — generates LLM summaries for each cluster
- **Query engine** — answers complex questions by synthesizing context across communities using the selected provider
- **Export** — writes provider-specific JSON, HTML, and checkpoint files

### Step 3 — Explore the visualization

Open `ai_copyright_graph.html` in a browser. Features:

- Force-directed graph layout (D3.js)
- Filter nodes by entity type via the sidebar legend
- Search nodes by name
- Click a node to highlight its direct connections
- Hover for entity details in a tooltip
- Adjust link distance with the slider
