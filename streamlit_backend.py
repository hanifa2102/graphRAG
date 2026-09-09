"""Server-side profile loading and queries; no Streamlit or LLM work at import."""
import json
import os
from pathlib import Path
import sys

from profile_config import PROFILES

ROOT = Path(__file__).resolve().parent


def activate_profile(profile):
    if profile not in PROFILES:
        raise ValueError(f"Unknown profile: {profile}")
    schema = sys.modules.get('src.graph_rag_schema')
    if schema is not None and schema.ACTIVE_ONTOLOGY != profile:
        raise RuntimeError('Run each profile in a separate Streamlit server process.')
    os.environ['GRAPH_RAG_PROFILE'] = profile
    return PROFILES[profile]


def checkpoint_path(profile):
    return ROOT / 'output' / f"{PROFILES[profile]['output_prefix']}_graph_store.pkl"


def load_graph(profile):
    activate_profile(profile)
    from src import GraphRAGManager, GraphRAGService
    # Browsing a saved graph does not need LLM clients or credentials.
    manager = GraphRAGManager.__new__(GraphRAGManager)
    store = manager.load_knowledge_graph(checkpoint_path(profile))
    data = GraphRAGService.get_graph_data(store)
    return data, bool(store.get_community_summaries())


def render_graph(data):
    template = (ROOT / 'graph_template.html').read_text(encoding='utf-8')
    return template.replace('GRAPH_DATA_PLACEHOLDER', json.dumps(data).replace('<', r'\u003c'))


def answer_question(profile, question):
    question = question.strip()
    if not question:
        raise ValueError('Enter a question first.')
    config = activate_profile(profile)
    from dotenv import load_dotenv
    load_dotenv(ROOT / '.env')
    variable = 'LITELLM_API_KEY' if config['llm']['provider'] == 'qwen' else 'OPENAI_API_KEY'
    if not os.getenv(variable):
        raise ValueError(f'{variable} is not configured in the server environment or .env.')
    from src import GraphRAGManager
    manager = GraphRAGManager(**config['llm'])
    manager.load_knowledge_graph(checkpoint_path(profile))
    return manager.query(question)[question]
