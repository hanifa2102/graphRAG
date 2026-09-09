"""Profile-specific Streamlit graph explorer and community-summary question UI."""
import logging
import sys

import streamlit as st
import streamlit.components.v1 as components

from profile_config import PROFILES
from streamlit_backend import ROOT, activate_profile, answer_question, checkpoint_path, load_graph, render_graph

_graph_component = components.declare_component('causalrag_graph', path=str(ROOT / 'components' / 'graph'))


@st.cache_data(show_spinner=False)
def cached_graph(profile, checkpoint_mtime_ns, checkpoint_size):
    return load_graph(profile)


def run_app(profile):
    st.set_page_config(page_title=f'{profile} · CausalRAG', layout='wide')
    st.title(f'{profile} · CausalRAG')
    try:
        config = activate_profile(profile)
    except (ValueError, RuntimeError) as exc:
        st.error(str(exc))
        st.stop()
    with st.sidebar:
        st.header(profile)
        st.caption(f"PDF: {config['input_file']}")
        st.caption(f"Query provider: {config['llm']['provider']} · {config['llm']['query_model']}")
        st.caption('Change profile and provider settings in profiles.yaml. Run each profile in its own server.')
        if st.button('Reload graph'):
            cached_graph.clear()
            st.rerun()
    path = checkpoint_path(profile)
    if not path.is_file():
        st.info(f"No saved graph for {profile}. Build this profile in the notebook first. Expected: output/{path.name}")
        st.stop()
    stamp = path.stat()
    try:
        with st.spinner('Loading graph…'):
            data, has_summaries = cached_graph(profile, stamp.st_mtime_ns, stamp.st_size)
    except Exception as exc:
        logging.getLogger(__name__).exception('Failed to load profile %s from %s', profile, path)
        st.error(f'Could not load the saved graph: {type(exc).__name__}: {exc}')
        with st.expander('Loading diagnostics', expanded=True):
            st.code(f"Python: {sys.executable}\nCheckpoint: {path}")
            st.caption('Use the project virtual environment. A loading error does not necessarily mean the checkpoint needs rebuilding.')
        st.stop()
    st.caption(f"{len(data['nodes'])} nodes · {len(data['links'])} edges · {data['communities']} communities")
    graph_tab, query_tab = st.tabs(['Graph explorer', 'Query the System'])
    with graph_tab:
        st.caption('Use the graph’s left panel for colors, communities, type filters, and compact edge provenance. Drag nodes, pan, or scroll to zoom.')
        _graph_component(html=render_graph(data), height=800, key=f'graph-{profile}')
    with query_tab:
        st.subheader('Query the System')
        st.caption('Answers use the saved community summaries, with the same two-stage query engine as the notebook. Graph filters do not restrict the question.')
        state_key = f'questions-{profile}'
        revision_key = f'question-revision-{profile}'
        revision = (stamp.st_mtime_ns, stamp.st_size)
        if st.session_state.get(revision_key) != revision:
            st.session_state[state_key] = []
            st.session_state[revision_key] = revision
        if not has_summaries:
            st.warning('This graph has no community summaries. Build communities in the notebook before querying.')
        with st.expander('Example questions'):
            for example in config['questions']:
                st.markdown(f'- {example}')
        with st.form(f'query-form-{profile}', clear_on_submit=False):
            question = st.text_area('Your question', placeholder=config['questions'][0])
            submitted = st.form_submit_button('Ask', disabled=not has_summaries)
        if submitted:
            if not question.strip():
                st.warning('Enter a question first.')
            else:
                try:
                    with st.spinner('Searching community summaries and composing an answer…'):
                        answer = answer_question(profile, question.strip())
                except ValueError as exc:
                    st.error(str(exc))
                except Exception:
                    st.error('The query could not be completed. Check the configured provider, server credentials, and connection, then try again.')
                else:
                    st.session_state[state_key].append({'question': question.strip(), 'answer': answer})
        if st.button('Clear question history'):
            st.session_state[state_key] = []
        for result in reversed(st.session_state[state_key]):
            with st.chat_message('user'):
                st.write(result['question'])
            with st.chat_message('assistant'):
                st.write(result['answer'])
