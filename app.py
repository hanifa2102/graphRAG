"""Launch: streamlit run app.py -- xelsis (or ai_news)."""
import sys
import streamlit as st
from profile_config import resolve_profile
from causalrag_app import run_app


def main(arguments=None):
    args = sys.argv[1:] if arguments is None else arguments
    if len(args) != 1:
        st.error('Select a profile: streamlit run app.py -- xelsis (or ai_news).')
        return
    try:
        profile = resolve_profile(args[0])
    except ValueError as exc:
        st.error(str(exc))
        return
    run_app(profile)


if __name__ == '__main__':
    main()
