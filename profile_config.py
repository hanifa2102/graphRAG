"""Read and validate profiles.yaml without importing graph or model dependencies."""
from pathlib import Path
import yaml

PROFILE_FILE = Path(__file__).resolve().parent / 'profiles.yaml'


def load_profiles(path=PROFILE_FILE):
    with Path(path).open(encoding='utf-8') as source:
        document = yaml.safe_load(source)
    profiles = document.get('profiles') if isinstance(document, dict) else None
    if not isinstance(profiles, dict) or not profiles:
        raise ValueError('profiles.yaml must contain a nonempty profiles mapping.')
    required = {'input_type', 'input_file', 'pages', 'pdf_chunk_size', 'chunk_overlap',
                'force_rebuild', 'output_prefix', 'questions', 'llm'}
    for name, config in profiles.items():
        if not isinstance(name, str) or not isinstance(config, dict) or required - config.keys():
            raise ValueError(f'Incomplete profile settings for {name!r}.')
        if config['input_type'] != 'pdf' or not str(config['input_file']).lower().endswith('.pdf'):
            raise ValueError(f'{name}: input must be a PDF.')
        size, overlap = config['pdf_chunk_size'], config['chunk_overlap']
        if type(size) is not int or type(overlap) is not int or not 0 <= overlap < size:
            raise ValueError(f'{name}: require 0 <= chunk_overlap < pdf_chunk_size.')
        if type(config['force_rebuild']) is not bool:
            raise ValueError(f'{name}: force_rebuild must be a YAML boolean.')
        if not isinstance(config['questions'], list) or not config['questions'] or not all(isinstance(q, str) and q.strip() for q in config['questions']):
            raise ValueError(f'{name}: questions must be a nonempty list of text.')
        if not isinstance(config['llm'], dict) or config['llm'].get('provider') not in ('qwen', 'openai'):
            raise ValueError(f'{name}: llm.provider must be qwen or openai.')
    return profiles


def resolve_profile(value):
    normalized = value.strip().lower().replace('_', ' ').replace('-', ' ')
    for name in PROFILES:
        if name.lower() == normalized:
            return name
    raise ValueError(f'Unknown profile {value!r}. Choose xelsis or ai_news.')


PROFILES = load_profiles()
