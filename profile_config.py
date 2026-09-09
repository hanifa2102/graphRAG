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
    required = {'input_type', 'input_file', 'pdf_chunk_size', 'chunk_overlap',
                'force_rebuild', 'output_prefix', 'questions', 'llm'}
    for name, config in profiles.items():
        if not isinstance(name, str) or not isinstance(config, dict):
            raise ValueError(f'Invalid profile settings for {name!r}: expected a named mapping.')
        missing = required - config.keys()
        if missing:
            raise ValueError(f'Incomplete profile settings for {name!r}: missing {sorted(missing)}.')
        modes = [key for key in ('pages', 'sections', 'section_ranges') if config.get(key) is not None]
        if len(modes) > 1:
            raise ValueError(f'{name}: choose only one of pages, sections, or section_ranges.')
        ranges = config.get('section_ranges')
        if ranges is not None and (
            not isinstance(ranges, list) or not ranges or any(
                not isinstance(pair, (list, tuple)) or len(pair) != 2
                or any(not isinstance(title, str) or not title.strip() for title in pair)
                for pair in ranges
            )
        ):
            raise ValueError(f'{name}: section_ranges must be a nonempty list of [title, next_title] pairs.')
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
    def normalize(name):
        return ' '.join(name.strip().lower().replace('_', ' ').replace('-', ' ').split())

    normalized = normalize(value)
    for name in PROFILES:
        if normalize(name) == normalized:
            return name
    raise ValueError(f'Unknown profile {value!r}. Choose one of {list(PROFILES)}.')


PROFILES = load_profiles()
