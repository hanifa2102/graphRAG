"""Ontology, extraction prompt, and structured extraction models."""

import os
from pathlib import Path
from string import Formatter
from typing import Literal, get_args

import yaml
from pydantic import BaseModel, Field

ONTOLOGY_FILE = Path(__file__).resolve().parent.parent / "ontology.yaml"
PROMPT_FILE = ONTOLOGY_FILE.with_name("prompt.yaml")


def _read_ontology(
    path: Path, profile: str | None = None,
) -> tuple[str, dict[str, dict[str, str]]]:
    """Read type labels and descriptions, failing early on invalid configuration."""

    with path.open(encoding="utf-8") as source:
        config = yaml.safe_load(source)
    if not isinstance(config, dict):
        raise ValueError(f"{path}: ontology must be a YAML mapping")
    profiles = config.get("ontologies")
    active = profile if profile is not None else config.get("active_ontology")
    if not isinstance(profiles, dict) or not profiles:
        raise ValueError(f"{path}: ontologies must be a non-empty mapping")
    if not isinstance(active, str) or active not in profiles:
        raise ValueError(f"{path}: selected profile {active!r} must name a profile in ontologies")
    ontology = profiles[active]
    if not isinstance(ontology, dict):
        raise ValueError(f"{path}: profile {active!r} must be a mapping")
    for section in ("entity_types", "relation_types"):
        definitions = ontology.get(section)
        if not isinstance(definitions, dict) or not definitions:
            raise ValueError(f"{path}: {section} must be a non-empty mapping")
        for label, description in definitions.items():
            if not isinstance(label, str) or not label.strip() or label != label.strip():
                raise ValueError(f"{path}: {section} labels must be non-empty strings without surrounding whitespace")
            if not isinstance(description, str) or not description.strip():
                raise ValueError(f"{path}: {section}.{label} must have a non-empty description")
    return active, ontology


def _load_ontology(path: Path) -> dict[str, dict[str, str]]:
    return _read_ontology(path)[1]


def _load_prompts(path: Path, profile: str) -> dict[str, str]:
    """Validate prompt placeholders before any LLM requests can be made."""

    with path.open(encoding="utf-8") as source:
        config = yaml.safe_load(source)
    profiles = config.get("prompts") if isinstance(config, dict) else None
    prompts = profiles.get(profile) if isinstance(profiles, dict) else None
    if not isinstance(prompts, dict):
        raise ValueError(f"{path}: missing or invalid prompt profile {profile!r}")
    expected_fields = {
        "extraction": {"entity_types", "relation_types", "max_knowledge_triplets", "text"},
        "community_summary": {"entities_text", "relationships_text"},
        "community_answer": {"summary", "query"},
        "aggregation": {"combined", "query"},
    }
    for name, required in expected_fields.items():
        template = prompts.get(name)
        context = f"{path}: {profile}.{name}"
        if not isinstance(template, str) or not template.strip():
            raise ValueError(f"{context} must be a non-empty prompt string")
        try:
            parsed = list(Formatter().parse(template))
        except ValueError as exc:
            raise ValueError(f"{context}: invalid template braces") from exc
        fields = {field for _, field, _, _ in parsed if field is not None}
        if fields != required:
            raise ValueError(f"{context}: expected placeholders {sorted(required)}, got {sorted(fields)}")
        if any(spec or conversion for _, _, spec, conversion in parsed):
            raise ValueError(f"{context}: format specifications and conversions are not supported")
    return prompts


ACTIVE_ONTOLOGY, _ONTOLOGY = _read_ontology(
    ONTOLOGY_FILE, profile=os.environ.get("GRAPH_RAG_PROFILE"),
)
_PROMPTS = _load_prompts(PROMPT_FILE, ACTIVE_ONTOLOGY)

# Runtime Literals retain Pydantic validation and JSON Schema enum constraints.
EntityType = Literal[tuple(_ONTOLOGY["entity_types"])]
RelationType = Literal[tuple(_ONTOLOGY["relation_types"])]


class ExtractedEntity(BaseModel):
    """One ontology-constrained entity returned by the extraction LLM."""

    name: str = Field(description="Name of the entity, capitalized")
    type: EntityType = Field(description="One of the allowed entity types")
    description: str = Field(description="Brief description of the entity and its role")


class ExtractedRelationship(BaseModel):
    """One ontology-constrained relationship returned by the extraction LLM."""

    source: str = Field(description="Name of the source entity")
    target: str = Field(description="Name of the target entity")
    relation: RelationType = Field(description="One of the allowed relationship types")
    description: str = Field(description="Sentence explaining the relationship")


class ExtractionResult(BaseModel):
    """Structured result returned for a document or chunk."""

    entities: list[ExtractedEntity] = Field(default_factory=list)
    relationships: list[ExtractedRelationship] = Field(default_factory=list)


class GraphRAGSchema:
    """Class 1: the domain ontology, prompt, and Pydantic output models."""

    ENTITY_TYPES = get_args(EntityType)
    RELATION_TYPES = get_args(RelationType)
    ACTIVE_ONTOLOGY = ACTIVE_ONTOLOGY

    ExtractedEntity = ExtractedEntity
    ExtractedRelationship = ExtractedRelationship
    ExtractionResult = ExtractionResult

    @classmethod
    def render_prompt(cls, name: str, **values: str) -> str:
        """Render a prompt from the profile selected in ontology.yaml."""

        return _PROMPTS[name].format(**values)

    @classmethod
    def extraction_prompt(cls) -> str:
        """Return the extraction prompt with this class's ontology embedded."""

        entity_types = "\n".join(
            f"- {label}: {description}"
            for label, description in _ONTOLOGY["entity_types"].items()
        )
        relation_types = "\n".join(
            f"- {label}: {description}"
            for label, description in _ONTOLOGY["relation_types"].items()
        )
        # Extraction is formatted twice: here for ontology definitions, then by
        # LlamaIndex for the source text and triplet limit. Preserve literal braces.
        parts = []
        definitions = {"entity_types": entity_types, "relation_types": relation_types}
        for literal, field, _, _ in Formatter().parse(_PROMPTS["extraction"]):
            parts.append(literal.replace("{", "{{").replace("}", "}}"))
            if field in definitions:
                parts.append(definitions[field].replace("{", "{{").replace("}", "}}"))
            elif field is not None:
                parts.append("{" + field + "}")
        return "".join(parts)
