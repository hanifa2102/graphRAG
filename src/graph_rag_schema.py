"""Ontology, extraction prompt, and structured extraction models."""

from typing import Literal, get_args

from pydantic import BaseModel, Field

EntityType = Literal[
    "ORGANIZATION",
    "PERSON",
    "LEGISLATION",
    "LEGAL_CASE",
    "CONCEPT",
    "GOVERNMENT",
    "AI_SYSTEM",
]

RelationType = Literal[
    "FILED_AGAINST",
    "DEFENDANT_IN",
    "REGULATES",
    "ADVOCATES_FOR",
    "TRAINED_ON",
    "PART_OF",
    "REFERENCES",
    "OPPOSES",
]


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

    ExtractedEntity = ExtractedEntity
    ExtractedRelationship = ExtractedRelationship
    ExtractionResult = ExtractionResult

    @classmethod
    def extraction_prompt(cls) -> str:
        """Return the extraction prompt with this class's ontology embedded."""

        entity_types = ", ".join(cls.ENTITY_TYPES)
        relation_types = ", ".join(cls.RELATION_TYPES)
        return f"""
-Goal-
Given a news article about AI copyright, governance, or intellectual property,
identify all entities mentioned in the article and their relationships.

Extract up to {{max_knowledge_triplets}} entity-relation triplets.

-Allowed Entity Types-
{entity_types}

-Allowed Relationship Types-
{relation_types}

-Steps-
1. Identify ALL entities. For each entity extract:
   - name: Name of the entity, capitalized
   - type: One of the allowed entity types above
   - description: A brief description of the entity and its role in AI copyright/governance

2. Identify relationships between entities. For each pair extract:
   - source: name of the source entity
   - target: name of the target entity
   - relation: one of the allowed relationship types above
   - description: a sentence explaining why and how these entities are related

-Real Data-
######################
text: {{text}}
######################
"""
