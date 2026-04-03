from __future__ import annotations

import copy

from pipeline.providers.base import OpenAICompatibleProvider


def _prepare_strict_schema(schema: dict) -> dict:
    """Convert a standard JSON Schema to OpenAI's strict structured output subset.

    Changes:
    - Removes ``$schema`` and ``title`` keys.
    - Ensures ``additionalProperties: false`` on all object types (recursively).
    - Puts all properties into ``required`` if not already present.
    """
    schema = copy.deepcopy(schema)
    schema.pop("$schema", None)
    schema.pop("title", None)
    _make_strict_recursive(schema)
    return schema


def _make_strict_recursive(node: dict) -> None:
    """Recursively enforce strict-mode constraints on a schema node."""
    if not isinstance(node, dict):
        return

    if node.get("type") == "object":
        node["additionalProperties"] = False
        props = node.get("properties", {})
        if props:
            node["required"] = list(props.keys())
        for prop in props.values():
            _make_strict_recursive(prop)

    if node.get("type") == "array":
        items = node.get("items", {})
        _make_strict_recursive(items)


def _is_strict_compatible(schema: dict) -> bool:
    """Check if a schema can be used with OpenAI strict structured outputs.

    Schemas with ``additionalProperties`` set to a type constraint (dict) rather
    than ``false`` are not compatible with strict mode.
    """
    ap = schema.get("additionalProperties")
    if isinstance(ap, dict):
        return False
    return schema.get("type") == "object" and "properties" in schema


class OpenAIProvider(OpenAICompatibleProvider):
    """Standard OpenAI provider with structured output support."""

    _provider_label = "OpenAI"
    _env_var = "OPENAI_API_KEY"
    _base_url = "https://eu.api.openai.com/v1"

    def _get_response_format(self, schema: dict | None = None) -> dict:
        """Use strict JSON Schema mode when a schema is provided and compatible.

        Schemas that use ``additionalProperties`` with a type constraint (rather
        than ``false``) are not compatible with OpenAI strict mode.  In that case,
        fall back to basic json_object mode.
        """
        if schema is not None and _is_strict_compatible(schema):
            strict_schema = _prepare_strict_schema(schema)
            return {
                "type": "json_schema",
                "json_schema": {
                    "name": strict_schema.get("title", "extraction"),
                    "strict": True,
                    "schema": strict_schema,
                },
            }
        return {"type": "json_object"}
