"""Stable provider wire schemas; dynamic coverage remains a local concern."""

from __future__ import annotations

import copy
from typing import Any

from pydantic import BaseModel

from story_engine.agents.evaluation import EvaluationWireResponse
from story_engine.planning.asset_alignment import AssetAlignmentWireResponse
from story_engine.planning.participation import FirstFrameParticipationWireResponse
from story_engine.planning.story_planner import StoryDraft
from story_engine.providers.ports import ResponseContract
from story_engine.spatial.grounding import GroundingWireResponse

_MODELS: dict[ResponseContract, type[BaseModel]] = {
    ResponseContract.ASSET_ALIGNMENT: AssetAlignmentWireResponse,
    ResponseContract.STORY_DRAFT: StoryDraft,
    ResponseContract.GROUNDING_ROWS: GroundingWireResponse,
    ResponseContract.EVALUATION_ROWS: EvaluationWireResponse,
    ResponseContract.FIRST_FRAME_PARTICIPATION: FirstFrameParticipationWireResponse,
}


def schema_for_contract(contract: ResponseContract) -> dict[str, Any]:
    """Return one stable strict JSON schema for a named response contract."""

    schema = _inline_local_references(_MODELS[contract].model_json_schema())
    _make_all_fields_required(schema)
    return schema


def _inline_local_references(schema: dict[str, Any]) -> dict[str, Any]:
    source = copy.deepcopy(schema)
    definitions = source.get("$defs", {})

    def normalize(value: Any, stack: tuple[str, ...] = ()) -> Any:
        if isinstance(value, dict):
            reference = value.get("$ref")
            if isinstance(reference, str) and reference.startswith("#/$defs/"):
                name = reference.removeprefix("#/$defs/")
                target = definitions.get(name)
                if not isinstance(target, dict):
                    raise ValueError(f"unresolved local schema reference: {reference}")
                if name in stack:
                    raise ValueError(f"recursive wire schema is unsupported: {reference}")
                merged = {
                    **target,
                    **{key: item for key, item in value.items() if key != "$ref"},
                }
                return normalize(merged, (*stack, name))
            return {
                key: normalize(item, stack)
                for key, item in value.items()
                if key not in {"$defs", "title", "default"}
            }
        if isinstance(value, list):
            return [normalize(item, stack) for item in value]
        return value

    result = normalize(source)
    if not isinstance(result, dict):
        raise ValueError("wire schema root must be an object")
    return result


def _make_all_fields_required(value: Any) -> None:
    if isinstance(value, dict):
        properties = value.get("properties")
        if isinstance(properties, dict):
            value["required"] = list(properties)
            value["additionalProperties"] = False
        for item in value.values():
            _make_all_fields_required(item)
    elif isinstance(value, list):
        for item in value:
            _make_all_fields_required(item)
