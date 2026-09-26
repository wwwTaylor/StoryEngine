"""Structured, validated alignment between provided assets and idea concepts."""

from __future__ import annotations

import json
from typing import Literal

from pydantic import Field, ValidationError, model_validator

from story_engine.domain.common import FrozenModel, WireModel
from story_engine.domain.request import ProjectRequest, ProvidedAssetBinding
from story_engine.errors import ContractError, ProviderError
from story_engine.providers.ports import (
    ProviderCallMetrics,
    ResponseContract,
    TextProvider,
    TextRequest,
)


ALIGNMENT_AUTO_CONFIDENCE = 0.85


class AssetIdeaConcept(WireModel):
    concept_id: str = Field(min_length=1, max_length=80)
    kind: Literal["character", "scene"]
    label: str = Field(min_length=1, max_length=500)
    description: str = Field(min_length=1, max_length=1_000)
    idea_mentions: list[str] = Field(default_factory=list)
    required_entity_alias: str = Field(
        default="",
        max_length=80,
        pattern=r"^$|^[A-Za-z][A-Za-z0-9_.-]*$",
    )


class AssetAlignmentMatch(WireModel):
    asset_id: str = Field(min_length=1, max_length=500)
    candidate_concept_ids: list[str] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0)
    rationale: str = Field(min_length=1, max_length=1_000)


class AssetAlignmentWireResponse(WireModel):
    concepts: list[AssetIdeaConcept]
    matches: list[AssetAlignmentMatch]


class AssetAlignmentIssue(FrozenModel):
    code: Literal[
        "ASSET_NO_MATCH",
        "ASSET_AMBIGUOUS",
        "ASSET_DUPLICATE_BINDING",
        "ASSET_TYPE_MISMATCH",
    ]
    asset_id: str
    asset_name: str
    message: str
    candidate_concept_ids: tuple[str, ...] = ()
    suggestions: tuple[str, ...] = ()


class AssetAlignmentProposal(FrozenModel):
    mode: Literal["auto", "manual"]
    status: Literal["auto_approved", "confirmation_required", "input_invalid"]
    concepts: tuple[AssetIdeaConcept, ...]
    matches: tuple[AssetAlignmentMatch, ...]
    suggested_bindings: tuple[ProvidedAssetBinding, ...]
    issues: tuple[AssetAlignmentIssue, ...] = ()
    provider_call_ref: str | None = None

    @model_validator(mode="after")
    def validate_status(self) -> AssetAlignmentProposal:
        if self.status == "auto_approved" and self.issues:
            raise ValueError("auto-approved alignment cannot contain issues")
        return self


async def align_project_assets(
    request: ProjectRequest,
    planner: TextProvider,
    *,
    mode: Literal["auto", "manual"],
) -> AssetAlignmentProposal:
    """Ask the planner for concepts, then enforce deterministic binding safety locally."""

    assets = tuple(asset for asset in request.provided_assets if asset.kind in {"character", "scene"})
    if not assets:
        return AssetAlignmentProposal(
            mode=mode,
            status="auto_approved",
            concepts=(),
            matches=(),
            suggested_bindings=(),
        )
    capability = planner.capability_profile.text
    if capability is None:
        raise ContractError("asset alignment requires planner text capability")
    prompt = render_asset_alignment_prompt(request)
    if len(prompt) > capability.max_prompt_characters:
        raise ContractError("asset alignment prompt exceeds planner context")
    try:
        result = await planner.generate_text(
            TextRequest(prompt=prompt, response_contract=ResponseContract.ASSET_ALIGNMENT)
        )
    except ProviderError:
        raise
    try:
        wire = AssetAlignmentWireResponse.model_validate(json.loads(result.text))
    except (json.JSONDecodeError, ValidationError) as exc:
        raise ContractError(f"invalid asset alignment response: {exc}") from exc
    return validate_asset_alignment(request, wire, mode=mode, metrics=result.metrics)


def validate_asset_alignment(
    request: ProjectRequest,
    wire: AssetAlignmentWireResponse,
    *,
    mode: Literal["auto", "manual"],
    metrics: ProviderCallMetrics | None = None,
) -> AssetAlignmentProposal:
    assets = tuple(asset for asset in request.provided_assets if asset.kind in {"character", "scene"})
    assets_by_id = {asset.asset_id: asset for asset in assets}
    concepts_by_id = {item.concept_id: item for item in wire.concepts}
    if len(concepts_by_id) != len(wire.concepts):
        raise ContractError("asset alignment returned duplicate concept IDs")
    required_aliases = set(request.generation_requirements.required_entities)
    claimed_required_aliases: set[str] = set()
    for concept in wire.concepts:
        alias = concept.required_entity_alias
        if not alias:
            continue
        if concept.kind != "character" or alias not in required_aliases:
            raise ContractError(
                f"asset alignment returned invalid required entity alias: {alias}"
            )
        if alias in claimed_required_aliases:
            raise ContractError(
                f"asset alignment returned duplicate required entity alias: {alias}"
            )
        claimed_required_aliases.add(alias)
    matches_by_id = {item.asset_id: item for item in wire.matches}
    if len(matches_by_id) != len(wire.matches) or set(matches_by_id) != set(assets_by_id):
        raise ContractError("asset alignment must return exactly one row per provided asset")

    issues: list[AssetAlignmentIssue] = []
    suggested: list[ProvidedAssetBinding] = []
    claimed: dict[str, str] = {}
    kind_positions: dict[str, int] = {"character": 0, "scene": 0}
    for asset in assets:
        match = matches_by_id[asset.asset_id]
        candidates = tuple(dict.fromkeys(match.candidate_concept_ids))
        valid_candidates: list[AssetIdeaConcept] = []
        type_mismatch = False
        for concept_id in candidates:
            concept = concepts_by_id.get(concept_id)
            if concept is None:
                raise ContractError(f"asset alignment references unknown concept: {concept_id}")
            if concept.kind != asset.kind:
                type_mismatch = True
                continue
            valid_candidates.append(concept)
        name = asset.name or asset.asset_id
        if type_mismatch:
            issues.append(
                AssetAlignmentIssue(
                    code="ASSET_TYPE_MISMATCH",
                    asset_id=asset.asset_id,
                    asset_name=name,
                    message="素材候选与上传区域的类型不一致",
                    candidate_concept_ids=candidates,
                    suggestions=("检查人物/场景上传区域", "修改素材描述后重新对齐"),
                )
            )
        if not valid_candidates:
            issues.append(
                AssetAlignmentIssue(
                    code="ASSET_NO_MATCH",
                    asset_id=asset.asset_id,
                    asset_name=name,
                    message="idea 中没有找到可绑定的同类型对象",
                    suggestions=("在 idea 中补充该对象", "明确素材描述", "删除这份素材"),
                )
            )
            continue
        if len(valid_candidates) > 1 or match.confidence < ALIGNMENT_AUTO_CONFIDENCE:
            issues.append(
                AssetAlignmentIssue(
                    code="ASSET_AMBIGUOUS",
                    asset_id=asset.asset_id,
                    asset_name=name,
                    message="存在多个候选或自动对齐置信度不足，需要人工确认",
                    candidate_concept_ids=tuple(item.concept_id for item in valid_candidates),
                    suggestions=("人工选择对应对象", "补充更明确的素材描述"),
                )
            )
        concept = valid_candidates[0]
        previous = claimed.get(concept.concept_id)
        if previous is not None:
            issues.append(
                AssetAlignmentIssue(
                    code="ASSET_DUPLICATE_BINDING",
                    asset_id=asset.asset_id,
                    asset_name=name,
                    message=f"该故事对象已被素材 {previous} 占用",
                    candidate_concept_ids=(concept.concept_id,),
                    suggestions=("人工选择另一个对象", "删除重复素材"),
                )
            )
            continue
        claimed[concept.concept_id] = asset.asset_id
        kind_positions[asset.kind] += 1
        alias = (
            concept.required_entity_alias
            or f"provided_{asset.kind}_{kind_positions[asset.kind]:02d}"
        )
        suggested.append(
            ProvidedAssetBinding(
                asset_id=asset.asset_id,
                kind=asset.kind,
                canonical_alias=alias,
                semantic_role=concept.label,
                idea_mentions=tuple(dict.fromkeys(concept.idea_mentions)),
            )
        )

    fatal = any(item.code in {"ASSET_NO_MATCH", "ASSET_TYPE_MISMATCH"} for item in issues)
    status: Literal["auto_approved", "confirmation_required", "input_invalid"]
    if fatal:
        status = "input_invalid"
    elif mode == "manual" or issues:
        status = "confirmation_required"
    else:
        status = "auto_approved"
    return AssetAlignmentProposal(
        mode=mode,
        status=status,
        concepts=tuple(wire.concepts),
        matches=tuple(wire.matches),
        suggested_bindings=tuple(suggested),
        issues=tuple(issues),
        provider_call_ref=metrics.call_ref if metrics is not None else None,
    )


def render_asset_alignment_prompt(request: ProjectRequest) -> str:
    assets = "\n".join(
        f"- asset_id={asset.asset_id}; kind={asset.kind}; display_name={asset.name or asset.asset_id}; "
        f"description={asset.description}"
        for asset in request.provided_assets
        if asset.kind in {"character", "scene"}
    )
    return "\n".join(
        (
            "Extract only character and scene concepts that already exist in the story idea, then align each listed provided asset.",
            "Display names are weak labels and may not match the idea. Use kind and description as primary evidence.",
            "If the idea contains a structured entity binding with an ASCII alias and an asset clearly matches it, preserve that exact alias in the concept label or idea_mentions.",
            "For a character concept that is the same object as a required structured entity, set required_entity_alias to that exact alias; otherwise set it to an empty string. Scene concepts must use an empty string.",
            f"Required structured entity aliases: {', '.join(request.generation_requirements.required_entities) or 'none'}.",
            "Do not invent a character or scene merely to consume an asset.",
            "For every asset return one match row. candidate_concept_ids is empty when no same-kind concept exists, one item when clear, or multiple ordered items when ambiguous.",
            "Concept IDs must be short stable ASCII identifiers unique within this response.",
            f"Idea: {request.idea}",
            "Provided assets:",
            assets,
        )
    )
