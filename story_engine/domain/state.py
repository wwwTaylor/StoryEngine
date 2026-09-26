"""Canonical world-state algebra and the sole state reducer."""

from __future__ import annotations

from enum import StrEnum
from typing import Annotated, Literal

from pydantic import Field, model_validator

from story_engine.domain.common import FrozenModel
from story_engine.errors import WorldStateError

type Scalar = bool | int | float | str


class EntityKind(StrEnum):
    CHARACTER = "character"
    PROP = "prop"
    CONTAINER = "container"
    SURFACE = "surface"
    LANDMARK = "landmark"


class AttributeValueType(StrEnum):
    BOOL = "bool"
    NUMBER = "number"
    SHORT_TEXT = "short_text"
    SYMBOL = "symbol"


class AttributeDefinition(FrozenModel):
    key: str = Field(min_length=1, max_length=80, pattern=r"^[A-Za-z][A-Za-z0-9_.-]*$")
    value_type: AttributeValueType
    allowed_values: tuple[Scalar, ...] = ()
    is_visual: bool = False

    @model_validator(mode="after")
    def validate_allowed_values(self) -> AttributeDefinition:
        for value in self.allowed_values:
            _validate_attribute_value(self.value_type, value)
        if len({repr(value) for value in self.allowed_values}) != len(self.allowed_values):
            raise ValueError("allowed attribute values must be unique")
        return self


class PlacementBase(FrozenModel):
    kind: str


class InSceneZone(PlacementBase):
    kind: Literal["in_scene_zone"] = "in_scene_zone"
    scene_key: str
    zone_key: str


class OnSurface(PlacementBase):
    kind: Literal["on_surface"] = "on_surface"
    surface_entity: str


class InContainer(PlacementBase):
    kind: Literal["in_container"] = "in_container"
    container_entity: str


class HeldBy(PlacementBase):
    kind: Literal["held_by"] = "held_by"
    character_entity: str


class AttachedTo(PlacementBase):
    kind: Literal["attached_to"] = "attached_to"
    entity: str


class Offscreen(PlacementBase):
    kind: Literal["offscreen"] = "offscreen"
    scene_key: str | None = None


type Placement = Annotated[
    InSceneZone | OnSurface | InContainer | HeldBy | AttachedTo | Offscreen,
    Field(discriminator="kind"),
]


class PlacementFact(FrozenModel):
    entity_key: str
    placement: Placement


class AttributeFact(FrozenModel):
    entity_key: str
    attribute_key: str
    value: Scalar


class WorldState(FrozenModel):
    placement_by_entity: tuple[PlacementFact, ...]
    attributes_by_entity: tuple[AttributeFact, ...] = ()

    @model_validator(mode="after")
    def validate_uniqueness_and_order(self) -> WorldState:
        placement_keys = [fact.entity_key for fact in self.placement_by_entity]
        if placement_keys != sorted(placement_keys):
            raise ValueError("placement facts must be sorted by entity_key")
        if len(placement_keys) != len(set(placement_keys)):
            raise ValueError("each entity must have exactly one placement")
        attribute_keys = [
            (fact.entity_key, fact.attribute_key) for fact in self.attributes_by_entity
        ]
        if attribute_keys != sorted(attribute_keys):
            raise ValueError("attribute facts must be sorted")
        if len(attribute_keys) != len(set(attribute_keys)):
            raise ValueError("attribute facts must be unique")
        return self

    def placement_for(self, entity_key: str) -> Placement:
        for fact in self.placement_by_entity:
            if fact.entity_key == entity_key:
                return fact.placement
        raise KeyError(entity_key)

    def attribute_for(self, entity_key: str, attribute_key: str) -> Scalar:
        for fact in self.attributes_by_entity:
            if fact.entity_key == entity_key and fact.attribute_key == attribute_key:
                return fact.value
        raise KeyError((entity_key, attribute_key))

    @classmethod
    def build(
        cls,
        placements: dict[str, Placement],
        attributes: dict[tuple[str, str], Scalar] | None = None,
    ) -> WorldState:
        return cls(
            placement_by_entity=tuple(
                PlacementFact(entity_key=key, placement=value)
                for key, value in sorted(placements.items())
            ),
            attributes_by_entity=tuple(
                AttributeFact(entity_key=key[0], attribute_key=key[1], value=value)
                for key, value in sorted((attributes or {}).items())
            ),
        )


class TransitionBase(FrozenModel):
    kind: str


class SetPlacement(TransitionBase):
    kind: Literal["set_placement"] = "set_placement"
    entity_key: str
    placement: Placement


class SetAttribute(TransitionBase):
    kind: Literal["set_attribute"] = "set_attribute"
    entity_key: str
    attribute_key: str
    value: Scalar


type Transition = Annotated[SetPlacement | SetAttribute, Field(discriminator="kind")]


class EntityRule(FrozenModel):
    entity_key: str
    kind: EntityKind
    attribute_definitions: tuple[AttributeDefinition, ...] = ()


class SceneRule(FrozenModel):
    scene_key: str
    zone_keys: tuple[str, ...]


class WorldRules(FrozenModel):
    entities: tuple[EntityRule, ...]
    scenes: tuple[SceneRule, ...]

    @model_validator(mode="after")
    def validate_unique_keys(self) -> WorldRules:
        entity_keys = [entry.entity_key for entry in self.entities]
        scene_keys = [entry.scene_key for entry in self.scenes]
        if len(entity_keys) != len(set(entity_keys)):
            raise ValueError("entity rules must have unique keys")
        if len(scene_keys) != len(set(scene_keys)):
            raise ValueError("scene rules must have unique keys")
        for scene in self.scenes:
            if len(scene.zone_keys) != len(set(scene.zone_keys)):
                raise ValueError(f"scene {scene.scene_key} has duplicate zones")
        return self


class ShotBoundary(FrozenModel):
    shot_key: str
    planned_start: WorldState
    planned_end: WorldState


class WorldReducer:
    """The only component allowed to apply planned world transitions."""

    def __init__(self, rules: WorldRules) -> None:
        self.rules = rules
        self._entities = {entry.entity_key: entry for entry in rules.entities}
        self._scenes = {entry.scene_key: set(entry.zone_keys) for entry in rules.scenes}
        self._attributes = {
            (entry.entity_key, definition.key): definition
            for entry in rules.entities
            for definition in entry.attribute_definitions
        }

    def validate_world(self, state: WorldState) -> None:
        known_entities = set(self._entities)
        placed_entities = {fact.entity_key for fact in state.placement_by_entity}
        if placed_entities != known_entities:
            missing = sorted(known_entities - placed_entities)
            extra = sorted(placed_entities - known_entities)
            raise WorldStateError(f"placement coverage mismatch missing={missing} extra={extra}")

        for placement_fact in state.placement_by_entity:
            self._validate_placement(placement_fact.entity_key, placement_fact.placement)
        for attribute_fact in state.attributes_by_entity:
            definition = self._attributes.get(
                (attribute_fact.entity_key, attribute_fact.attribute_key)
            )
            if definition is None:
                raise WorldStateError(
                    f"undefined attribute "
                    f"{attribute_fact.entity_key}.{attribute_fact.attribute_key}"
                )
            _validate_attribute_value(definition.value_type, attribute_fact.value)
            if definition.allowed_values and attribute_fact.value not in definition.allowed_values:
                raise WorldStateError(
                    f"value {attribute_fact.value!r} is outside domain for "
                    f"{attribute_fact.entity_key}.{attribute_fact.attribute_key}"
                )
        self._validate_container_cycles(state)

    def apply(self, state: WorldState, transition: Transition) -> WorldState:
        placements = {fact.entity_key: fact.placement for fact in state.placement_by_entity}
        attributes = {
            (fact.entity_key, fact.attribute_key): fact.value for fact in state.attributes_by_entity
        }
        if isinstance(transition, SetPlacement):
            if transition.entity_key not in self._entities:
                raise WorldStateError(f"unknown transition entity {transition.entity_key}")
            placements[transition.entity_key] = transition.placement
        elif isinstance(transition, SetAttribute):
            if transition.entity_key not in self._entities:
                raise WorldStateError(f"unknown transition entity {transition.entity_key}")
            attributes[(transition.entity_key, transition.attribute_key)] = transition.value
        else:  # pragma: no cover - sealed by the type union
            raise AssertionError(f"unsupported transition {transition}")
        result = WorldState.build(placements, attributes)
        self.validate_world(result)
        return result

    def boundaries(
        self,
        initial_world: WorldState,
        shots: tuple[tuple[str, tuple[Transition, ...]], ...],
    ) -> tuple[ShotBoundary, ...]:
        self.validate_world(initial_world)
        state = initial_world
        result: list[ShotBoundary] = []
        for shot_key, transitions in shots:
            start = state
            for transition in transitions:
                state = self.apply(state, transition)
            result.append(ShotBoundary(shot_key=shot_key, planned_start=start, planned_end=state))
        return tuple(result)

    def _validate_placement(self, owner: str, placement: Placement) -> None:
        if isinstance(placement, InSceneZone):
            zones = self._scenes.get(placement.scene_key)
            if zones is None or placement.zone_key not in zones:
                raise WorldStateError(
                    f"unknown scene zone {placement.scene_key}/{placement.zone_key}"
                )
            return
        if isinstance(placement, Offscreen):
            if placement.scene_key is not None and placement.scene_key not in self._scenes:
                raise WorldStateError(f"unknown offscreen scene {placement.scene_key}")
            return

        if isinstance(placement, OnSurface):
            target = placement.surface_entity
            required_kind = EntityKind.SURFACE
        elif isinstance(placement, InContainer):
            target = placement.container_entity
            required_kind = EntityKind.CONTAINER
        elif isinstance(placement, HeldBy):
            target = placement.character_entity
            required_kind = EntityKind.CHARACTER
        elif isinstance(placement, AttachedTo):
            target = placement.entity
            required_kind = None
        else:  # pragma: no cover - sealed by the type union
            raise AssertionError(f"unsupported placement {placement}")

        if target == owner:
            raise WorldStateError(f"entity {owner} cannot be placed relative to itself")
        target_rule = self._entities.get(target)
        if target_rule is None:
            raise WorldStateError(f"placement target does not exist: {target}")
        if required_kind is not None and target_rule.kind != required_kind:
            raise WorldStateError(
                f"placement target {target} must be {required_kind.value}, "
                f"got {target_rule.kind.value}"
            )

    @staticmethod
    def _validate_container_cycles(state: WorldState) -> None:
        parent = {
            fact.entity_key: fact.placement.container_entity
            for fact in state.placement_by_entity
            if isinstance(fact.placement, InContainer)
        }
        for origin in parent:
            seen: set[str] = set()
            current = origin
            while current in parent:
                if current in seen:
                    chain = " -> ".join((*sorted(seen), current))
                    raise WorldStateError(f"container cycle detected: {chain}")
                seen.add(current)
                current = parent[current]


def _validate_attribute_value(value_type: AttributeValueType, value: Scalar) -> None:
    valid = False
    if value_type == AttributeValueType.BOOL:
        valid = isinstance(value, bool)
    elif value_type == AttributeValueType.NUMBER:
        valid = isinstance(value, (int, float)) and not isinstance(value, bool)
    elif value_type in {
        AttributeValueType.SHORT_TEXT,
        AttributeValueType.SYMBOL,
    } and isinstance(value, str):
        valid = 0 < len(value) <= 200
        if value_type == AttributeValueType.SYMBOL and valid:
            valid = value.replace("_", "").replace("-", "").isalnum()
    if not valid:
        raise WorldStateError(f"{value!r} is not a valid {value_type.value} value")
