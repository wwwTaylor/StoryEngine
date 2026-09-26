"""Shared strict and frozen domain primitives."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict


class FrozenModel(BaseModel):
    """Pydantic model with strict ownership-friendly defaults."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        validate_assignment=True,
        use_enum_values=False,
        hide_input_in_errors=True,
    )


class WireModel(BaseModel):
    """Strict model for data that crosses an LLM or file boundary."""

    model_config = ConfigDict(extra="forbid", hide_input_in_errors=True)
