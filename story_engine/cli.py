"""Command-line entry point."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import tempfile
from collections.abc import Sequence
from pathlib import Path

from story_engine.errors import StoryEngineError
from story_engine.run_spec import load_run_spec
from story_engine.version import __version__
from story_engine.security import redact_text


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="story-engine")
    parser.add_argument("--version", action="version", version=__version__)
    commands = parser.add_subparsers(dest="command", required=True)

    validate = commands.add_parser("validate", help="validate a complete run definition")
    validate.add_argument("definition", type=Path)

    doctor = commands.add_parser("doctor", help="report local runtime prerequisites")
    doctor.add_argument("--json", action="store_true")

    align = commands.add_parser("align-assets", help="align provided assets with idea concepts")
    align.add_argument("definition", type=Path)
    align.add_argument("--mode", choices=("auto", "manual"), default="auto")

    run = commands.add_parser("run", help="run a project")
    run.add_argument("definition", type=Path)
    run.add_argument("--run-id")

    resume = commands.add_parser("resume", help="resume a run directory")
    resume.add_argument("run_dir", type=Path)

    return parser


def _doctor() -> dict[str, object]:
    import shutil

    return {
        "python": sys.version.split()[0],
        "ffmpeg": shutil.which("ffmpeg") or False,
        "ffprobe": shutil.which("ffprobe") or False,
        "storyEngine": __version__,
        "features": {"guidedProvidedAssets": True, "assetAlignment": True},
    }


async def _align_assets(definition: Path, mode: str) -> dict[str, object]:
    from story_engine.planning.asset_alignment import align_project_assets
    from story_engine.providers.registry import build_provider_set
    from story_engine.storage import ArtifactStore

    spec = load_run_spec(definition)
    with tempfile.TemporaryDirectory(prefix="story-engine-align-") as directory:
        providers = build_provider_set(spec.config, ArtifactStore(Path(directory) / "artifacts"))
        try:
            await providers.planner.preflight()
            proposal = await align_project_assets(  # type: ignore[arg-type]
                spec.request,
                providers.planner,
                mode=mode,
            )
            return proposal.model_dump(mode="json")
        finally:
            await providers.aclose()


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "validate":
            request = load_run_spec(args.definition).request
            print(json.dumps({"task_id": request.task_id, "request_hash": request.request_hash}))
            return 0
        if args.command == "doctor":
            report = _doctor()
            if args.json:
                print(json.dumps(report, sort_keys=True))
            else:
                for key, value in report.items():
                    print(f"{key}: {value}")
            return 0 if report["ffmpeg"] and report["ffprobe"] else 1
        if args.command == "align-assets":
            print(json.dumps(asyncio.run(_align_assets(args.definition, args.mode)), ensure_ascii=False))
            return 0
        if args.command in {"run", "resume"}:
            # Imported lazily so validation/doctor remain side-effect free.
            from story_engine.bootstrap import run_command

            return run_command(args)
    except StoryEngineError as exc:
        print(f"error: {redact_text(str(exc))}", file=sys.stderr)
        return 2
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
