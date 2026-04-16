"""Profile-aware composition preview, generation, and validation."""

from __future__ import annotations

import ast
import json
import math
import os
import re
from pathlib import Path
from typing import Any, Optional

from data_generator_pipeline.loaders import get_loader
from data_generator_pipeline.prompts.profile_templates import (
    ProfilePromptTemplateManager,
)
from data_generator_pipeline.services.source_pack_service import (
    SourcePack,
    SourcePackService,
)
from shared.async_utils import run_sync
from shared.capability_registry import get_profile, list_profiles, resolve_profile
from shared.composition_manifest import CompositionManifest
from shared.composition_models import (
    CapabilityTarget,
    TrainerObjective,
)
from shared.schema_adapter_models import SchemaAdapter
from shared.schema_adapter_registry import (
    default_profile_schema_adapter_name,
    list_schema_adapters,
    objective_schema_kind,
    register_schema_adapter,
    resolve_schema_adapter,
)
from shared.text_messages import normalize_text_message_content
from shared.workspace_paths import resolve_workspace_path


class CompositionService:
    """Planning and generation surface for profiled dataset composition."""

    _VALID_MODES: set[str] = {"general", "coding", "agent"}
    _VALID_OBJECTIVES: set[str] = {"sft", "dpo", "grpo", "kto", "vlm_sft"}
    _VALID_SCHEMA_KINDS: set[str] = {
        "text_sft",
        "preference_pair",
        "reward_group",
        "binary_label",
    }
    _SUPPORTED_GENERATION_OBJECTIVES: set[str] = {"sft", "dpo", "grpo", "kto"}
    _OBJECTIVE_TO_GENERATION_TECHNIQUE: dict[str, str] = {
        "sft": "sft",
        "dpo": "dpo",
        "grpo": "grpo",
        "kto": "dpo",
    }
    _PROFILE_COMPOSE_CAPABILITIES: dict[str, set[str]] = {
        "general": {
            "instruction_following",
            "grounded_qa",
            "grounded_synthesis",
            "reasoning",
            "multi_hop",
            "unanswerable_or_refusal",
        },
        "coding": {
            "repo_qa",
            "bug_localization",
            "patch_generation",
            "test_debug",
            "code_review",
            "tool_use_planning",
            "instruction_following",
        },
        "agent": {
            "tool_selection",
            "argument_fidelity",
            "tool_result_grounding",
            "multi_step_state",
            "recovery",
            "stop_or_no_tool",
        },
    }
    _LOADER_SUFFIXES: set[str] = {".md", ".txt", ".json", ".jsonl"}
    _CODE_SUFFIXES: set[str] = {
        ".c",
        ".cc",
        ".cpp",
        ".cs",
        ".css",
        ".go",
        ".h",
        ".hpp",
        ".html",
        ".java",
        ".js",
        ".jsx",
        ".kt",
        ".kts",
        ".php",
        ".ps1",
        ".py",
        ".rb",
        ".rs",
        ".scala",
        ".scss",
        ".sh",
        ".sql",
        ".swift",
        ".ts",
        ".tsx",
    }
    _TEXT_SUFFIXES: set[str] = _LOADER_SUFFIXES | _CODE_SUFFIXES | {
        ".cfg",
        ".env",
        ".ini",
        ".toml",
        ".xml",
        ".yaml",
        ".yml",
    }
    _TEXT_FILENAMES: set[str] = {
        "dockerfile",
        "makefile",
        "cmakelists.txt",
        "justfile",
        "jenkinsfile",
    }
    _IGNORED_DIRS: set[str] = {
        ".cache",
        ".git",
        ".venv",
        "__pycache__",
        "data",
        "dist",
        "logs",
        "node_modules",
        "output",
    }
    _MAX_DIRECTORY_FILES = 200
    _TARGET_CHARS_PER_CHUNK = 4000
    _MAX_RAW_TEXT_CHARS = 500_000

    def __init__(
        self,
        *,
        generator_service: Any = None,
        dataset_service: Any = None,
    ) -> None:
        self._generator_service = generator_service
        self._dataset_service = dataset_service
        self._source_pack_service = SourcePackService()
        self._profile_prompt_manager = ProfilePromptTemplateManager()

    async def list_profiles(self, mode: Optional[str] = None) -> dict[str, Any]:
        if mode is not None and mode not in self._VALID_MODES:
            return {"success": False, "error": f"Unsupported mode: {mode}"}

        profiles = list_profiles(mode=mode)
        return {
            "success": True,
            "count": len(profiles),
            "profiles": profiles,
        }

    async def get_profile(self, profile_name: str) -> dict[str, Any]:
        profile = get_profile(profile_name)
        if profile is None:
            return {"success": False, "error": f"Unknown composition profile: {profile_name}"}

        return {
            "success": True,
            "profile_name": profile_name,
            "profile": profile,
        }

    async def list_schema_adapters(
        self,
        canonical_kind: Optional[str] = None,
    ) -> dict[str, Any]:
        if canonical_kind is not None and canonical_kind not in self._VALID_SCHEMA_KINDS:
            return {
                "success": False,
                "error": f"Unsupported canonical schema kind: {canonical_kind}",
            }

        adapters = list_schema_adapters(canonical_kind=canonical_kind)
        return {
            "success": True,
            "count": len(adapters),
            "schema_adapters": adapters,
        }

    async def register_schema_adapter(
        self,
        *,
        name: str,
        canonical_kind: str,
        field_map: dict[str, str],
        defaults: Optional[dict[str, Any]] = None,
        strict: bool = True,
        description: str = "",
    ) -> dict[str, Any]:
        if canonical_kind not in self._VALID_SCHEMA_KINDS:
            return {
                "success": False,
                "error": f"Unsupported canonical schema kind: {canonical_kind}",
            }

        try:
            adapter = SchemaAdapter(
                name=name,
                canonical_kind=canonical_kind,  # type: ignore[arg-type]
                description=description,
                field_map=field_map,
                defaults=defaults or {},
                strict=strict,
            )
            stored = register_schema_adapter(adapter)
        except Exception as exc:
            return {"success": False, "error": str(exc)}

        return {
            "success": True,
            "schema_adapter": stored.model_dump(),
        }

    async def preview_composition(
        self,
        *,
        profile_name: str,
        source_paths: list[str],
        row_target: int = 200,
        objective: Optional[str] = None,
        capability_overrides: Optional[dict[str, int]] = None,
        schema_adapter_name: Optional[str] = None,
    ) -> dict[str, Any]:
        return await run_sync(
            self._preview_composition_sync,
            profile_name=profile_name,
            source_paths=source_paths,
            row_target=row_target,
            objective=objective,
            capability_overrides=capability_overrides,
            schema_adapter_name=schema_adapter_name,
        )

    async def compose_profiled_dataset(
        self,
        *,
        profile_name: str,
        source_paths: list[str],
        output_path: str,
        row_target: int = 200,
        objective: Optional[str] = None,
        capability_overrides: Optional[dict[str, int]] = None,
        schema_adapter_name: Optional[str] = None,
        format: str = "jsonl",
    ) -> dict[str, Any]:
        if self._generator_service is None:
            return {
                "success": False,
                "error": "Composition generation requires a configured generator service.",
            }
        if self._dataset_service is None:
            return {
                "success": False,
                "error": "Composition generation requires a configured dataset service.",
            }

        preview = self._preview_composition_sync(
            profile_name=profile_name,
            source_paths=source_paths,
            row_target=row_target,
            objective=objective,
            capability_overrides=capability_overrides,
            schema_adapter_name=schema_adapter_name,
        )
        if not preview.get("success"):
            return preview

        if preview["mode"] not in self._PROFILE_COMPOSE_CAPABILITIES:
            return {
                "success": False,
                "error": (
                    "Profiled dataset generation is currently implemented only for "
                    "general, coding, and agent profiles."
                ),
            }
        if preview["objective"] not in self._SUPPORTED_GENERATION_OBJECTIVES:
            return {
                "success": False,
                "error": (
                    "Profiled dataset generation currently supports only "
                    "objective='sft', objective='dpo', objective='grpo', "
                    "or objective='kto'."
                ),
            }

        pack_bundle = await run_sync(
            self._source_pack_service.build_packs,
            source_paths,
            include_multi_hop=True,
        )
        single_packs: list[SourcePack] = pack_bundle["single_packs"]
        multi_hop_packs: list[SourcePack] = pack_bundle["multi_hop_packs"]
        if not single_packs:
            return {
                "success": False,
                "error": "No usable source chunks were found for dataset composition.",
                "source_summaries": pack_bundle["source_summaries"],
            }

        generation_objective = self._generation_objective(preview["objective"])
        warnings = self._dedupe_warnings(preview["warnings"] + pack_bundle["warnings"])
        generation_errors: list[dict[str, Any]] = []
        canonical_rows: list[dict[str, Any]] = []
        supported_capabilities = self._PROFILE_COMPOSE_CAPABILITIES[preview["mode"]]

        for capability, planned_count in preview["row_plan"].items():
            if planned_count <= 0:
                continue
            if capability not in supported_capabilities:
                warnings.append(
                    f"Skipped unsupported capability for generation: {capability}"
                )
                continue

            pack_pool = self._pack_pool_for_capability(
                mode=preview["mode"],
                capability=capability,
                single_packs=single_packs,
                multi_hop_packs=multi_hop_packs,
                warnings=warnings,
            )
            if not pack_pool:
                warnings.append(f"No source packs available for capability: {capability}")
                continue

            template = self._profile_prompt_manager.load(
                preview["mode"],
                capability,
                generation_objective,
            )
            produced_for_capability = 0
            attempts = 0
            max_attempts = max(planned_count, len(pack_pool)) * 3

            while produced_for_capability < planned_count and attempts < max_attempts:
                pack = pack_pool[attempts % len(pack_pool)]
                attempts += 1

                result = await self._generator_service.generate_from_page(
                    technique=generation_objective,
                    page_text=pack.text,
                    page_index=pack.page_index,
                    file_name=pack.file_name,
                    custom_template=template,
                )
                if not result.get("success"):
                    generation_errors.append(
                        {
                            "capability": capability,
                            "pack_id": pack.pack_id,
                            "error": str(result.get("error") or "generation failed"),
                        }
                    )
                    continue

                data_points = result.get("data_points") or []
                if not data_points:
                    continue

                remaining = planned_count - produced_for_capability
                for data_point in data_points[:remaining]:
                    canonical_rows_for_point = self._canonicalize_rows(
                        preview["objective"],
                        data_point,
                        capability=capability,
                        profile_name=preview["profile_name"],
                        pack=pack,
                        remaining=remaining,
                    )
                    if not canonical_rows_for_point:
                        continue
                    canonical_rows.extend(canonical_rows_for_point)
                    produced_for_capability += len(canonical_rows_for_point)
                    if produced_for_capability >= planned_count:
                        break

            if produced_for_capability < planned_count:
                warnings.append(
                    f"Capability '{capability}' produced {produced_for_capability} of "
                    f"{planned_count} planned rows."
                )

        if not canonical_rows:
            return {
                "success": False,
                "error": "Profiled generation completed without any usable rows.",
                "warnings": warnings,
                "generation_errors": generation_errors,
            }

        adapted_rows = [
            self._apply_schema_adapter(row, preview["schema_adapter"])
            for row in canonical_rows
        ]
        save_result = await self._dataset_service.save(
            adapted_rows,
            output_path,
            format,
        )
        if not save_result.get("success"):
            return save_result

        achieved_mix = self._compute_mix_from_rows(adapted_rows)
        manifest = CompositionManifest(
            profile_name=preview["profile_name"],
            mode=preview["mode"],
            objective=preview["objective"],
            schema_adapter_name=preview["schema_adapter"]["name"],
            canonical_kind=preview["schema_adapter"]["expected_canonical_kind"],
            dataset_path=save_result["file_path"],
            dataset_format=save_result["format"],
            row_target=preview["row_target"],
            row_count=save_result["row_count"],
            requested_mix=preview["requested_mix"],
            resolved_mix=preview["resolved_mix"],
            achieved_mix=achieved_mix,
            row_plan=preview["row_plan"],
            source_paths=list(source_paths),
            source_totals=pack_bundle["source_totals"],
            source_summaries=pack_bundle["source_summaries"],
            warnings=self._dedupe_warnings(warnings),
        )
        manifest_path = self._manifest_path_for_dataset(save_result["file_path"])
        await run_sync(
            self._write_manifest_sync,
            manifest_path,
            manifest.model_dump(),
        )

        return {
            "success": True,
            "profile_name": preview["profile_name"],
            "mode": preview["mode"],
            "objective": preview["objective"],
            "row_target": preview["row_target"],
            "row_count": save_result["row_count"],
            "requested_mix": preview["requested_mix"],
            "resolved_mix": preview["resolved_mix"],
            "achieved_mix": achieved_mix,
            "row_plan": preview["row_plan"],
            "dataset": save_result,
            "manifest_path": str(manifest_path),
            "warnings": manifest.warnings,
            "generation_errors": generation_errors,
        }

    async def validate_composition(
        self,
        *,
        dataset_path: str,
        manifest_path: Optional[str] = None,
    ) -> dict[str, Any]:
        if self._dataset_service is None:
            return {
                "success": False,
                "error": "Composition validation requires a configured dataset service.",
            }

        info_result = await self._dataset_service.info(dataset_path)
        if not info_result.get("success"):
            return info_result

        load_result = await self._dataset_service.load(dataset_path)
        if not load_result.get("success"):
            return load_result

        resolved_manifest_path = (
            resolve_workspace_path(manifest_path)
            if manifest_path
            else self._manifest_path_for_dataset(info_result["metadata"]["file_path"])
        )
        if not resolved_manifest_path.exists():
            return {
                "success": False,
                "error": f"Composition manifest not found: {resolved_manifest_path}",
            }

        try:
            raw_manifest = await run_sync(self._load_manifest_sync, resolved_manifest_path)
            manifest = CompositionManifest(**raw_manifest)
        except Exception as exc:
            return {
                "success": False,
                "error": f"Invalid composition manifest: {exc}",
                "manifest_path": str(resolved_manifest_path),
            }

        rows = load_result.get("data_points", [])
        columns = info_result["metadata"].get("columns", [])
        col_set = set(columns)
        warnings: list[str] = []
        errors: list[str] = []

        if info_result["metadata"].get("row_count") != manifest.row_count:
            errors.append(
                "Dataset row_count does not match the persisted composition manifest."
            )

        adapter = resolve_schema_adapter(manifest.schema_adapter_name)
        expected_columns: list[str] = []
        if adapter is None:
            warnings.append(
                f"Manifest references unknown schema adapter: {manifest.schema_adapter_name}"
            )
        else:
            expected_columns = sorted(set(adapter.field_map.values()))
            missing_columns = sorted(set(expected_columns) - col_set)
            if missing_columns:
                errors.append(
                    "Dataset is missing schema-adapter columns: "
                    + ", ".join(missing_columns)
                )
            if adapter.canonical_kind != manifest.canonical_kind:
                errors.append(
                    "Manifest canonical_kind does not match the registered schema adapter."
                )

        capability_counts = self._compute_capability_counts(rows)
        achieved_mix = self._normalize_percentages(capability_counts)
        if "composition_capability" not in col_set:
            warnings.append("Dataset rows do not expose a composition_capability column.")
        elif not capability_counts:
            warnings.append("Dataset did not contain any composition capability labels.")

        if "source_refs" not in col_set:
            warnings.append("Dataset rows do not expose source_refs metadata.")

        if capability_counts:
            for capability, expected_percent in manifest.achieved_mix.items():
                actual_percent = achieved_mix.get(capability, 0)
                if abs(actual_percent - expected_percent) > 5:
                    warnings.append(
                        f"Capability mix drift for '{capability}': "
                        f"manifest={expected_percent}, dataset={actual_percent}."
                    )

        rows_with_refs = sum(1 for row in rows if row.get("source_refs"))
        source_ref_coverage = round(rows_with_refs / len(rows), 3) if rows else 0.0
        status = "fail" if errors else ("warn" if warnings else "pass")
        return {
            "success": not errors,
            "status": status,
            "dataset_path": info_result["metadata"]["file_path"],
            "manifest_path": str(resolved_manifest_path),
            "profile_name": manifest.profile_name,
            "mode": manifest.mode,
            "objective": manifest.objective,
            "row_count": info_result["metadata"]["row_count"],
            "expected_columns": expected_columns,
            "columns": columns,
            "requested_mix": manifest.requested_mix,
            "resolved_mix": manifest.resolved_mix,
            "manifest_achieved_mix": manifest.achieved_mix,
            "dataset_achieved_mix": achieved_mix,
            "capability_counts": capability_counts,
            "source_ref_coverage": source_ref_coverage,
            "warnings": warnings,
            "errors": errors,
        }

    def _preview_composition_sync(
        self,
        *,
        profile_name: str,
        source_paths: list[str],
        row_target: int = 200,
        objective: Optional[str] = None,
        capability_overrides: Optional[dict[str, int]] = None,
        schema_adapter_name: Optional[str] = None,
    ) -> dict[str, Any]:
        profile = resolve_profile(profile_name)
        if profile is None:
            return {"success": False, "error": f"Unknown composition profile: {profile_name}"}

        if row_target <= 0:
            return {"success": False, "error": "row_target must be greater than 0."}

        if not source_paths:
            return {"success": False, "error": "Provide at least one source path."}

        resolved_objective = objective or profile.default_objective
        if resolved_objective not in self._VALID_OBJECTIVES:
            return {"success": False, "error": f"Unsupported objective: {resolved_objective}"}
        if resolved_objective not in profile.allowed_objectives:
            return {
                "success": False,
                "error": (
                    f"Objective '{resolved_objective}' is not allowed for profile "
                    f"'{profile_name}'."
                ),
            }

        warnings: list[str] = []
        try:
            targets, requested_mix, resolved_mix = self._resolve_capability_mix(
                profile.capability_targets,
                capability_overrides or {},
                warnings,
            )
        except ValueError as exc:
            return {"success": False, "error": str(exc)}

        if not resolved_mix:
            return {"success": False, "error": "No capability weights remain after overrides."}

        source_summaries = [self._inspect_source(path) for path in source_paths]
        valid_sources = sum(
            1 for summary in source_summaries if summary["estimated_chunks"] > 0
        )
        total_chunks = sum(summary["estimated_chunks"] for summary in source_summaries)
        total_chars = sum(summary["estimated_text_chars"] for summary in source_summaries)
        scanned_files = sum(summary["scanned_files"] for summary in source_summaries)
        code_files = sum(summary["code_files"] for summary in source_summaries)

        if valid_sources == 0:
            return {
                "success": False,
                "error": "None of the provided source paths could be inspected.",
                "source_summaries": source_summaries,
            }

        if resolved_mix.get("multi_hop", 0) > 0 and total_chunks < 2:
            warnings.append(
                "Multi-hop coverage is requested, but the inspected sources only yielded "
                "one chunk."
            )

        if profile.mode == "coding" and code_files == 0:
            warnings.append(
                "Coding profile selected, but no code-like files were detected in the "
                "provided sources."
            )

        if total_chunks > 0 and row_target > total_chunks * 8:
            warnings.append(
                "Requested row_target is dense relative to the estimated source chunks. "
                "Expect repeated or weaker supervision unless you add more source material."
            )
        if resolved_objective == "kto":
            warnings.append(
                "KTO composition currently derives binary-label rows from generated "
                "preference pairs."
            )
            if row_target % 2 != 0:
                warnings.append(
                    "Odd row_target requested for KTO. The final label balance may be uneven."
                )

        schema_result = self._resolve_schema_adapter(
            mode=profile.mode,
            objective=resolved_objective,  # type: ignore[arg-type]
            schema_adapter_name=schema_adapter_name,
        )
        if not schema_result["success"]:
            return schema_result

        row_plan = self._allocate_rows(targets, resolved_mix, row_target)

        return {
            "success": True,
            "profile_name": profile.name,
            "mode": profile.mode,
            "objective": resolved_objective,
            "allowed_objectives": profile.allowed_objectives,
            "row_target": row_target,
            "requested_mix": requested_mix,
            "resolved_mix": resolved_mix,
            "row_plan": row_plan,
            "schema_adapter": schema_result["schema_adapter"],
            "source_summaries": source_summaries,
            "source_totals": {
                "valid_sources": valid_sources,
                "estimated_chunks": total_chunks,
                "estimated_text_chars": total_chars,
                "scanned_files": scanned_files,
                "code_files": code_files,
            },
            "warnings": warnings,
        }

    def _resolve_capability_mix(
        self,
        targets: list[CapabilityTarget],
        overrides: dict[str, int],
        warnings: list[str],
    ) -> tuple[list[CapabilityTarget], dict[str, int], dict[str, int]]:
        working_targets = [target.model_copy(deep=True) for target in targets if target.enabled]
        by_capability = {target.capability: target for target in working_targets}

        for capability, override in overrides.items():
            if capability not in by_capability:
                warnings.append(f"Ignored unknown capability override: {capability}")
                continue
            if int(override) < 0:
                raise ValueError(f"Capability override must be non-negative: {capability}")
            by_capability[capability].weight_percent = int(override)

        requested_mix = {
            target.capability: target.weight_percent for target in working_targets
        }
        total_weight = sum(requested_mix.values())
        if total_weight <= 0:
            return working_targets, requested_mix, {}

        if total_weight != 100:
            warnings.append(
                f"Capability weights sum to {total_weight}; normalized to 100 for preview."
            )

        resolved_mix = self._normalize_percentages(requested_mix)
        for target in working_targets:
            target.weight_percent = resolved_mix.get(target.capability, 0)

        return working_targets, requested_mix, resolved_mix

    def _resolve_schema_adapter(
        self,
        *,
        mode: str,
        objective: TrainerObjective,
        schema_adapter_name: Optional[str],
    ) -> dict[str, Any]:
        expected_kind = objective_schema_kind(objective)
        adapter_name = schema_adapter_name or default_profile_schema_adapter_name(
            mode, objective
        )
        if not adapter_name:
            return {
                "success": False,
                "error": f"No default schema adapter is configured for objective: {objective}",
            }

        adapter = resolve_schema_adapter(adapter_name)
        if adapter is None:
            return {
                "success": False,
                "error": f"Unknown schema adapter: {adapter_name}",
            }

        if adapter.canonical_kind != expected_kind:
            return {
                "success": False,
                "error": (
                    f"Schema adapter '{adapter_name}' targets '{adapter.canonical_kind}', "
                    f"but objective '{objective}' expects '{expected_kind}'."
                ),
            }

        adapter_payload = adapter.model_dump()
        adapter_payload["is_default"] = schema_adapter_name is None
        adapter_payload["expected_canonical_kind"] = expected_kind
        return {"success": True, "schema_adapter": adapter_payload}

    def _pack_pool_for_capability(
        self,
        *,
        mode: str,
        capability: str,
        single_packs: list[SourcePack],
        multi_hop_packs: list[SourcePack],
        warnings: list[str],
    ) -> list[SourcePack]:
        if mode == "coding":
            code_single_packs = [pack for pack in single_packs if self._is_code_pack(pack)]
            code_multi_hop_packs = [
                pack for pack in multi_hop_packs if self._is_code_pack(pack)
            ]

            if capability == "tool_use_planning":
                if code_multi_hop_packs:
                    return code_multi_hop_packs
                if multi_hop_packs:
                    warnings.append(
                        "No code-focused multi-hop packs were available; using the broader multi-hop pool."
                    )
                    return multi_hop_packs
                if code_single_packs:
                    warnings.append(
                        "No multi-hop coding packs were available; fell back to single code packs."
                    )
                    return code_single_packs
                return single_packs

            if code_single_packs:
                return code_single_packs
            warnings.append(
                "Coding profile generation did not find code-like packs; using all available text chunks."
            )
            return single_packs

        if mode == "agent":
            if capability in {"multi_step_state", "recovery"}:
                if multi_hop_packs:
                    return multi_hop_packs
                warnings.append(
                    f"No multi-hop packs were available for agent capability '{capability}'; "
                    "fell back to single packs."
                )
                return single_packs

            if capability == "tool_result_grounding" and multi_hop_packs:
                return multi_hop_packs

            return single_packs

        if capability == "multi_hop":
            if multi_hop_packs:
                return multi_hop_packs
            warnings.append(
                "No multi-hop pack pairs were available; fell back to single packs."
            )
        return single_packs

    def _is_code_pack(self, pack: SourcePack) -> bool:
        for source_ref in pack.source_refs:
            file_path = str(source_ref.get("file_path") or "")
            file_name = str(source_ref.get("file_name") or "")
            path = Path(file_path or file_name)
            if self._is_code_path(path):
                return True
        return False

    def _is_code_path(self, path: Path) -> bool:
        suffix = path.suffix.lower()
        if suffix in self._CODE_SUFFIXES:
            return True
        return path.name.lower() in self._TEXT_FILENAMES

    def _generation_objective(self, objective: str) -> str:
        return self._OBJECTIVE_TO_GENERATION_TECHNIQUE.get(objective, objective)

    def _canonicalize_rows(
        self,
        objective: str,
        data_point: dict[str, Any],
        *,
        capability: str,
        profile_name: str,
        pack: SourcePack,
        remaining: int,
    ) -> list[dict[str, Any]]:
        if objective == "sft":
            row = self._canonicalize_sft_row(
                data_point,
                capability=capability,
                profile_name=profile_name,
                pack=pack,
            )
            return [row] if row is not None else []
        if objective == "dpo":
            row = self._canonicalize_dpo_row(
                data_point,
                capability=capability,
                profile_name=profile_name,
                pack=pack,
            )
            return [row] if row is not None else []
        if objective == "grpo":
            row = self._canonicalize_grpo_row(
                data_point,
                capability=capability,
                profile_name=profile_name,
                pack=pack,
            )
            return [row] if row is not None else []
        if objective == "kto":
            return self._canonicalize_kto_rows(
                data_point,
                capability=capability,
                profile_name=profile_name,
                pack=pack,
                remaining=remaining,
            )
        return []

    def _canonicalize_sft_row(
        self,
        data_point: dict[str, Any],
        *,
        capability: str,
        profile_name: str,
        pack: SourcePack,
    ) -> dict[str, Any] | None:
        instruction = str(
            data_point.get("instruction")
            or data_point.get("prompt")
            or ""
        ).strip()
        input_text = str(data_point.get("input") or "").strip()
        output = str(
            data_point.get("output")
            or data_point.get("response")
            or ""
        ).strip()
        if not instruction or not output:
            return None

        messages = self._build_structured_messages(
            instruction=instruction,
            input_text=input_text,
            output=output,
            capability=capability,
        )

        return {
            "instruction": instruction,
            "input": input_text,
            "output": output,
            "messages": messages,
            "id": int(data_point.get("id") or 0),
            "file_name": str(data_point.get("file_name") or pack.file_name),
            "page": int(data_point.get("page") or (pack.page_index + 1)),
            "text": str(data_point.get("text") or pack.text),
            "composition_capability": capability,
            "composition_profile": profile_name,
            "source_pack_id": pack.pack_id,
            "source_pack_kind": pack.kind,
            "source_refs": pack.source_refs,
        }

    def _canonicalize_dpo_row(
        self,
        data_point: dict[str, Any],
        *,
        capability: str,
        profile_name: str,
        pack: SourcePack,
    ) -> dict[str, Any] | None:
        prompt = str(data_point.get("prompt") or "").strip()
        chosen = str(data_point.get("chosen") or "").strip()
        rejected = str(data_point.get("rejected") or "").strip()
        if not prompt or not chosen or not rejected:
            return None

        return {
            "prompt": prompt,
            "chosen": chosen,
            "rejected": rejected,
            "id": int(data_point.get("id") or 0),
            "file_name": str(data_point.get("file_name") or pack.file_name),
            "page": int(data_point.get("page") or (pack.page_index + 1)),
            "text": str(data_point.get("text") or pack.text),
            "composition_capability": capability,
            "composition_profile": profile_name,
            "source_pack_id": pack.pack_id,
            "source_pack_kind": pack.kind,
            "source_refs": pack.source_refs,
        }

    def _canonicalize_grpo_row(
        self,
        data_point: dict[str, Any],
        *,
        capability: str,
        profile_name: str,
        pack: SourcePack,
    ) -> dict[str, Any] | None:
        prompt = str(data_point.get("prompt") or "").strip()
        responses = data_point.get("responses")
        rewards = data_point.get("rewards")
        if (
            not prompt
            or not isinstance(responses, list)
            or not responses
            or not isinstance(rewards, list)
            or not rewards
        ):
            return None

        normalized_responses = [str(response).strip() for response in responses if str(response).strip()]
        normalized_rewards: list[float] = []
        for reward in rewards[: len(normalized_responses)]:
            try:
                normalized_rewards.append(float(reward))
            except (TypeError, ValueError):
                normalized_rewards.append(0.0)

        if not normalized_responses or len(normalized_responses) != len(normalized_rewards):
            return None

        return {
            "prompt": prompt,
            "responses": normalized_responses,
            "rewards": normalized_rewards,
            "id": int(data_point.get("id") or 0),
            "file_name": str(data_point.get("file_name") or pack.file_name),
            "page": int(data_point.get("page") or (pack.page_index + 1)),
            "text": str(data_point.get("text") or pack.text),
            "composition_capability": capability,
            "composition_profile": profile_name,
            "source_pack_id": pack.pack_id,
            "source_pack_kind": pack.kind,
            "source_refs": pack.source_refs,
        }

    def _canonicalize_kto_rows(
        self,
        data_point: dict[str, Any],
        *,
        capability: str,
        profile_name: str,
        pack: SourcePack,
        remaining: int,
    ) -> list[dict[str, Any]]:
        prompt = str(data_point.get("prompt") or "").strip()
        chosen = str(data_point.get("chosen") or "").strip()
        rejected = str(data_point.get("rejected") or "").strip()
        if not prompt or (not chosen and not rejected) or remaining <= 0:
            return []

        rows: list[dict[str, Any]] = []
        candidates = [(chosen, True), (rejected, False)]
        if remaining == 1:
            preferred_positive = self._prefer_positive_kto_row(data_point, prompt)
            candidates = candidates if preferred_positive else list(reversed(candidates))
        for completion, label in candidates:
            if not completion or len(rows) >= remaining:
                continue
            rows.append(
                {
                    "prompt": prompt,
                    "completion": completion,
                    "label": label,
                    "id": int(data_point.get("id") or 0),
                    "file_name": str(data_point.get("file_name") or pack.file_name),
                    "page": int(data_point.get("page") or (pack.page_index + 1)),
                    "text": str(data_point.get("text") or pack.text),
                    "composition_capability": capability,
                    "composition_profile": profile_name,
                    "source_pack_id": pack.pack_id,
                    "source_pack_kind": pack.kind,
                    "source_refs": pack.source_refs,
                }
            )
        return rows

    @staticmethod
    def _prefer_positive_kto_row(
        data_point: dict[str, Any],
        prompt: str,
    ) -> bool:
        raw_id = data_point.get("id")
        try:
            return int(raw_id) % 2 == 1
        except (TypeError, ValueError):
            return sum(ord(char) for char in prompt) % 2 == 1

    def _apply_schema_adapter(
        self,
        row: dict[str, Any],
        schema_adapter: dict[str, Any],
    ) -> dict[str, Any]:
        base_fields = {
            "instruction": str(row.get("instruction") or "").strip(),
            "input": str(row.get("input") or "").strip(),
            "output": str(row.get("output") or "").strip(),
        }
        derived_fields = {
            **base_fields,
            "prompt": f"{base_fields['instruction']} {base_fields['input']}".strip(),
            "response": base_fields["output"],
            "messages": row.get("messages") or self._build_structured_messages(
                instruction=base_fields["instruction"],
                input_text=base_fields["input"],
                output=base_fields["output"],
                capability=str(row.get("composition_capability") or "").strip(),
            ),
        }

        adapter = SchemaAdapter(**{
            key: schema_adapter[key]
            for key in SchemaAdapter.model_fields
            if key in schema_adapter
        })
        adapted: dict[str, Any] = {}

        for source_field, target_field in adapter.field_map.items():
            if source_field in derived_fields:
                adapted[target_field] = derived_fields[source_field]
            elif source_field in row:
                adapted[target_field] = row[source_field]

        for key, value in adapter.defaults.items():
            adapted.setdefault(key, value)

        if not adapter.strict:
            for key in ("instruction", "input", "output"):
                adapted.setdefault(key, base_fields[key])

        for meta_key in (
            "id",
            "file_name",
            "page",
            "text",
            "composition_capability",
            "composition_profile",
            "source_pack_id",
            "source_pack_kind",
            "source_refs",
        ):
            if meta_key in row:
                adapted[meta_key] = row[meta_key]

        return adapted

    def _build_structured_messages(
        self,
        *,
        instruction: str,
        input_text: str,
        output: str,
        capability: str,
    ) -> list[dict[str, Any]]:
        if capability in self._PROFILE_COMPOSE_CAPABILITIES["agent"]:
            return self._build_agent_messages(
                instruction=instruction,
                input_text=input_text,
                output=output,
            )

        user_parts = [instruction.strip()]
        if input_text.strip():
            user_parts.append(f"Context:\n{input_text.strip()}")
        return [
            {
                "role": "user",
                "content": "\n\n".join(part for part in user_parts if part),
            },
            {
                "role": "assistant",
                "content": output.strip(),
            },
        ]

    def _build_agent_messages(
        self,
        *,
        instruction: str,
        input_text: str,
        output: str,
    ) -> list[dict[str, Any]]:
        user_parts = [instruction.strip()]
        if input_text.strip():
            user_parts.append(f"Grounding:\n{input_text.strip()}")
        messages: list[dict[str, Any]] = [
            {
                "role": "system",
                "content": (
                    "Use tools only when the provided grounding justifies the call. "
                    "Base the final answer on tool results."
                ),
            },
            {
                "role": "user",
                "content": "\n\n".join(part for part in user_parts if part),
            },
        ]

        parsed = self._parse_agent_output(output)
        if parsed["tool_calls"]:
            messages.append(
                {
                    "role": "assistant",
                    "content": parsed["assistant_preamble"],
                    "tool_calls": parsed["tool_calls"],
                }
            )
            for tool_call in parsed["tool_calls"]:
                messages.append(
                    {
                        "role": "tool",
                        "name": tool_call["function"]["name"],
                        "tool_call_id": tool_call["id"],
                        "content": input_text.strip() or "No grounded tool result provided.",
                    }
                )
            if parsed["final_answer"]:
                messages.append({"role": "assistant", "content": parsed["final_answer"]})
            return [message for message in messages if self._message_has_payload(message)]

        messages.append({"role": "assistant", "content": output.strip()})
        return [message for message in messages if self._message_has_payload(message)]

    @staticmethod
    def _message_has_payload(message: dict[str, Any]) -> bool:
        if str(message.get("role") or "").strip().lower() == "assistant" and message.get(
            "tool_calls"
        ):
            return True
        return bool(normalize_text_message_content(message.get("content")))

    def _parse_agent_output(self, output: str) -> dict[str, Any]:
        cleaned = str(output or "").strip()
        tool_calls = self._parse_xml_tool_calls(cleaned)
        if not tool_calls:
            tool_calls = self._parse_inline_tool_calls(cleaned)

        assistant_preamble = ""
        final_answer = cleaned
        final_match = re.search(
            r"final answer\s*:\s*(.+)$",
            cleaned,
            flags=re.IGNORECASE | re.DOTALL,
        )
        if final_match:
            final_answer = final_match.group(1).strip()
            assistant_preamble = cleaned[: final_match.start()].strip()
        elif tool_calls:
            stripped = re.sub(
                r"<tool_call>[\s\S]*?</tool_call>",
                "",
                cleaned,
                flags=re.IGNORECASE,
            )
            stripped = re.sub(
                r"(?im)^\s*step\s+\d+\s*:\s*[A-Za-z_][\w]*\s*\(\s*\{[\s\S]*?\}\s*\)\s*$",
                "",
                stripped,
            )
            final_answer = stripped.strip() or cleaned

        if tool_calls and not assistant_preamble:
            assistant_preamble = "Using the grounded tool flow."

        return {
            "assistant_preamble": assistant_preamble.strip(),
            "tool_calls": tool_calls,
            "final_answer": final_answer.strip(),
        }

    def _parse_xml_tool_calls(self, text: str) -> list[dict[str, Any]]:
        matches = re.findall(
            r"<tool_call>\s*(\{[\s\S]*?\})\s*</tool_call>",
            text,
            flags=re.IGNORECASE,
        )
        tool_calls: list[dict[str, Any]] = []
        for index, block in enumerate(matches, start=1):
            try:
                payload = json.loads(block)
            except json.JSONDecodeError:
                continue
            normalized = self._normalize_tool_call_payload(payload, index=index)
            if normalized is not None:
                tool_calls.append(normalized)
        return tool_calls

    def _parse_inline_tool_calls(self, text: str) -> list[dict[str, Any]]:
        pattern = re.compile(
            r"(?im)^\s*(?:step\s+\d+\s*:\s*)?(?P<name>[A-Za-z_][\w]*)\s*"
            r"\(\s*(?P<args>\{[\s\S]*?\})\s*\)\s*$"
        )
        tool_calls: list[dict[str, Any]] = []
        for index, match in enumerate(pattern.finditer(text), start=1):
            args_text = match.group("args")
            try:
                arguments = json.loads(args_text)
            except json.JSONDecodeError:
                try:
                    arguments = ast.literal_eval(args_text)
                except (ValueError, SyntaxError):
                    continue
            normalized = self._normalize_tool_call_payload(
                {"name": match.group("name"), "arguments": arguments},
                index=index,
            )
            if normalized is not None:
                tool_calls.append(normalized)
        return tool_calls

    @staticmethod
    def _normalize_tool_call_payload(
        payload: Any,
        *,
        index: int,
    ) -> dict[str, Any] | None:
        if not isinstance(payload, dict):
            return None

        function = payload.get("function")
        if isinstance(function, dict):
            name = str(function.get("name") or payload.get("name") or "").strip()
            arguments = function.get("arguments", {})
        else:
            name = str(payload.get("name") or "").strip()
            arguments = payload.get("arguments", {})

        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except json.JSONDecodeError:
                try:
                    arguments = ast.literal_eval(arguments)
                except (ValueError, SyntaxError):
                    arguments = {"_raw": arguments}

        if not isinstance(arguments, dict):
            arguments = {"_raw": arguments}
        if not name:
            return None

        return {
            "id": str(payload.get("id") or f"call_{index}"),
            "type": "function",
            "function": {
                "name": name,
                "arguments": json.dumps(
                    arguments,
                    ensure_ascii=False,
                    sort_keys=True,
                    default=str,
                ),
            },
        }

    @staticmethod
    def _compute_capability_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
        counts: dict[str, int] = {}
        for row in rows:
            capability = str(row.get("composition_capability") or "").strip()
            if not capability:
                continue
            counts[capability] = counts.get(capability, 0) + 1
        return counts

    def _compute_mix_from_rows(self, rows: list[dict[str, Any]]) -> dict[str, int]:
        return self._normalize_percentages(self._compute_capability_counts(rows))

    @staticmethod
    def _manifest_path_for_dataset(dataset_path: str | Path) -> Path:
        path = resolve_workspace_path(dataset_path)
        return path.with_name(f"{path.stem}.composition_manifest.json")

    @staticmethod
    def _write_manifest_sync(path: Path, payload: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    @staticmethod
    def _load_manifest_sync(path: Path) -> dict[str, Any]:
        return json.loads(path.read_text(encoding="utf-8"))

    @staticmethod
    def _dedupe_warnings(warnings: list[str]) -> list[str]:
        return list(dict.fromkeys(warning for warning in warnings if warning))

    def _inspect_source(self, source_path: str) -> dict[str, Any]:
        resolved = resolve_workspace_path(source_path)
        if not resolved.exists():
            return {
                "input_path": source_path,
                "resolved_path": str(resolved),
                "kind": "missing",
                "exists": False,
                "estimated_chunks": 0,
                "estimated_text_chars": 0,
                "scanned_files": 0,
                "code_files": 0,
                "truncated": False,
                "warnings": ["Path does not exist."],
            }

        if resolved.is_dir():
            return self._inspect_directory(source_path, resolved)

        return self._inspect_file(source_path, resolved)

    def _inspect_directory(self, input_path: str, path: Path) -> dict[str, Any]:
        warnings: list[str] = []
        estimated_chunks = 0
        estimated_text_chars = 0
        scanned_files = 0
        code_files = 0
        truncated = False

        for root, dirs, files in os.walk(path):
            dirs[:] = [
                directory
                for directory in dirs
                if directory not in self._IGNORED_DIRS and not directory.startswith(".")
            ]

            for file_name in files:
                if scanned_files >= self._MAX_DIRECTORY_FILES:
                    truncated = True
                    break

                candidate = Path(root) / file_name
                if not self._is_text_like_file(candidate):
                    continue

                summary = self._inspect_file(str(candidate), candidate, resolved_input=False)
                if not summary["exists"]:
                    continue

                scanned_files += summary["scanned_files"]
                estimated_chunks += summary["estimated_chunks"]
                estimated_text_chars += summary["estimated_text_chars"]
                code_files += summary["code_files"]
                warnings.extend(summary["warnings"])

            if truncated:
                break

        if truncated:
            warnings.append(
                f"Directory scan capped at {self._MAX_DIRECTORY_FILES} files for preview."
            )
        if scanned_files == 0:
            warnings.append("Directory preview did not find any supported text or code files.")

        return {
            "input_path": input_path,
            "resolved_path": str(path),
            "kind": "directory",
            "exists": True,
            "estimated_chunks": estimated_chunks,
            "estimated_text_chars": estimated_text_chars,
            "scanned_files": scanned_files,
            "code_files": code_files,
            "truncated": truncated,
            "warnings": warnings,
        }

    def _inspect_file(
        self,
        input_path: str,
        path: Path,
        *,
        resolved_input: bool = True,
    ) -> dict[str, Any]:
        warnings: list[str] = []
        suffix = path.suffix.lower()
        estimated_chunks = 0
        estimated_text_chars = 0

        if suffix in self._LOADER_SUFFIXES:
            try:
                loader = get_loader(str(path))
                _, pages = loader.load(str(path))
                estimated_chunks = len(pages)
                estimated_text_chars = sum(
                    len(str(page.get("markdown", ""))) for page in pages
                )
            except Exception as exc:
                warnings.append(f"Fell back to raw text scan for {path.name}: {exc}")

        if estimated_chunks == 0 and self._is_text_like_file(path):
            try:
                char_count = self._read_text_char_count(path)
                estimated_text_chars = max(estimated_text_chars, char_count)
                estimated_chunks = max(1, math.ceil(char_count / self._TARGET_CHARS_PER_CHUNK))
            except Exception as exc:
                warnings.append(f"Could not inspect text file {path.name}: {exc}")

        if estimated_chunks == 0:
            return {
                "input_path": input_path,
                "resolved_path": str(path if resolved_input else path.resolve()),
                "kind": "unsupported",
                "exists": False,
                "estimated_chunks": 0,
                "estimated_text_chars": 0,
                "scanned_files": 0,
                "code_files": 0,
                "truncated": False,
                "warnings": warnings or ["Unsupported or unreadable file."],
            }

        return {
            "input_path": input_path,
            "resolved_path": str(path if resolved_input else path.resolve()),
            "kind": "file",
            "exists": True,
            "estimated_chunks": estimated_chunks,
            "estimated_text_chars": estimated_text_chars,
            "scanned_files": 1,
            "code_files": 1 if self._is_code_file(path) else 0,
            "truncated": False,
            "warnings": warnings,
        }

    def _read_text_char_count(self, path: Path) -> int:
        file_size = path.stat().st_size
        if file_size > self._MAX_RAW_TEXT_CHARS:
            return file_size

        with path.open("r", encoding="utf-8", errors="ignore") as handle:
            return len(handle.read())

    def _is_text_like_file(self, path: Path) -> bool:
        suffix = path.suffix.lower()
        if suffix in self._TEXT_SUFFIXES:
            return True
        return path.name.lower() in self._TEXT_FILENAMES

    def _is_code_file(self, path: Path) -> bool:
        return path.suffix.lower() in self._CODE_SUFFIXES or path.name.lower() in {
            "dockerfile",
            "makefile",
            "cmakelists.txt",
            "justfile",
            "jenkinsfile",
        }

    @staticmethod
    def _normalize_percentages(weights: dict[str, int]) -> dict[str, int]:
        total = sum(weights.values())
        if total <= 0:
            return {}

        normalized_floats = {
            capability: (weight / total) * 100.0 for capability, weight in weights.items()
        }
        normalized = {
            capability: int(math.floor(value))
            for capability, value in normalized_floats.items()
        }
        remaining = 100 - sum(normalized.values())
        if remaining > 0:
            ordered = sorted(
                normalized_floats.items(),
                key=lambda item: (item[1] - math.floor(item[1]), item[0]),
                reverse=True,
            )
            for capability, _ in ordered[:remaining]:
                normalized[capability] += 1
        return normalized

    def _allocate_rows(
        self,
        targets: list[CapabilityTarget],
        resolved_mix: dict[str, int],
        row_target: int,
    ) -> dict[str, int]:
        min_total = sum(target.min_rows for target in targets)
        if min_total > row_target:
            min_total = 0

        row_plan = {target.capability: (target.min_rows if min_total else 0) for target in targets}
        remaining_rows = row_target - sum(row_plan.values())

        float_counts = {
            capability: (weight / 100.0) * remaining_rows
            for capability, weight in resolved_mix.items()
        }
        floor_counts = {
            capability: int(math.floor(value))
            for capability, value in float_counts.items()
        }
        for capability, count in floor_counts.items():
            row_plan[capability] = row_plan.get(capability, 0) + count

        remaining = row_target - sum(row_plan.values())
        if remaining > 0:
            ordered = sorted(
                float_counts.items(),
                key=lambda item: (item[1] - math.floor(item[1]), item[0]),
                reverse=True,
            )
            for capability, _ in ordered[:remaining]:
                row_plan[capability] = row_plan.get(capability, 0) + 1

        return row_plan
