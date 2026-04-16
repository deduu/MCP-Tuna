"""Build reusable source chunks and pack combinations for profiled generation."""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from data_generator_pipeline.loaders import get_loader
from shared.workspace_paths import resolve_workspace_path


@dataclass(frozen=True)
class SourceChunk:
    chunk_id: str
    file_name: str
    file_path: str
    page_index: int
    text: str


@dataclass(frozen=True)
class SourcePack:
    pack_id: str
    kind: str
    file_name: str
    page_index: int
    text: str
    source_refs: list[dict[str, Any]]


class SourcePackService:
    """Turn input paths into chunk inventories and composed source packs."""

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

    def build_packs(
        self,
        source_paths: list[str],
        *,
        include_multi_hop: bool = True,
    ) -> dict[str, Any]:
        source_summaries: list[dict[str, Any]] = []
        chunks: list[SourceChunk] = []
        warnings: list[str] = []

        for input_path in source_paths:
            summary, new_chunks = self._load_source(input_path)
            source_summaries.append(summary)
            chunks.extend(new_chunks)
            warnings.extend(summary["warnings"])

        single_packs = [self._chunk_to_pack(chunk) for chunk in chunks]
        multi_hop_packs = (
            self._build_multi_hop_packs(chunks) if include_multi_hop else []
        )

        return {
            "source_summaries": source_summaries,
            "source_totals": {
                "valid_sources": sum(
                    1 for summary in source_summaries if summary["estimated_chunks"] > 0
                ),
                "estimated_chunks": sum(
                    summary["estimated_chunks"] for summary in source_summaries
                ),
                "estimated_text_chars": sum(
                    summary["estimated_text_chars"] for summary in source_summaries
                ),
                "scanned_files": sum(summary["scanned_files"] for summary in source_summaries),
                "code_files": sum(summary["code_files"] for summary in source_summaries),
            },
            "warnings": warnings,
            "chunks": chunks,
            "single_packs": single_packs,
            "multi_hop_packs": multi_hop_packs,
        }

    def _load_source(self, source_path: str) -> tuple[dict[str, Any], list[SourceChunk]]:
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
            }, []

        if resolved.is_dir():
            return self._load_directory(source_path, resolved)

        return self._load_file(source_path, resolved)

    def _load_directory(
        self,
        input_path: str,
        path: Path,
    ) -> tuple[dict[str, Any], list[SourceChunk]]:
        warnings: list[str] = []
        chunks: list[SourceChunk] = []
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

                summary, file_chunks = self._load_file(
                    str(candidate.resolve()), candidate, resolved_input=False
                )
                if summary["estimated_chunks"] == 0:
                    continue

                scanned_files += 1
                code_files += summary["code_files"]
                warnings.extend(summary["warnings"])
                chunks.extend(file_chunks)

            if truncated:
                break

        if truncated:
            warnings.append(
                f"Directory scan capped at {self._MAX_DIRECTORY_FILES} files for generation."
            )
        if scanned_files == 0:
            warnings.append("Directory did not contain supported text or code files.")

        return {
            "input_path": input_path,
            "resolved_path": str(path),
            "kind": "directory",
            "exists": True,
            "estimated_chunks": len(chunks),
            "estimated_text_chars": sum(len(chunk.text) for chunk in chunks),
            "scanned_files": scanned_files,
            "code_files": code_files,
            "truncated": truncated,
            "warnings": warnings,
        }, chunks

    def _load_file(
        self,
        input_path: str,
        path: Path,
        *,
        resolved_input: bool = True,
    ) -> tuple[dict[str, Any], list[SourceChunk]]:
        warnings: list[str] = []
        chunks: list[SourceChunk] = []
        file_name = path.stem
        suffix = path.suffix.lower()

        if suffix in self._LOADER_SUFFIXES:
            try:
                loader = get_loader(str(path))
                loaded_file_name, pages = loader.load(str(path))
                file_name = loaded_file_name or file_name
                for page in pages:
                    text = str(page.get("markdown", "")).strip()
                    if not text:
                        continue
                    chunks.append(
                        SourceChunk(
                            chunk_id=f"{path.resolve()}::{page.get('index', 0)}",
                            file_name=file_name,
                            file_path=str(path.resolve()),
                            page_index=int(page.get("index", 0)),
                            text=text,
                        )
                    )
            except Exception as exc:
                warnings.append(f"Fell back to raw text scan for {path.name}: {exc}")

        if not chunks and self._is_text_like_file(path):
            try:
                raw_text = self._read_text(path)
                text_chunks = self._split_text(raw_text)
                chunks.extend(
                    SourceChunk(
                        chunk_id=f"{path.resolve()}::{index}",
                        file_name=file_name,
                        file_path=str(path.resolve()),
                        page_index=index,
                        text=chunk,
                    )
                    for index, chunk in enumerate(text_chunks)
                    if chunk.strip()
                )
            except Exception as exc:
                warnings.append(f"Could not inspect text file {path.name}: {exc}")

        resolved_path = str(path if resolved_input else path.resolve())
        if not chunks:
            return {
                "input_path": input_path,
                "resolved_path": resolved_path,
                "kind": "unsupported",
                "exists": False,
                "estimated_chunks": 0,
                "estimated_text_chars": 0,
                "scanned_files": 0,
                "code_files": 0,
                "truncated": False,
                "warnings": warnings or ["Unsupported or unreadable file."],
            }, []

        return {
            "input_path": input_path,
            "resolved_path": resolved_path,
            "kind": "file",
            "exists": True,
            "estimated_chunks": len(chunks),
            "estimated_text_chars": sum(len(chunk.text) for chunk in chunks),
            "scanned_files": 1,
            "code_files": 1 if self._is_code_file(path) else 0,
            "truncated": False,
            "warnings": warnings,
        }, chunks

    def _build_multi_hop_packs(self, chunks: list[SourceChunk]) -> list[SourcePack]:
        if len(chunks) < 2:
            return []

        packs: list[SourcePack] = []
        adjacent_pairs = zip(chunks, chunks[1:])
        for left, right in adjacent_pairs:
            refs = [
                {
                    "file_name": left.file_name,
                    "file_path": left.file_path,
                    "page": left.page_index + 1,
                    "chunk_id": left.chunk_id,
                },
                {
                    "file_name": right.file_name,
                    "file_path": right.file_path,
                    "page": right.page_index + 1,
                    "chunk_id": right.chunk_id,
                },
            ]
            text = (
                "Excerpt A:\n"
                f"{left.text}\n\n"
                "Excerpt B:\n"
                f"{right.text}"
            )
            packs.append(
                SourcePack(
                    pack_id=f"{left.chunk_id}++{right.chunk_id}",
                    kind="multi_hop",
                    file_name=f"{left.file_name}+{right.file_name}",
                    page_index=min(left.page_index, right.page_index),
                    text=text,
                    source_refs=refs,
                )
            )
        return packs

    @staticmethod
    def _chunk_to_pack(chunk: SourceChunk) -> SourcePack:
        return SourcePack(
            pack_id=chunk.chunk_id,
            kind="single",
            file_name=chunk.file_name,
            page_index=chunk.page_index,
            text=chunk.text,
            source_refs=[
                {
                    "file_name": chunk.file_name,
                    "file_path": chunk.file_path,
                    "page": chunk.page_index + 1,
                    "chunk_id": chunk.chunk_id,
                }
            ],
        )

    def _read_text(self, path: Path) -> str:
        file_size = path.stat().st_size
        if file_size > self._MAX_RAW_TEXT_CHARS:
            with path.open("r", encoding="utf-8", errors="ignore") as handle:
                return handle.read(self._MAX_RAW_TEXT_CHARS)

        with path.open("r", encoding="utf-8", errors="ignore") as handle:
            return handle.read()

    def _split_text(self, text: str) -> list[str]:
        stripped = text.strip()
        if not stripped:
            return []
        if len(stripped) <= self._TARGET_CHARS_PER_CHUNK:
            return [stripped]

        parts = [part.strip() for part in stripped.split("\n\n") if part.strip()]
        if len(parts) <= 1:
            return [
                stripped[index:index + self._TARGET_CHARS_PER_CHUNK].strip()
                for index in range(0, len(stripped), self._TARGET_CHARS_PER_CHUNK)
            ]

        chunks: list[str] = []
        current: list[str] = []
        current_len = 0
        for part in parts:
            part_len = len(part) + 2
            if current and current_len + part_len > self._TARGET_CHARS_PER_CHUNK:
                chunks.append("\n\n".join(current).strip())
                current = [part]
                current_len = len(part)
                continue

            if len(part) > self._TARGET_CHARS_PER_CHUNK:
                if current:
                    chunks.append("\n\n".join(current).strip())
                    current = []
                    current_len = 0
                chunks.extend(
                    part[index:index + self._TARGET_CHARS_PER_CHUNK].strip()
                    for index in range(0, len(part), self._TARGET_CHARS_PER_CHUNK)
                )
                continue

            current.append(part)
            current_len += part_len

        if current:
            chunks.append("\n\n".join(current).strip())

        return chunks

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
