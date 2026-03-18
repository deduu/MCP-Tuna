"""Dataset persistence service — save, load, preview, split, merge datasets."""

from __future__ import annotations

import asyncio
import json
import math
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from shared.dataset_models import DatasetConfig, DatasetMetadata
from shared.multimodal_models import (
    extract_text_from_content,
    is_vlm_sample,
)


class DatasetService:
    """Stateless service for dataset persistence operations.

    The dataset_id is derived from the filename stem (human-readable, not UUID).
    Example: saving to "my_sft_data.jsonl" yields dataset_id="my_sft_data".
    """

    _SUPPORTED_SAVE_FORMATS = {"jsonl", "json", "parquet"}
    _SUPPORTED_LOAD_EXTENSIONS = {".jsonl", ".json", ".parquet", ".csv"}

    def __init__(self, config: Optional[DatasetConfig] = None) -> None:
        self.config = config or DatasetConfig()

    @staticmethod
    def _resolve_dir(dir_path: Path) -> Optional[Path]:
        """Pick the first dataset file from a directory, or None."""
        supported = (".jsonl", ".json", ".csv", ".parquet")
        for f in sorted(dir_path.iterdir()):
            if f.is_file() and f.suffix.lower() in supported:
                return f
        return None

    # ------------------------------------------------------------------
    # save
    # ------------------------------------------------------------------

    async def save(
        self,
        data_points: List[Dict[str, Any]],
        output_path: str,
        format: str = "jsonl",
    ) -> Dict[str, Any]:
        """Save data_points to disk.

        Returns ``{success, dataset_id, file_path, row_count, columns, technique, ...}``.
        Supported formats: jsonl, json, parquet.
        """
        if format not in self._SUPPORTED_SAVE_FORMATS:
            return {
                "success": False,
                "error": f"Unsupported format: {format}. Use one of {sorted(self._SUPPORTED_SAVE_FORMATS)}",
            }

        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        if format == "jsonl":
            await asyncio.to_thread(self._write_jsonl, path, data_points)
        elif format == "json":
            await asyncio.to_thread(self._write_json, path, data_points)
        elif format == "parquet":
            await asyncio.to_thread(self._write_parquet, path, data_points)

        columns = list(data_points[0].keys()) if data_points else []
        stat = path.stat() if path.exists() else None
        sample_row = data_points[0] if data_points else None
        meta = DatasetMetadata(
            dataset_id=path.stem,
            file_path=str(path.resolve()),
            format=format,
            row_count=len(data_points),
            columns=columns,
            technique=self._detect_technique(columns, sample_row),
            size_bytes=stat.st_size if stat else 0,
            modified_at=(
                datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).isoformat()
                if stat
                else None
            ),
        )
        return {"success": True, **meta.model_dump()}

    # ------------------------------------------------------------------
    # load
    # ------------------------------------------------------------------

    async def load(self, file_path: str) -> Dict[str, Any]:
        """Load dataset from disk — auto-detects format from extension."""
        path = Path(file_path)
        if not path.exists():
            return {"success": False, "error": f"File not found: {file_path}"}

        if path.is_dir():
            path = self._resolve_dir(path)
            if path is None:
                return {"success": False, "error": f"No dataset files in {file_path}"}

        ext = path.suffix.lower()
        if ext not in self._SUPPORTED_LOAD_EXTENSIONS:
            return {"success": False, "error": f"Unsupported extension: {ext}"}

        if ext == ".jsonl":
            data_points = await asyncio.to_thread(self._read_jsonl, path)
        elif ext == ".json":
            data_points = await asyncio.to_thread(self._read_json, path)
        elif ext == ".parquet":
            data_points = await asyncio.to_thread(self._read_parquet, path)
        elif ext == ".csv":
            data_points = await asyncio.to_thread(self._read_csv, path)
        else:
            return {"success": False, "error": f"Unsupported extension: {ext}"}

        columns = list(data_points[0].keys()) if data_points else []
        return {
            "success": True,
            "data_points": data_points,
            "row_count": len(data_points),
            "columns": columns,
            "technique": self._detect_technique(columns, data_points[0] if data_points else None),
        }

    # ------------------------------------------------------------------
    # preview
    # ------------------------------------------------------------------

    async def preview(self, file_path: str, n: int = 5) -> Dict[str, Any]:
        """Return first *n* rows without loading the entire file (JSONL-optimised)."""
        path = Path(file_path)
        if not path.exists():
            return {"success": False, "error": f"File not found: {file_path}"}

        ext = path.suffix.lower()

        if ext == ".jsonl":
            rows, total = await asyncio.to_thread(self._preview_jsonl, path, n)
        else:
            # For other formats fall back to full load then slice
            full = await self.load(file_path)
            if not full["success"]:
                return full
            all_rows = full["data_points"]
            rows = all_rows[:n]
            total = len(all_rows)

        columns = list(rows[0].keys()) if rows else []
        return {
            "success": True,
            "rows": rows,
            "total_rows": total,
            "columns": columns,
        }

    # ------------------------------------------------------------------
    # info
    # ------------------------------------------------------------------

    async def info(self, file_path: str) -> Dict[str, Any]:
        """Return metadata about a dataset file (or first dataset in a dir)."""
        path = Path(file_path)
        if not path.exists():
            return {"success": False, "error": f"File not found: {file_path}"}

        if path.is_dir():
            path = self._resolve_dir(path)
            if path is None:
                return {"success": False, "error": f"No dataset files in {file_path}"}

        ext = path.suffix.lower()
        fmt = ext.lstrip(".")

        # Determine row count and columns efficiently
        if ext == ".jsonl":
            row_count, columns, sample_row = await asyncio.to_thread(self._inspect_jsonl, path)
        else:
            full = await self.load(file_path)
            if not full["success"]:
                return full
            row_count = full["row_count"]
            columns = full["columns"]
            sample_row = full["data_points"][0] if full["data_points"] else None

        meta = DatasetMetadata(
            dataset_id=path.stem,
            file_path=str(path.resolve()),
            format=fmt,
            row_count=row_count,
            columns=columns,
            technique=self._detect_technique(columns, sample_row),
            size_bytes=path.stat().st_size,
            modified_at=datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc).isoformat(),
        )
        return {"success": True, "metadata": meta.model_dump()}

    # ------------------------------------------------------------------
    # delete
    # ------------------------------------------------------------------

    async def delete(self, file_path: str) -> Dict[str, Any]:
        """Delete a dataset file from disk."""
        path = Path(file_path)
        if not path.exists():
            return {"success": False, "error": f"File not found: {file_path}"}

        if path.is_dir():
            path = self._resolve_dir(path)
            if path is None:
                return {"success": False, "error": f"No dataset files in {file_path}"}

        if not path.is_file():
            return {"success": False, "error": f"Not a file: {file_path}"}

        ext = path.suffix.lower()
        if ext not in self._SUPPORTED_LOAD_EXTENSIONS:
            return {"success": False, "error": f"Unsupported extension: {ext}"}

        resolved = str(path.resolve())
        await asyncio.to_thread(path.unlink)
        return {"success": True, "file_path": resolved, "deleted": True}

    # ------------------------------------------------------------------
    # split
    # ------------------------------------------------------------------

    async def split(
        self,
        file_path: str,
        output_dir: str,
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        test_ratio: float = 0.1,
        seed: int = 42,
        format: str = "jsonl",
    ) -> Dict[str, Any]:
        """Split a dataset into train/val/test files.

        For JSONL files, uses a memory-efficient streaming approach that
        avoids loading the entire dataset into RAM.
        """
        if not math.isclose(train_ratio + val_ratio + test_ratio, 1.0, abs_tol=1e-6):
            return {
                "success": False,
                "error": f"Ratios must sum to 1.0, got {train_ratio + val_ratio + test_ratio:.4f}",
            }

        p = Path(file_path)
        is_jsonl = p.suffix.lower() in (".jsonl",) and format == "jsonl"

        if is_jsonl:
            return await asyncio.to_thread(
                self._split_jsonl_streaming,
                p, output_dir, train_ratio, val_ratio, seed,
            )

        # Fallback: load full dataset for non-JSONL formats
        loaded = await self.load(file_path)
        if not loaded["success"]:
            return loaded

        data = list(loaded["data_points"])
        rng = random.Random(seed)
        rng.shuffle(data)

        n = len(data)
        n_train = round(n * train_ratio)
        n_val = round(n * val_ratio)
        train_data = data[:n_train]
        val_data = data[n_train : n_train + n_val]
        test_data = data[n_train + n_val :]

        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        stem = Path(file_path).stem

        splits: Dict[str, Dict[str, Any]] = {}
        for name, subset in [("train", train_data), ("val", val_data), ("test", test_data)]:
            sp = out / f"{stem}_{name}.{format}"
            await self.save(subset, str(sp), format=format)
            splits[name] = {"path": str(sp.resolve()), "count": len(subset), "format": format}

        return {"success": True, "splits": splits}

    @staticmethod
    def _split_jsonl_streaming(
        src: Path,
        output_dir: str,
        train_ratio: float,
        val_ratio: float,
        seed: int,
    ) -> Dict[str, Any]:
        """Stream JSONL split without loading entire dataset into RAM.

        Pass 1: count lines and build a shuffled index mapping.
        Pass 2: stream lines into the appropriate split file.
        """
        # Pass 1 — count rows
        n = 0
        with open(src, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    n += 1

        if n == 0:
            return {"success": False, "error": "Dataset is empty"}

        # Build shuffled assignment: index → split name
        rng = random.Random(seed)
        indices = list(range(n))
        rng.shuffle(indices)

        n_train = round(n * train_ratio)
        n_val = round(n * val_ratio)
        assignment = ["test"] * n  # default
        for i in indices[:n_train]:
            assignment[i] = "train"
        for i in indices[n_train : n_train + n_val]:
            assignment[i] = "val"

        # Pass 2 — stream into split files
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        stem = src.stem

        counts: Dict[str, int] = {"train": 0, "val": 0, "test": 0}
        paths: Dict[str, Path] = {}
        for name in ("train", "val", "test"):
            paths[name] = out / f"{stem}_{name}.jsonl"

        handles = {name: open(paths[name], "w", encoding="utf-8") for name in paths}
        try:
            row_idx = 0
            with open(src, "r", encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    split_name = assignment[row_idx]
                    handles[split_name].write(line if line.endswith("\n") else line + "\n")
                    counts[split_name] += 1
                    row_idx += 1
        finally:
            for h in handles.values():
                h.close()

        splits: Dict[str, Dict[str, Any]] = {}
        for name in ("train", "val", "test"):
            splits[name] = {
                "path": str(paths[name].resolve()),
                "count": counts[name],
                "format": "jsonl",
            }

        return {"success": True, "splits": splits}

    # ------------------------------------------------------------------
    # merge
    # ------------------------------------------------------------------

    async def merge(
        self,
        file_paths: List[str],
        output_path: str,
        deduplicate: bool = False,
        dedup_key: str = "instruction",
    ) -> Dict[str, Any]:
        """Merge multiple dataset files into one.

        For JSONL→JSONL merges, streams rows to avoid loading everything into RAM.
        """
        if not file_paths:
            return {"success": False, "error": "No file paths provided"}

        out_fmt = Path(output_path).suffix.lstrip(".") or "jsonl"
        all_jsonl = out_fmt == "jsonl" and all(
            Path(fp).suffix.lower() == ".jsonl" for fp in file_paths
        )

        if all_jsonl:
            return await asyncio.to_thread(
                self._merge_jsonl_streaming, file_paths, output_path,
                deduplicate, dedup_key,
            )

        # Fallback: load everything for non-JSONL formats
        all_points: List[Dict[str, Any]] = []
        per_file: Dict[str, int] = {}

        for fp in file_paths:
            loaded = await self.load(fp)
            if not loaded["success"]:
                return {"success": False, "error": f"Failed to load {fp}: {loaded.get('error')}"}
            pts = loaded["data_points"]
            per_file[fp] = len(pts)
            all_points.extend(pts)

        if deduplicate:
            seen: set[str] = set()
            unique: List[Dict[str, Any]] = []
            for dp in all_points:
                key_val = str(dp.get(dedup_key, ""))
                if key_val not in seen:
                    seen.add(key_val)
                    unique.append(dp)
            all_points = unique

        save_result = await self.save(all_points, output_path, format=out_fmt)
        if not save_result["success"]:
            return save_result

        return {
            "success": True,
            "file_path": str(Path(output_path).resolve()),
            "total_rows": len(all_points),
            "per_file_counts": per_file,
        }

    # ==================================================================
    # Private helpers
    # ==================================================================

    # ---- technique detection ----

    @staticmethod
    def _detect_technique(
        columns: List[str],
        sample_row: Optional[Dict[str, Any]] = None,
    ) -> Optional[str]:
        cols = set(columns)
        if sample_row and is_vlm_sample(sample_row):
            return "vlm_sft"
        if {"instruction", "output"}.issubset(cols):
            return "sft"
        if {"prompt", "chosen", "rejected"}.issubset(cols):
            return "dpo"
        if {"prompt", "responses", "rewards"}.issubset(cols):
            return "grpo"
        if {"prompt", "completion", "label"}.issubset(cols):
            return "kto"
        return None

    # ---- writers ----

    @staticmethod
    def _write_jsonl(path: Path, data: List[Dict]) -> None:
        with open(path, "w", encoding="utf-8") as f:
            for row in data:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

    @staticmethod
    def _write_json(path: Path, data: List[Dict]) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    @staticmethod
    def _write_parquet(path: Path, data: List[Dict]) -> None:
        import pandas as pd

        df = pd.DataFrame(data)
        df.to_parquet(path, index=False)

    # ---- readers ----

    @staticmethod
    def _read_jsonl(path: Path) -> List[Dict]:
        rows: List[Dict] = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        return rows

    @staticmethod
    def _read_json(path: Path) -> List[Dict]:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    @staticmethod
    def _read_parquet(path: Path) -> List[Dict]:
        import pandas as pd

        df = pd.read_parquet(path)
        return df.to_dict(orient="records")

    @staticmethod
    def _read_csv(path: Path) -> List[Dict]:
        import pandas as pd

        df = pd.read_csv(path)
        return df.to_dict(orient="records")

    # ---- streaming helpers ----

    @staticmethod
    def _merge_jsonl_streaming(
        file_paths: List[str],
        output_path: str,
        deduplicate: bool = False,
        dedup_key: str = "instruction",
    ) -> Dict[str, Any]:
        """Stream-merge JSONL files without loading all into RAM."""
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        seen: set[str] = set()
        total = 0
        per_file: Dict[str, int] = {}

        with open(out, "w", encoding="utf-8") as fout:
            for fp in file_paths:
                count = 0
                with open(fp, "r", encoding="utf-8") as fin:
                    for line in fin:
                        line = line.strip()
                        if not line:
                            continue
                        if deduplicate:
                            row = json.loads(line)
                            key_val = str(row.get(dedup_key, ""))
                            if key_val in seen:
                                continue
                            seen.add(key_val)
                        fout.write(line + "\n")
                        count += 1
                        total += 1
                per_file[fp] = count

        return {
            "success": True,
            "file_path": str(out.resolve()),
            "total_rows": total,
            "per_file_counts": per_file,
        }

    # ---- efficient inspection ----

    @staticmethod
    def _preview_jsonl(path: Path, n: int) -> tuple[List[Dict], int]:
        """Read first *n* rows and count total lines without full load."""
        rows: List[Dict] = []
        total = 0
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                total += 1
                if len(rows) < n:
                    rows.append(json.loads(line))
        return rows, total

    @staticmethod
    def _inspect_jsonl(path: Path) -> tuple[int, List[str], Optional[Dict[str, Any]]]:
        """Count rows and extract columns and the first row."""
        columns: List[str] = []
        count = 0
        sample_row: Optional[Dict[str, Any]] = None
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                count += 1
                if count == 1:
                    sample_row = json.loads(line)
                    columns = list(sample_row.keys())
        return count, columns, sample_row

    async def sample_text_stats(
        self,
        file_path: str,
        sample_size: int = 50,
    ) -> Dict[str, Any]:
        """Sample rows to compute text length statistics without full load.

        Reads up to *sample_size* rows and measures character lengths of the
        primary text fields (instruction/prompt + input/output/response).
        Returns avg_length, max_length, and p95_length.
        """
        path = Path(file_path)
        if not path.exists():
            return {"success": False, "error": f"File not found: {file_path}"}

        if path.is_dir():
            path = self._resolve_dir(path)
            if path is None:
                return {"success": False, "error": f"No dataset files in {file_path}"}

        _TEXT_FIELDS = ("instruction", "prompt", "input", "output", "response",
                        "chosen", "rejected", "completion", "text")

        def _read_sample() -> List[int]:
            lengths: List[int] = []
            with open(path, "r", encoding="utf-8") as f:
                for i, line in enumerate(f):
                    if i >= sample_size:
                        break
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        row = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    total = sum(
                        len(str(row.get(k, "")))
                        for k in _TEXT_FIELDS
                        if k in row
                    )
                    messages = row.get("messages")
                    if isinstance(messages, list):
                        total += sum(
                            len(extract_text_from_content(message.get("content")))
                            for message in messages
                            if isinstance(message, dict)
                        )
                    if total > 0:
                        lengths.append(total)
            return lengths

        lengths = await asyncio.to_thread(_read_sample)
        if not lengths:
            return {
                "success": True,
                "avg_length": 0,
                "max_length": 0,
                "p95_length": 0,
                "sampled_rows": 0,
            }

        lengths.sort()
        p95_idx = min(len(lengths) - 1, int(len(lengths) * 0.95))
        return {
            "success": True,
            "avg_length": round(sum(lengths) / len(lengths)),
            "max_length": max(lengths),
            "p95_length": lengths[p95_idx],
            "sampled_rows": len(lengths),
        }
