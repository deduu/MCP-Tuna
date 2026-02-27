"""Dataset persistence service — save, load, preview, split, merge datasets."""

from __future__ import annotations

import json
import math
import random
from pathlib import Path
from typing import Any, Dict, List, Optional

from shared.dataset_models import DatasetConfig, DatasetMetadata


class DatasetService:
    """Stateless service for dataset persistence operations.

    The dataset_id is derived from the filename stem (human-readable, not UUID).
    Example: saving to "my_sft_data.jsonl" yields dataset_id="my_sft_data".
    """

    _SUPPORTED_SAVE_FORMATS = {"jsonl", "json", "parquet"}
    _SUPPORTED_LOAD_EXTENSIONS = {".jsonl", ".json", ".parquet", ".csv"}

    def __init__(self, config: Optional[DatasetConfig] = None) -> None:
        self.config = config or DatasetConfig()

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
            self._write_jsonl(path, data_points)
        elif format == "json":
            self._write_json(path, data_points)
        elif format == "parquet":
            self._write_parquet(path, data_points)

        columns = list(data_points[0].keys()) if data_points else []
        meta = DatasetMetadata(
            dataset_id=path.stem,
            file_path=str(path.resolve()),
            format=format,
            row_count=len(data_points),
            columns=columns,
            technique=self._detect_technique(columns),
            size_bytes=path.stat().st_size if path.exists() else 0,
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

        ext = path.suffix.lower()
        if ext not in self._SUPPORTED_LOAD_EXTENSIONS:
            return {"success": False, "error": f"Unsupported extension: {ext}"}

        if ext == ".jsonl":
            data_points = self._read_jsonl(path)
        elif ext == ".json":
            data_points = self._read_json(path)
        elif ext == ".parquet":
            data_points = self._read_parquet(path)
        elif ext == ".csv":
            data_points = self._read_csv(path)
        else:
            return {"success": False, "error": f"Unsupported extension: {ext}"}

        columns = list(data_points[0].keys()) if data_points else []
        return {
            "success": True,
            "data_points": data_points,
            "row_count": len(data_points),
            "columns": columns,
            "technique": self._detect_technique(columns),
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
            rows, total = self._preview_jsonl(path, n)
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
        """Return metadata about a dataset file."""
        path = Path(file_path)
        if not path.exists():
            return {"success": False, "error": f"File not found: {file_path}"}

        ext = path.suffix.lower()
        fmt = ext.lstrip(".")

        # Determine row count and columns efficiently
        if ext == ".jsonl":
            row_count, columns = self._inspect_jsonl(path)
        else:
            full = await self.load(file_path)
            if not full["success"]:
                return full
            row_count = full["row_count"]
            columns = full["columns"]

        meta = DatasetMetadata(
            dataset_id=path.stem,
            file_path=str(path.resolve()),
            format=fmt,
            row_count=row_count,
            columns=columns,
            technique=self._detect_technique(columns),
            size_bytes=path.stat().st_size,
        )
        return {"success": True, "metadata": meta.model_dump()}

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
        """Split a dataset into train/val/test files."""
        if not math.isclose(train_ratio + val_ratio + test_ratio, 1.0, abs_tol=1e-6):
            return {
                "success": False,
                "error": f"Ratios must sum to 1.0, got {train_ratio + val_ratio + test_ratio:.4f}",
            }

        loaded = await self.load(file_path)
        if not loaded["success"]:
            return loaded

        data = list(loaded["data_points"])
        rng = random.Random(seed)
        rng.shuffle(data)

        n = len(data)
        n_train = round(n * train_ratio)
        n_val = round(n * val_ratio)
        # test gets the remainder to guarantee exact total
        train_data = data[:n_train]
        val_data = data[n_train : n_train + n_val]
        test_data = data[n_train + n_val :]

        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        stem = Path(file_path).stem

        splits: Dict[str, Dict[str, Any]] = {}
        for name, subset in [("train", train_data), ("val", val_data), ("test", test_data)]:
            p = out / f"{stem}_{name}.{format}"
            await self.save(subset, str(p), format=format)
            splits[name] = {"path": str(p.resolve()), "count": len(subset), "format": format}

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
        """Merge multiple dataset files into one."""
        if not file_paths:
            return {"success": False, "error": "No file paths provided"}

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

        fmt = Path(output_path).suffix.lstrip(".") or "jsonl"
        save_result = await self.save(all_points, output_path, format=fmt)
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
    def _detect_technique(columns: List[str]) -> Optional[str]:
        cols = set(columns)
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
    def _inspect_jsonl(path: Path) -> tuple[int, List[str]]:
        """Count rows and extract columns from first row."""
        columns: List[str] = []
        count = 0
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                count += 1
                if count == 1:
                    columns = list(json.loads(line).keys())
        return count, columns
