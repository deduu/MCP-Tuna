"""File management service — read, write, list, upload files within configured boundaries."""

import base64
from pathlib import Path
from typing import Any, Dict


class FileService:
    """Provides file operations for MCP tool consumption.

    All paths are resolved relative to a root directory to prevent
    directory traversal attacks.
    """

    def __init__(self, root_dir: str):
        self.root = Path(root_dir).resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    def _safe_path(self, relative_path: str) -> Path:
        """Resolve path safely within root directory."""
        resolved = (self.root / relative_path).resolve()
        if not str(resolved).startswith(str(self.root)):
            raise ValueError(f"Path escapes root directory: {relative_path}")
        return resolved

    async def read(self, path: str, encoding: str = "utf-8") -> Dict[str, Any]:
        """Read a file's contents. Returns text for text files, base64 for binary."""
        full_path = self._safe_path(path)
        if not full_path.exists():
            return {"success": False, "error": f"File not found: {path}"}
        if not full_path.is_file():
            return {"success": False, "error": f"Not a file: {path}"}

        try:
            content = full_path.read_text(encoding=encoding)
            return {
                "success": True,
                "path": path,
                "content": content,
                "size": full_path.stat().st_size,
                "encoding": encoding,
            }
        except UnicodeDecodeError:
            content_b64 = base64.b64encode(full_path.read_bytes()).decode("ascii")
            return {
                "success": True,
                "path": path,
                "content_base64": content_b64,
                "size": full_path.stat().st_size,
                "encoding": "base64",
            }

    async def write(self, path: str, content: str, encoding: str = "utf-8") -> Dict[str, Any]:
        """Write text content to a file. Creates parent directories if needed."""
        full_path = self._safe_path(path)
        full_path.parent.mkdir(parents=True, exist_ok=True)
        full_path.write_text(content, encoding=encoding)
        return {
            "success": True,
            "path": path,
            "size": full_path.stat().st_size,
        }

    async def list_dir(self, path: str = ".") -> Dict[str, Any]:
        """List files and directories at a path."""
        full_path = self._safe_path(path)
        if not full_path.exists():
            return {"success": False, "error": f"Directory not found: {path}"}
        if not full_path.is_dir():
            return {"success": False, "error": f"Not a directory: {path}"}

        entries = []
        for entry in sorted(full_path.iterdir()):
            rel = entry.relative_to(self.root)
            entries.append({
                "name": entry.name,
                "path": str(rel),
                "type": "directory" if entry.is_dir() else "file",
                "size": entry.stat().st_size if entry.is_file() else None,
            })
        return {"success": True, "path": path, "entries": entries, "count": len(entries)}

    async def upload(self, filename: str, content_base64: str) -> Dict[str, Any]:
        """Save a base64-encoded file to the uploads directory."""
        full_path = self._safe_path(filename)
        full_path.parent.mkdir(parents=True, exist_ok=True)
        data = base64.b64decode(content_base64)
        full_path.write_bytes(data)
        return {
            "success": True,
            "path": filename,
            "file_path": str(full_path),
            "size": len(data),
        }
