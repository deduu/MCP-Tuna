"""Database service — safe read/write operations against the configured PostgreSQL database."""

import json
from typing import Any, Dict, List, Optional

from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker
from sqlalchemy import text


class DatabaseService:
    """Provides database operations for MCP tool consumption."""

    def __init__(self, database_url: str):
        self._engine = create_async_engine(database_url, pool_pre_ping=True)
        self._session_factory = sessionmaker(
            self._engine, class_=AsyncSession, expire_on_commit=False
        )

    async def query(self, sql: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Execute a read-only SQL query and return results as JSON-serializable dict."""
        async with self._session_factory() as session:
            result = await session.execute(text(sql), params or {})
            rows = result.fetchall()
            columns = list(result.keys())
            data = [dict(zip(columns, row)) for row in rows]
            return {
                "success": True,
                "columns": columns,
                "rows": data,
                "count": len(data),
            }

    async def list_tables(self) -> Dict[str, Any]:
        """List all user tables in the database."""
        sql = """
            SELECT table_name, table_schema
            FROM information_schema.tables
            WHERE table_schema NOT IN ('pg_catalog', 'information_schema')
            ORDER BY table_schema, table_name
        """
        async with self._session_factory() as session:
            result = await session.execute(text(sql))
            rows = result.fetchall()
            tables = [{"table": row[0], "schema": row[1]} for row in rows]
            return {"success": True, "tables": tables, "count": len(tables)}

    async def describe_table(self, table_name: str) -> Dict[str, Any]:
        """Get column info for a specific table."""
        sql = """
            SELECT column_name, data_type, is_nullable, column_default
            FROM information_schema.columns
            WHERE table_name = :table_name
            ORDER BY ordinal_position
        """
        async with self._session_factory() as session:
            result = await session.execute(text(sql), {"table_name": table_name})
            rows = result.fetchall()
            columns = [
                {
                    "name": row[0],
                    "type": row[1],
                    "nullable": row[2] == "YES",
                    "default": row[3],
                }
                for row in rows
            ]
            return {
                "success": True,
                "table": table_name,
                "columns": columns,
                "count": len(columns),
            }

    async def insert(
        self, table_name: str, data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Insert a single row into a table using parameterized query."""
        columns = list(data.keys())
        placeholders = ", ".join(f":{col}" for col in columns)
        col_names = ", ".join(columns)
        sql = f"INSERT INTO {table_name} ({col_names}) VALUES ({placeholders}) RETURNING *"

        async with self._session_factory() as session:
            result = await session.execute(text(sql), data)
            await session.commit()
            row = result.fetchone()
            keys = list(result.keys())
            return {
                "success": True,
                "inserted": dict(zip(keys, row)) if row else {},
            }
