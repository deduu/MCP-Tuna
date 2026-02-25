# services/base.py

from typing import Type, Optional, Any, List, Dict, Union
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from fastapi import HTTPException, status
import logging

logger = logging.getLogger(__name__)


class BaseService:
    def __init__(self, db: AsyncSession, model: Type[Any]):
        self.db = db
        self.model = model

    async def get_by_id(self, item_id: int) -> Optional[Any]:
        primary_key_col = self.model.__table__.primary_key.columns.values()[0]
        try:
            stmt = select(self.model).where(primary_key_col == item_id)
            result = await self.db.execute(stmt)
            item = result.scalar_one_or_none()
            if not item:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"{self.model.__name__} with ID {item_id} not found"
                )
            return item
        except Exception as e:
            logger.error(
                f"Error fetching {self.model.__name__} by ID {item_id}: {e}")
            raise

    async def list(self, filters: Dict = None) -> List[Any]:
        try:
            stmt = select(self.model)
            if filters:
                for key, value in filters.items():
                    stmt = stmt.where(getattr(self.model, key) == value)
            result = await self.db.execute(stmt)
            return result.scalars().all()
        except Exception as e:
            logger.error(f"Error fetching {self.model.__name__}: {e}")
            raise

    async def get(self, obj_id: int) -> Any:
        try:
            res = await self.db.get(self.model, obj_id)
            if not res:
                raise ValueError(f"{self.model.__name__} {obj_id} not found")
            return res
        except Exception as e:
            logger.error(
                f"Error fetching {self.model.__name__} by ID {obj_id}: {e}")
            raise

    async def get_all(self, skip: int = 0, limit: int = 100) -> List[Any]:
        try:
            stmt = select(self.model).offset(skip).limit(limit)
            result = await self.db.execute(stmt)
            return result.scalars().all()
        except Exception as e:
            logger.error(f"Error fetching all {self.model.__name__}: {e}")
            raise

    async def create(self, schema: Any) -> Any:
        try:
            obj = self.model(**schema.model_dump())
            self.db.add(obj)
            await self.db.commit()
            await self.db.refresh(obj)
            return obj
        except IntegrityError:
            await self.db.rollback()
            logger.warning(
                f"Integrity error while creating {self.model.__name__}")
            raise HTTPException(status_code=400, detail="Integrity error")
        except Exception as e:
            await self.db.rollback()
            logger.error(
                f"Unexpected error while creating {self.model.__name__}: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    async def update(self, item_id: int, schema: Any) -> Any:
        obj = await self.get_by_id(item_id)
        data = schema.model_dump(exclude_unset=True)
        for key, value in data.items():
            setattr(obj, key, value)
        try:
            await self.db.commit()
            await self.db.refresh(obj)
            return obj
        except IntegrityError:
            await self.db.rollback()
            logger.warning(
                f"Integrity error while updating {self.model.__name__} with ID {item_id}")
            raise HTTPException(
                status_code=400, detail="Update failed due to constraint violation")
        except Exception as e:
            await self.db.rollback()
            logger.error(
                f"Unexpected error while updating {self.model.__name__} with ID {item_id}: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    async def delete(self, item_id: int) -> bool:
        obj = await self.get_by_id(item_id)
        try:
            await self.db.delete(obj)
            await self.db.commit()
            return True
        except Exception as e:
            await self.db.rollback()
            logger.error(
                f"Failed to delete {self.model.__name__} with ID {item_id}: {e}")
            raise HTTPException(
                status_code=500, detail=f"Failed to delete {self.model.__name__}: {e}")

    async def filter_by(self, filters: Dict[str, Union[str, int, float]]) -> List[Any]:
        try:
            stmt = select(self.model)
            for key, value in filters.items():
                stmt = stmt.where(getattr(self.model, key) == value)
            result = await self.db.execute(stmt)
            return result.scalars().all()
        except Exception as e:
            logger.error(
                f"Error filtering {self.model.__name__} with filters {filters}: {e}")
            raise
