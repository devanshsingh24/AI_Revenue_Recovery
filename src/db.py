from __future__ import annotations
import os
from contextlib import contextmanager
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, DeclarativeBase

DATABASE_URL = os.getenv(
    "DATABASE_URL", "postgresql://postgres:aBCpostgres&ai@localhost:5432/postgres"
)

engine = create_engine(DATABASE_URL, pool_pre_ping=True)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)


class Base(DeclarativeBase):
    pass


@contextmanager
def get_session():
    """Use as: with get_session() as session: ... — commits on success,
    rolls back on exception, always closes."""
    session = SessionLocal()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def init_db() -> None:
    """Creates tables if they don't exist. No Alembic/migrations yet —
    fine for a single dev database, not fine once schema changes need
    to roll out against data you can't just drop. Flagging, not building,
    since migrations are out of scope for this pass."""
    from src import db_models  # noqa: F401 - import registers models on Base
    Base.metadata.create_all(bind=engine)