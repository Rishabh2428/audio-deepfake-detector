"""SQLite persistence for metadata and actual DSP reports (never raw audio)."""

from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import JSON, DateTime, Float, Integer, String, Text, create_engine, select
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column, sessionmaker


class Base(DeclarativeBase):
    pass


class AnalysisRecord(Base):
    __tablename__ = "analyses"

    analysis_id: Mapped[str] = mapped_column(String(36), primary_key=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    filename: Mapped[str] = mapped_column(String(160), nullable=False)
    file_sha256: Mapped[str] = mapped_column(String(64), nullable=False)
    file_bytes: Mapped[int] = mapped_column(Integer, nullable=False)
    duration_seconds: Mapped[float] = mapped_column(Float, nullable=False)
    original_sample_rate: Mapped[int] = mapped_column(Integer, nullable=False)
    original_channels: Mapped[int] = mapped_column(Integer, nullable=False)
    decision: Mapped[str] = mapped_column(String(32), nullable=False)
    report: Mapped[dict] = mapped_column(JSON, nullable=False)


class TrainingExample(Base):
    __tablename__ = "training_examples"

    example_id: Mapped[str] = mapped_column(String(36), primary_key=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    filename: Mapped[str] = mapped_column(String(160), nullable=False)
    file_sha256: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    label: Mapped[str] = mapped_column(String(16), nullable=False)
    group_id: Mapped[str] = mapped_column(String(160), nullable=False)
    dataset_name: Mapped[str] = mapped_column(String(160), nullable=False)
    features: Mapped[list | dict] = mapped_column(JSON, nullable=False)


def make_engine(database_url: str):
    connect_args = {"check_same_thread": False} if database_url.startswith("sqlite:") else {}
    return create_engine(database_url, connect_args=connect_args, future=True)


def initialize_database(engine) -> None:
    """Create initial schema and reject future schema versions we cannot read."""
    if engine.dialect.name == "sqlite":
        with engine.begin() as connection:
            version = connection.exec_driver_sql("PRAGMA user_version").scalar_one()
            if version > 2:
                raise RuntimeError(f"Database schema version {version} is newer than this AegisVoice build.")
            Base.metadata.create_all(bind=connection)
            connection.exec_driver_sql("PRAGMA user_version = 2")
    else:
        Base.metadata.create_all(bind=engine)


def make_session_factory(engine):
    return sessionmaker(bind=engine, class_=Session, expire_on_commit=False)


def serialize_record(record: AnalysisRecord, *, include_report: bool = True) -> dict:
    created = record.created_at
    if created.tzinfo is None:
        created = created.replace(tzinfo=timezone.utc)
    result = {
        "analysis_id": record.analysis_id,
        "created_at": created.astimezone(timezone.utc).isoformat(),
        "filename": record.filename,
        "file_sha256": record.file_sha256,
        "file_bytes": record.file_bytes,
        "duration_seconds": record.duration_seconds,
        "original_sample_rate": record.original_sample_rate,
        "original_channels": record.original_channels,
        "decision": record.decision,
    }
    if include_report:
        result["report"] = record.report
    return result

