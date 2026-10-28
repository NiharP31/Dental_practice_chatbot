from datetime import UTC, date, datetime

from sqlalchemy import JSON, Boolean, ForeignKey, String, Text, create_engine, event, make_url
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, sessionmaker
from sqlalchemy.pool import NullPool

from app import config

SQLITE_BUSY_TIMEOUT_MS = 5000  # below pytest-timeout's 10s so a stuck lock fails loudly
LOCAL_HOSTS = ("localhost", "127.0.0.1", "::1", None)


def _utcnow() -> datetime:
    return datetime.now(UTC).replace(tzinfo=None)


class Base(DeclarativeBase):
    pass


class Patient(Base):
    __tablename__ = "patients"
    id: Mapped[int] = mapped_column(primary_key=True)
    full_name: Mapped[str] = mapped_column(String(200), index=True)
    phone: Mapped[str] = mapped_column(String(20))
    dob: Mapped[date]
    insurance: Mapped[str] = mapped_column(String(200))  # "none" for self-pay
    # Household link: a verified guardian may book/modify for dependents. Spouses aren't linked.
    guardian_id: Mapped[int | None] = mapped_column(ForeignKey("patients.id"), index=True, default=None)
    created_at: Mapped[datetime] = mapped_column(default=_utcnow)


class Slot(Base):
    __tablename__ = "slots"
    id: Mapped[int] = mapped_column(primary_key=True)
    start: Mapped[datetime] = mapped_column(unique=True, index=True)  # office wall-clock time
    status: Mapped[str] = mapped_column(String(10), default="open")  # open | booked


class Appointment(Base):
    __tablename__ = "appointments"
    id: Mapped[int] = mapped_column(primary_key=True)
    patient_id: Mapped[int] = mapped_column(ForeignKey("patients.id"), index=True)
    slot_id: Mapped[int] = mapped_column(ForeignKey("slots.id"), index=True)
    appt_type: Mapped[str] = mapped_column(String(20))
    status: Mapped[str] = mapped_column(String(10), default="booked")  # booked | cancelled
    notes: Mapped[str] = mapped_column(Text, default="")
    created_at: Mapped[datetime] = mapped_column(default=_utcnow)


class StaffAlert(Base):
    __tablename__ = "staff_alerts"
    id: Mapped[int] = mapped_column(primary_key=True)
    session_id: Mapped[str] = mapped_column(String(64), index=True)
    priority: Mapped[str] = mapped_column(String(10))  # urgent | normal
    summary: Mapped[str] = mapped_column(Text)
    status: Mapped[str] = mapped_column(String(10), default="open")  # open | resolved
    created_at: Mapped[datetime] = mapped_column(default=_utcnow)


class Conversation(Base):
    __tablename__ = "conversations"
    session_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    messages: Mapped[list] = mapped_column(JSON, default=list)
    # verified patient ids, pending modify confirmation, failed verify count
    state: Mapped[dict] = mapped_column(JSON, default=dict)
    flagged: Mapped[bool] = mapped_column(Boolean, default=False)
    updated_at: Mapped[datetime] = mapped_column(default=_utcnow, onupdate=_utcnow)


class IdempotencyKey(Base):
    __tablename__ = "idempotency_keys"
    key: Mapped[str] = mapped_column(String(120), primary_key=True)  # session:turn:fingerprint
    result: Mapped[dict] = mapped_column(JSON)
    created_at: Mapped[datetime] = mapped_column(default=_utcnow)


class Feedback(Base):
    __tablename__ = "feedback"
    id: Mapped[int] = mapped_column(primary_key=True)
    session_id: Mapped[str] = mapped_column(String(64), index=True)
    rating: Mapped[str] = mapped_column(String(4))  # up | down
    comment: Mapped[str] = mapped_column(Text, default="")
    created_at: Mapped[datetime] = mapped_column(default=_utcnow)


class LlmUsage(Base):
    """One row per LLM call: token counts and estimated cost, for the daily budget and cost metrics."""
    __tablename__ = "llm_usage"
    id: Mapped[int] = mapped_column(primary_key=True)
    day: Mapped[date] = mapped_column(index=True)  # office-local date
    session_id: Mapped[str] = mapped_column(String(64), default="")
    provider: Mapped[str] = mapped_column(String(80))
    input_tokens: Mapped[int] = mapped_column(default=0)
    output_tokens: Mapped[int] = mapped_column(default=0)
    cache_read_tokens: Mapped[int] = mapped_column(default=0)
    cache_write_tokens: Mapped[int] = mapped_column(default=0)
    cost_usd: Mapped[float] = mapped_column(default=0.0)
    created_at: Mapped[datetime] = mapped_column(default=_utcnow)


def make_engine(url: str):
    # Supabase hands out postgres:// URLs; SQLAlchemy needs the psycopg3 driver named.
    for prefix in ("postgres://", "postgresql://"):
        if url.startswith(prefix):
            url = "postgresql+psycopg://" + url[len(prefix):]
    if url.startswith("sqlite"):
        engine = create_engine(url, connect_args={"check_same_thread": False})

        @event.listens_for(engine, "connect")
        def _pragmas(conn, _):
            # Wait for a competing writer instead of failing with "database is locked".
            conn.execute(f"PRAGMA busy_timeout = {SQLITE_BUSY_TIMEOUT_MS}")

        return engine
    # Built for Supabase's transaction pooler (port 6543): it already pools server connections and
    # hands each transaction to any of them, so no client-side pool (NullPool) and no prepared
    # statements (prepare_threshold=None), which can't survive moving between server connections.
    connect_args = {"prepare_threshold": None}
    if make_url(url).host not in LOCAL_HOSTS and "sslmode" not in url:
        connect_args["sslmode"] = "require"  # local docker Postgres (tests) has no TLS
    return create_engine(url, poolclass=NullPool, connect_args=connect_args)


engine = make_engine(config.DATABASE_URL)
SessionLocal = sessionmaker(engine, expire_on_commit=False)
