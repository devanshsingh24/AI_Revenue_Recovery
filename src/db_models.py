from __future__ import annotations
from datetime import datetime
from sqlalchemy import String, Float, Integer, DateTime, JSON, ForeignKey
from sqlalchemy.orm import Mapped, mapped_column
from src.db import Base


class Customer(Base):
    __tablename__ = "customers"
    customer_id: Mapped[str] = mapped_column(String, primary_key=True)
    segment: Mapped[str] = mapped_column(String, default="individual")
    tenure_months: Mapped[int] = mapped_column(Integer, default=0)
    past_payment_success_rate: Mapped[float] = mapped_column(Float, default=0.5)
    historical_engagement_score: Mapped[float] = mapped_column(Float, default=0.5)
    previous_failed_payments: Mapped[int] = mapped_column(Integer, default=0)
    previous_recovered_payments: Mapped[int] = mapped_column(Integer, default=0)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class Payment(Base):
    __tablename__ = "payments"
    payment_id: Mapped[str] = mapped_column(String, primary_key=True)
    customer_id: Mapped[str] = mapped_column(String, ForeignKey("customers.customer_id"), nullable=True)
    order_id: Mapped[str] = mapped_column(String, nullable=True)
    amount: Mapped[float] = mapped_column(Float)
    currency: Mapped[str] = mapped_column(String, default="INR")
    decline_reason: Mapped[str] = mapped_column(String, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


class RecoveryAttempt(Base):
    __tablename__ = "recovery_attempts"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    payment_id: Mapped[str] = mapped_column(String, ForeignKey("payments.payment_id"), nullable=True)
    ml_suggested_action: Mapped[str] = mapped_column(String, nullable=True)
    final_action: Mapped[str] = mapped_column(String, nullable=True)
    policy_status: Mapped[str] = mapped_column(String, nullable=True)
    policy_reason: Mapped[str] = mapped_column(String, nullable=True)
    execution_status: Mapped[str] = mapped_column(String, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


class AuditLog(Base):
    __tablename__ = "audit_logs"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    event_id: Mapped[str] = mapped_column(String, nullable=True, index=True)
    record_id: Mapped[str] = mapped_column(String, nullable=True)
    payment_id: Mapped[str] = mapped_column(String, nullable=True)
    record: Mapped[dict] = mapped_column(JSON)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


class WebhookEvent(Base):
    __tablename__ = "webhook_events"
    event_id: Mapped[str] = mapped_column(String, primary_key=True)
    response: Mapped[dict] = mapped_column(JSON)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)