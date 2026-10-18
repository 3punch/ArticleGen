"""Atomic persisted checks, holds, and settlements for hard project budgets."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from decimal import Decimal
from typing import Any, Mapping
from uuid import UUID, uuid4

from sqlalchemy import select
from sqlalchemy.orm import Session, sessionmaker

from projectforge_api.orm import AgentRunRow, BudgetLedgerRow, BudgetReservationRow


class BudgetExceeded(RuntimeError):
    def __init__(self, project_id: UUID, message: str) -> None:
        super().__init__(message)
        self.project_id = project_id


class BudgetGuard:
    """Use row-locked ledger totals so concurrent provider calls cannot overspend a hold."""

    def __init__(self, session_factory: sessionmaker[Session]) -> None:
        self._session_factory = session_factory

    def check(
        self,
        project_id: UUID,
        *,
        reserve_tokens: int = 0,
        reserve_spend: Decimal = Decimal("0"),
    ) -> None:
        if reserve_tokens < 0 or reserve_spend < 0:
            raise ValueError("budget reservations must be nonnegative")
        with self._session_factory() as session:
            ledger = session.scalar(
                select(BudgetLedgerRow)
                .where(BudgetLedgerRow.project_id == project_id)
                .with_for_update()
            )
            if ledger is None:
                raise BudgetExceeded(project_id, "persisted budget ledger is missing")
            projected_tokens = ledger.used_tokens + ledger.reserved_tokens + reserve_tokens
            projected_spend = ledger.used_spend + ledger.reserved_spend + reserve_spend
            if projected_tokens > ledger.token_cap:
                raise BudgetExceeded(
                    project_id,
                    f"token budget exceeded: {projected_tokens} > {ledger.token_cap}",
                )
            if projected_spend > ledger.spend_cap:
                raise BudgetExceeded(
                    project_id,
                    f"spend budget exceeded: {projected_spend} > {ledger.spend_cap}",
                )

    def reserve(
        self,
        project_id: UUID,
        workflow_id: UUID,
        *,
        tokens: int,
        spend: Decimal,
        ttl_seconds: int,
    ) -> UUID:
        """Persist a provider-call hold before sending any request outside the process."""
        if tokens <= 0 or spend < 0 or ttl_seconds <= 0:
            raise ValueError("provider reservations require positive tokens and TTL, and nonnegative spend")
        now = datetime.now(timezone.utc)
        reservation_id = uuid4()
        exceeded_message: str | None = None
        with self._session_factory() as session, session.begin():
            ledger = session.scalar(
                select(BudgetLedgerRow)
                .where(BudgetLedgerRow.project_id == project_id)
                .with_for_update()
            )
            if ledger is None:
                raise BudgetExceeded(project_id, "persisted budget ledger is missing")
            self._expire_old_holds(session, ledger, now)
            projected_tokens = ledger.used_tokens + ledger.reserved_tokens + tokens
            projected_spend = ledger.used_spend + ledger.reserved_spend + spend
            if projected_tokens > ledger.token_cap:
                exceeded_message = (
                    f"token budget exceeded before provider call: {projected_tokens} > {ledger.token_cap}"
                )
            elif projected_spend > ledger.spend_cap:
                exceeded_message = (
                    f"spend budget exceeded before provider call: {projected_spend} > {ledger.spend_cap}"
                )
            else:
                ledger.reserved_tokens += tokens
                ledger.reserved_spend += spend
                session.add(BudgetReservationRow(
                    id=reservation_id,
                    project_id=project_id,
                    workflow_id=workflow_id,
                    reserved_tokens=tokens,
                    reserved_spend=spend,
                    state="reserved",
                    expires_at=now + timedelta(seconds=ttl_seconds),
                    created_at=now,
                ))
        if exceeded_message:
            raise BudgetExceeded(project_id, exceeded_message)
        return reservation_id

    def release(self, reservation_id: UUID) -> None:
        """Release a hold only when preflight proves the provider was not called."""
        now = datetime.now(timezone.utc)
        with self._session_factory() as session, session.begin():
            reservation = session.scalar(
                select(BudgetReservationRow).where(
                    BudgetReservationRow.id == reservation_id,
                ).with_for_update()
            )
            if reservation is None or reservation.state != "reserved":
                return
            ledger = session.scalar(
                select(BudgetLedgerRow)
                .where(BudgetLedgerRow.project_id == reservation.project_id)
                .with_for_update()
            )
            if ledger is None:
                raise BudgetExceeded(reservation.project_id, "persisted budget ledger is missing")
            ledger.reserved_tokens = max(0, ledger.reserved_tokens - reservation.reserved_tokens)
            ledger.reserved_spend = max(
                Decimal("0"), ledger.reserved_spend - reservation.reserved_spend,
            )
            reservation.state = "released"
            reservation.completed_at = now

    def settle_provider_call(
        self,
        reservation_id: UUID,
        *,
        tokens: int,
        spend: Decimal,
        agent_run_values: Mapping[str, Any],
    ) -> UUID:
        """Atomically swap a hold for actual usage and persist the audit-run row."""
        if tokens < 0 or spend < 0:
            raise ValueError("recorded budget usage must be nonnegative")
        project_id = agent_run_values["project_id"]
        run_id = agent_run_values.get("id") or uuid4()
        exceeded_message: str | None = None
        with self._session_factory() as session, session.begin():
            idempotency_key = agent_run_values.get("idempotency_key")
            if idempotency_key:
                existing_run = session.scalar(
                    select(AgentRunRow).where(AgentRunRow.idempotency_key == idempotency_key)
                )
                if existing_run is not None:
                    return existing_run.id
            reservation = session.scalar(
                select(BudgetReservationRow).where(
                    BudgetReservationRow.id == reservation_id,
                ).with_for_update()
            )
            if reservation is None or reservation.state != "reserved":
                raise BudgetExceeded(project_id, "provider budget reservation is missing or no longer active")
            if reservation.project_id != project_id:
                raise BudgetExceeded(project_id, "provider budget reservation belongs to another project")
            ledger = session.scalar(
                select(BudgetLedgerRow)
                .where(BudgetLedgerRow.project_id == project_id)
                .with_for_update()
            )
            if ledger is None:
                raise BudgetExceeded(project_id, "persisted budget ledger is missing")
            ledger.reserved_tokens = max(0, ledger.reserved_tokens - reservation.reserved_tokens)
            ledger.reserved_spend = max(Decimal("0"), ledger.reserved_spend - reservation.reserved_spend)
            ledger.used_tokens += tokens
            ledger.used_spend += spend
            reservation.actual_tokens = tokens
            reservation.actual_spend = spend
            reservation.state = "settled"
            reservation.completed_at = datetime.now(timezone.utc)
            session.add(AgentRunRow(**{**dict(agent_run_values), "id": run_id,
                                       "budget_reservation_id": reservation_id,
                                       "tokens_used": tokens, "cost": spend}))
            projected_tokens = ledger.used_tokens + ledger.reserved_tokens
            projected_spend = ledger.used_spend + ledger.reserved_spend
            if ledger.used_tokens > ledger.token_cap or projected_tokens > ledger.token_cap:
                exceeded_message = (
                    f"actual token usage exceeded budget: {projected_tokens} > {ledger.token_cap}"
                )
            elif ledger.used_spend > ledger.spend_cap or projected_spend > ledger.spend_cap:
                exceeded_message = (
                    f"actual spend exceeded budget: {projected_spend} > {ledger.spend_cap}"
                )
        if exceeded_message:
            raise BudgetExceeded(project_id, exceeded_message)
        return run_id

    def record_failed_provider_call(self, *, agent_run_values: Mapping[str, Any]) -> UUID:
        """Audit an ambiguous failed provider call and keep its hold until TTL expiry.

        A timeout/error can happen after the remote endpoint consumed tokens. Releasing
        the hold here would make a retry capable of overspending; expire/reconcile it
        explicitly instead.
        """
        run_id = agent_run_values.get("id") or uuid4()
        with self._session_factory() as session, session.begin():
            idempotency_key = agent_run_values.get("idempotency_key")
            if idempotency_key:
                existing_run = session.scalar(
                    select(AgentRunRow).where(AgentRunRow.idempotency_key == idempotency_key)
                )
                if existing_run is not None:
                    return existing_run.id
            session.add(AgentRunRow(**{**dict(agent_run_values), "id": run_id}))
        return run_id

    def record_usage(self, project_id: UUID, *, tokens: int, spend: Decimal) -> None:
        """Record non-provider calls (the retained deterministic role stubs)."""
        if tokens < 0 or spend < 0:
            raise ValueError("recorded budget usage must be nonnegative")
        exceeded_message: str | None = None
        with self._session_factory() as session, session.begin():
            ledger = session.scalar(
                select(BudgetLedgerRow)
                .where(BudgetLedgerRow.project_id == project_id)
                .with_for_update()
            )
            if ledger is None:
                raise BudgetExceeded(project_id, "persisted budget ledger is missing")
            ledger.used_tokens += tokens
            ledger.used_spend += spend
            projected_tokens = ledger.used_tokens + ledger.reserved_tokens
            projected_spend = ledger.used_spend + ledger.reserved_spend
            if projected_tokens > ledger.token_cap:
                exceeded_message = (
                    f"actual token usage exceeded budget: {projected_tokens} > {ledger.token_cap}"
                )
            elif projected_spend > ledger.spend_cap:
                exceeded_message = (
                    f"actual spend exceeded budget: {projected_spend} > {ledger.spend_cap}"
                )
        if exceeded_message:
            raise BudgetExceeded(project_id, exceeded_message)

    @staticmethod
    def _expire_old_holds(session: Session, ledger: BudgetLedgerRow, now: datetime) -> None:
        expired = session.scalars(
            select(BudgetReservationRow).where(
                BudgetReservationRow.project_id == ledger.project_id,
                BudgetReservationRow.state == "reserved",
                BudgetReservationRow.expires_at <= now,
            ).with_for_update()
        ).all()
        for reservation in expired:
            ledger.reserved_tokens = max(0, ledger.reserved_tokens - reservation.reserved_tokens)
            ledger.reserved_spend = max(
                Decimal("0"), ledger.reserved_spend - reservation.reserved_spend,
            )
            reservation.state = "expired"
            reservation.completed_at = now
