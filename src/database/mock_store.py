"""SQLite-backed operational store with realistic seeded delivery incidents."""

from datetime import datetime, timezone
from pathlib import Path
import sqlite3
import threading
from typing import Optional

from src.config import get_settings
from src.schemas import (
    AuditTrailEntry,
    CustomerProfile,
    DeliveryTelemetry,
    OrderItem,
    OrderRecord,
)


def _build_seed_orders() -> list[OrderRecord]:
    """Return the five baseline orders covering normal, edge, and risk scenarios."""
    now_iso = datetime.now(timezone.utc).isoformat()
    return [
        OrderRecord(
            order_id="PED-101",
            restaurant_name="Burger Artesanal Prime",
            status="IN_DELIVERY",
            total_amount=68.00,
            payment_status="APPROVED",
            cancel_reason=None,
            items=[
                OrderItem(
                    item_id="ITEM-101-1",
                    name="Smash Burger Duplo Artesanal",
                    quantity=1,
                    unit_price=52.00,
                ),
                OrderItem(
                    item_id="ITEM-101-2",
                    name="Batata Rústica com Alecrim",
                    quantity=1,
                    unit_price=16.00,
                ),
            ],
            telemetry=DeliveryTelemetry(
                courier_name="Carlos Mendes",
                courier_status="STOPPED",
                stopped_duration_minutes=15,
                weather_condition="HEAVY_RAIN",
                eta_remaining_minutes=20,
                delay_minutes=45,
                delivered_minutes_ago=None,
                gps_coordinates="-23.5614,-46.6559",
            ),
            customer=CustomerProfile(
                customer_id="CUST-001",
                name="Mariana Oliveira",
                tier="CLUB",
                total_orders=47,
                prior_refund_requests_30d=0,
                is_recurrent=True,
                risk_flag=False,
            ),
            created_at=now_iso,
        ),
        OrderRecord(
            order_id="PED-102",
            restaurant_name="Poke House Vila Madalena",
            status="IN_DELIVERY",
            total_amount=42.00,
            payment_status="APPROVED",
            cancel_reason=None,
            items=[
                OrderItem(
                    item_id="ITEM-102-1",
                    name="Poke Salmão Clássico",
                    quantity=1,
                    unit_price=42.00,
                ),
            ],
            telemetry=DeliveryTelemetry(
                courier_name="Rafael Souza",
                courier_status="IN_TRANSIT",
                stopped_duration_minutes=0,
                weather_condition="CLEAR",
                eta_remaining_minutes=10,
                delay_minutes=0,
                delivered_minutes_ago=None,
                gps_coordinates="-23.5505,-46.6889",
            ),
            customer=CustomerProfile(
                customer_id="CUST-002",
                name="Lucas Ferreira",
                tier="STANDARD",
                total_orders=12,
                prior_refund_requests_30d=0,
                is_recurrent=True,
                risk_flag=False,
            ),
            created_at=now_iso,
        ),
        OrderRecord(
            order_id="PED-103",
            restaurant_name="Cantina Bella Pasta",
            status="DELIVERED",
            total_amount=55.00,
            payment_status="APPROVED",
            cancel_reason=None,
            items=[
                OrderItem(
                    item_id="ITEM-103-1",
                    name="Parmegiana de Frango Executivo",
                    quantity=1,
                    unit_price=43.00,
                ),
                OrderItem(
                    item_id="ITEM-103-2",
                    name="Refrigerante Lata 350ml",
                    quantity=1,
                    unit_price=12.00,
                ),
            ],
            telemetry=DeliveryTelemetry(
                courier_name="Fernanda Lima",
                courier_status="DELIVERED",
                stopped_duration_minutes=0,
                weather_condition="CLEAR",
                eta_remaining_minutes=0,
                delay_minutes=0,
                delivered_minutes_ago=10,
                gps_coordinates="-23.5732,-46.6417",
            ),
            customer=CustomerProfile(
                customer_id="CUST-003",
                name="Beatriz Santos",
                tier="CLUB",
                total_orders=29,
                prior_refund_requests_30d=1,
                is_recurrent=True,
                risk_flag=False,
            ),
            created_at=now_iso,
        ),
        OrderRecord(
            order_id="PED-104",
            restaurant_name="Pizzaria Forno a Lenha",
            status="CANCELLED_BY_RESTAURANT",
            total_amount=89.00,
            payment_status="APPROVED",
            cancel_reason="Falta de ingrediente essencial na cozinha",
            items=[
                OrderItem(
                    item_id="ITEM-104-1",
                    name="Pizza Grande Margherita Especial",
                    quantity=1,
                    unit_price=75.00,
                ),
                OrderItem(
                    item_id="ITEM-104-2",
                    name="Borda Recheada Catupiry",
                    quantity=1,
                    unit_price=14.00,
                ),
            ],
            telemetry=DeliveryTelemetry(
                courier_name=None,
                courier_status="UNASSIGNED",
                stopped_duration_minutes=0,
                weather_condition="CLEAR",
                eta_remaining_minutes=None,
                delay_minutes=0,
                delivered_minutes_ago=None,
                gps_coordinates=None,
            ),
            customer=CustomerProfile(
                customer_id="CUST-004",
                name="Pedro Henrique",
                tier="STANDARD",
                total_orders=8,
                prior_refund_requests_30d=0,
                is_recurrent=False,
                risk_flag=False,
            ),
            created_at=now_iso,
        ),
        OrderRecord(
            order_id="PED-105",
            restaurant_name="Sakura Sushi Lounge",
            status="DELIVERED",
            total_amount=150.00,
            payment_status="APPROVED",
            cancel_reason=None,
            items=[
                OrderItem(
                    item_id="ITEM-105-1",
                    name="Combinado Sushi Premium 40 Peças",
                    quantity=1,
                    unit_price=150.00,
                ),
            ],
            telemetry=DeliveryTelemetry(
                courier_name="Diego Martins",
                courier_status="DELIVERED",
                stopped_duration_minutes=0,
                weather_condition="CLEAR",
                eta_remaining_minutes=0,
                delay_minutes=0,
                delivered_minutes_ago=120,
                gps_coordinates="-23.5874,-46.6576",
            ),
            customer=CustomerProfile(
                customer_id="CUST-005",
                name="Rodrigo Almeida",
                tier="STANDARD",
                total_orders=6,
                prior_refund_requests_30d=4,
                is_recurrent=False,
                risk_flag=True,
            ),
            created_at=now_iso,
        ),
    ]


class MockOrderStore:
    """Thread-safe SQLite store for orders and resolution audit entries."""

    def __init__(self, db_path: Optional[str] = None) -> None:
        resolved_path = db_path or get_settings().sqlite_db_path
        if resolved_path != ":memory:":
            Path(resolved_path).parent.mkdir(parents=True, exist_ok=True)
        self._db_path = resolved_path
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(self._db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._init_schema()
        self._seed_if_empty()

    def _init_schema(self) -> None:
        with self._lock:
            cursor = self._conn.cursor()
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS orders (
                    order_id TEXT PRIMARY KEY,
                    restaurant_name TEXT NOT NULL,
                    status TEXT NOT NULL,
                    total_amount REAL NOT NULL,
                    payment_status TEXT NOT NULL,
                    cancel_reason TEXT,
                    payload_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
                """
            )
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS resolutions_audit (
                    audit_id TEXT PRIMARY KEY,
                    timestamp TEXT NOT NULL,
                    order_id TEXT NOT NULL,
                    action_type TEXT NOT NULL,
                    amount REAL NOT NULL,
                    justification TEXT NOT NULL,
                    guardrail_triggered INTEGER NOT NULL,
                    raw_requested_action TEXT,
                    raw_requested_amount REAL
                )
                """
            )
            self._conn.commit()

    def _seed_if_empty(self) -> None:
        with self._lock:
            row = self._conn.execute("SELECT COUNT(*) AS cnt FROM orders").fetchone()
            if row and row["cnt"] == 0:
                self._insert_seed_orders()

    def _insert_seed_orders(self) -> None:
        now_iso = datetime.now(timezone.utc).isoformat()
        for order in _build_seed_orders():
            self._conn.execute(
                """
                INSERT INTO orders (
                    order_id, restaurant_name, status, total_amount,
                    payment_status, cancel_reason, payload_json, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    order.order_id,
                    order.restaurant_name,
                    order.status,
                    order.total_amount,
                    order.payment_status,
                    order.cancel_reason,
                    order.model_dump_json(),
                    order.created_at,
                    now_iso,
                ),
            )
        self._conn.commit()

    def reset(self) -> None:
        """Clear all tables and re-seed baseline orders."""
        with self._lock:
            self._conn.execute("DELETE FROM resolutions_audit")
            self._conn.execute("DELETE FROM orders")
            self._insert_seed_orders()

    def get_order_by_id(self, order_id: str) -> Optional[OrderRecord]:
        """Fetch an order record by exact or case-normalized ID."""
        normalized_id = order_id.strip().upper()
        with self._lock:
            row = self._conn.execute(
                "SELECT payload_json FROM orders WHERE UPPER(order_id) = ?",
                (normalized_id,),
            ).fetchone()
        if row is None:
            return None
        return OrderRecord.model_validate_json(row["payload_json"])

    def list_all_orders(self) -> list[OrderRecord]:
        """Return all orders ordered by order_id."""
        with self._lock:
            rows = self._conn.execute(
                "SELECT payload_json FROM orders ORDER BY order_id ASC"
            ).fetchall()
        return [OrderRecord.model_validate_json(r["payload_json"]) for r in rows]

    def save_audit_entry(self, entry: AuditTrailEntry) -> AuditTrailEntry:
        """Persist an immutable resolution audit record."""
        with self._lock:
            self._conn.execute(
                """
                INSERT INTO resolutions_audit (
                    audit_id, timestamp, order_id, action_type, amount,
                    justification, guardrail_triggered, raw_requested_action, raw_requested_amount
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    entry.audit_id,
                    entry.timestamp,
                    entry.order_id,
                    entry.action_type,
                    entry.amount,
                    entry.justification,
                    1 if entry.guardrail_triggered else 0,
                    entry.raw_requested_action,
                    entry.raw_requested_amount,
                ),
            )
            self._conn.commit()
        return entry

    def get_audit_entries(self, order_id: Optional[str] = None) -> list[AuditTrailEntry]:
        """Retrieve audit records, optionally filtered by order_id."""
        with self._lock:
            if order_id is not None:
                rows = self._conn.execute(
                    """
                    SELECT * FROM resolutions_audit
                    WHERE UPPER(order_id) = ?
                    ORDER BY timestamp ASC
                    """,
                    (order_id.strip().upper(),),
                ).fetchall()
            else:
                rows = self._conn.execute(
                    "SELECT * FROM resolutions_audit ORDER BY timestamp ASC"
                ).fetchall()

        return [
            AuditTrailEntry(
                audit_id=r["audit_id"],
                timestamp=r["timestamp"],
                order_id=r["order_id"],
                action_type=r["action_type"],
                amount=r["amount"],
                justification=r["justification"],
                guardrail_triggered=bool(r["guardrail_triggered"]),
                raw_requested_action=r["raw_requested_action"],
                raw_requested_amount=r["raw_requested_amount"],
            )
            for r in rows
        ]

    def close(self) -> None:
        """Close the underlying SQLite connection."""
        with self._lock:
            self._conn.close()


_store_lock = threading.RLock()
_store_instance: Optional[MockOrderStore] = None


def get_store() -> MockOrderStore:
    """Return the singleton MockOrderStore instance."""
    global _store_instance
    with _store_lock:
        if _store_instance is None:
            _store_instance = MockOrderStore()
        return _store_instance


def reset_store(db_path: Optional[str] = None) -> MockOrderStore:
    """Reset the store to baseline seeded state (optionally targeting a new db_path)."""
    global _store_instance
    with _store_lock:
        if db_path is not None:
            if _store_instance is not None:
                _store_instance.close()
            _store_instance = MockOrderStore(db_path=db_path)
        else:
            store = get_store()
            store.reset()
        return _store_instance


def get_order_by_id(order_id: str) -> Optional[OrderRecord]:
    """Convenience wrapper to look up an order from the active store."""
    return get_store().get_order_by_id(order_id)


def list_all_orders() -> list[OrderRecord]:
    """Convenience wrapper to list all orders from the active store."""
    return get_store().list_all_orders()


def save_audit_entry(entry: AuditTrailEntry) -> AuditTrailEntry:
    """Convenience wrapper to persist an audit entry in the active store."""
    return get_store().save_audit_entry(entry)


def get_audit_entries(order_id: Optional[str] = None) -> list[AuditTrailEntry]:
    """Convenience wrapper to list audit entries from the active store."""
    return get_store().get_audit_entries(order_id=order_id)
