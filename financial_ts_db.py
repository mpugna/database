"""
Financial Time Series Database Interface

A SQLite-based database for storing and managing financial time series data,
including instruments, fields, aliases, and data provider configurations.

Requires Python 3.9+

Author: Claude
"""

from __future__ import annotations  # Enable PEP 604 type hints on Python 3.9

import sqlite3
import json
import logging
from dataclasses import dataclass, field, asdict
from datetime import datetime, date
from enum import Enum
from pathlib import Path
from typing import Optional, Any
from contextlib import contextmanager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# =============================================================================
# Enums and Constants
# =============================================================================

class Frequency(Enum):
    """Supported data frequencies."""
    TICK = "tick"
    SECOND = "second"
    MINUTE = "minute"
    HOURLY = "hourly"
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"
    QUARTERLY = "quarterly"
    YEARLY = "yearly"


class InstrumentType(Enum):
    """Types of financial instruments."""
    STOCK = "stock"
    INDEX = "index"
    ETF = "etf"
    BOND = "bond"
    COMMODITY = "commodity"
    CURRENCY = "currency"
    ECONOMIC_INDICATOR = "economic_indicator"
    DERIVATIVE = "derivative"
    MUTUAL_FUND = "mutual_fund"
    CRYPTO = "crypto"
    OTHER = "other"


class DataProvider(Enum):
    """Supported data providers."""
    BLOOMBERG = "bloomberg"
    REUTERS = "reuters"
    YAHOO_FINANCE = "yahoo_finance"
    ALPHA_VANTAGE = "alpha_vantage"
    QUANDL = "quandl"
    FRED = "fred"
    CUSTOM = "custom"


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class Instrument:
    """Represents a financial instrument."""
    id: Optional[int] = None
    ticker: str = ""
    name: str = ""
    instrument_type: InstrumentType = InstrumentType.OTHER
    description: str = ""
    currency: str = "USD"
    exchange: str = ""
    metadata: dict = field(default_factory=dict)
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None


@dataclass
class InstrumentField:
    """Represents a field for an instrument at a specific frequency."""
    id: Optional[int] = None
    instrument_id: int = 0
    field_name: str = ""
    frequency: Frequency = Frequency.DAILY
    description: str = ""
    unit: str = ""
    # Alias fields - if set, this field points to another instrument's field
    alias_instrument_id: Optional[int] = None
    alias_field_id: Optional[int] = None
    metadata: dict = field(default_factory=dict)
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None


@dataclass
class ProviderConfig:
    """Configuration for retrieving data from a provider."""
    id: Optional[int] = None
    field_id: int = 0
    provider: DataProvider = DataProvider.CUSTOM
    # Provider-specific configuration (e.g., Bloomberg ticker, API endpoint, etc.)
    config: dict = field(default_factory=dict)
    is_active: bool = True
    priority: int = 0  # Lower number = higher priority
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None


@dataclass
class TimeSeriesPoint:
    """A single point in a time series."""
    id: Optional[int] = None
    field_id: int = 0
    timestamp: datetime = field(default_factory=datetime.now)
    value: float = 0.0
    metadata: dict = field(default_factory=dict)  # For additional info like volume, etc.


@dataclass
class DeletionImpact:
    """Describes the impact of a deletion operation."""
    target_type: str = ""
    target_id: int = 0
    target_name: str = ""
    fields_to_delete: list = field(default_factory=list)
    aliases_to_delete: list = field(default_factory=list)
    provider_configs_to_delete: list = field(default_factory=list)
    time_series_points_to_delete: int = 0
    warnings: list = field(default_factory=list)


# =============================================================================
# Database Schema
# =============================================================================

SCHEMA_SQL = """
-- Instruments table
CREATE TABLE IF NOT EXISTS instruments (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ticker TEXT NOT NULL UNIQUE,
    name TEXT NOT NULL,
    instrument_type TEXT NOT NULL,
    description TEXT DEFAULT '',
    currency TEXT DEFAULT 'USD',
    exchange TEXT DEFAULT '',
    metadata TEXT DEFAULT '{}',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Instrument fields table
CREATE TABLE IF NOT EXISTS instrument_fields (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    instrument_id INTEGER NOT NULL,
    field_name TEXT NOT NULL,
    frequency TEXT NOT NULL,
    description TEXT DEFAULT '',
    unit TEXT DEFAULT '',
    alias_instrument_id INTEGER,
    alias_field_id INTEGER,
    metadata TEXT DEFAULT '{}',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (instrument_id) REFERENCES instruments(id) ON DELETE CASCADE,
    FOREIGN KEY (alias_instrument_id) REFERENCES instruments(id) ON DELETE SET NULL,
    FOREIGN KEY (alias_field_id) REFERENCES instrument_fields(id) ON DELETE SET NULL,
    UNIQUE(instrument_id, field_name, frequency)
);

-- Provider configurations table
CREATE TABLE IF NOT EXISTS provider_configs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    field_id INTEGER NOT NULL,
    provider TEXT NOT NULL,
    config TEXT DEFAULT '{}',
    is_active INTEGER DEFAULT 1,
    priority INTEGER DEFAULT 0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (field_id) REFERENCES instrument_fields(id) ON DELETE CASCADE,
    UNIQUE(field_id, provider)
);

-- Time series data table
CREATE TABLE IF NOT EXISTS time_series_data (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    field_id INTEGER NOT NULL,
    timestamp TIMESTAMP NOT NULL,
    value REAL NOT NULL,
    metadata TEXT DEFAULT '{}',
    FOREIGN KEY (field_id) REFERENCES instrument_fields(id) ON DELETE CASCADE,
    UNIQUE(field_id, timestamp)
);

-- Indexes for performance
CREATE INDEX IF NOT EXISTS idx_instruments_ticker ON instruments(ticker);
CREATE INDEX IF NOT EXISTS idx_instruments_type ON instruments(instrument_type);
CREATE INDEX IF NOT EXISTS idx_fields_instrument ON instrument_fields(instrument_id);
CREATE INDEX IF NOT EXISTS idx_fields_alias ON instrument_fields(alias_field_id);
CREATE INDEX IF NOT EXISTS idx_provider_field ON provider_configs(field_id);
CREATE INDEX IF NOT EXISTS idx_timeseries_field ON time_series_data(field_id);
CREATE INDEX IF NOT EXISTS idx_timeseries_timestamp ON time_series_data(timestamp);
CREATE INDEX IF NOT EXISTS idx_timeseries_field_timestamp ON time_series_data(field_id, timestamp);

-- Trigger to update timestamps
CREATE TRIGGER IF NOT EXISTS update_instrument_timestamp
AFTER UPDATE ON instruments
BEGIN
    UPDATE instruments SET updated_at = CURRENT_TIMESTAMP WHERE id = NEW.id;
END;

CREATE TRIGGER IF NOT EXISTS update_field_timestamp
AFTER UPDATE ON instrument_fields
BEGIN
    UPDATE instrument_fields SET updated_at = CURRENT_TIMESTAMP WHERE id = NEW.id;
END;

CREATE TRIGGER IF NOT EXISTS update_provider_timestamp
AFTER UPDATE ON provider_configs
BEGIN
    UPDATE provider_configs SET updated_at = CURRENT_TIMESTAMP WHERE id = NEW.id;
END;
"""


# =============================================================================
# Main Database Class
# =============================================================================

class FinancialTimeSeriesDB:
    """
    Interface for the Financial Time Series SQLite Database.

    This class provides methods to:
    - Manage instruments (add, update, delete, query)
    - Manage fields per instrument at different frequencies
    - Handle field aliases (pointing to other instruments' fields)
    - Configure data provider settings
    - Store and retrieve time series data
    - Preview deletion impacts before executing

    Example:
        db = FinancialTimeSeriesDB("my_database.db")

        # Add an instrument
        apple = db.add_instrument(
            ticker="AAPL",
            name="Apple Inc.",
            instrument_type=InstrumentType.STOCK
        )

        # Add a field
        price_field = db.add_field(
            instrument_id=apple.id,
            field_name="PRICE",
            frequency=Frequency.DAILY
        )

        # Configure provider
        db.add_provider_config(
            field_id=price_field.id,
            provider=DataProvider.BLOOMBERG,
            config={"ticker": "AAPL US Equity", "field": "PX_LAST"}
        )
    """

    def __init__(self, db_path: str | Path = ":memory:"):
        """
        Initialize the database connection.

        Args:
            db_path: Path to the SQLite database file. Use ":memory:" for in-memory database.
        """
        self.db_path = Path(db_path) if db_path != ":memory:" else db_path
        self._is_memory = db_path == ":memory:"
        self._persistent_conn: Optional[sqlite3.Connection] = None

        # For in-memory databases, we need a persistent connection
        if self._is_memory:
            self._persistent_conn = self._create_connection()

        self._initialize_db()

    def _create_connection(self) -> sqlite3.Connection:
        """Create a new database connection with proper settings."""
        conn = sqlite3.connect(
            self.db_path if isinstance(self.db_path, str) else str(self.db_path),
            detect_types=sqlite3.PARSE_DECLTYPES | sqlite3.PARSE_COLNAMES
        )
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        return conn

    def _initialize_db(self) -> None:
        """Initialize the database schema."""
        with self._get_connection() as conn:
            conn.executescript(SCHEMA_SQL)
            conn.commit()
        logger.info(f"Database initialized: {self.db_path}")

    @contextmanager
    def _get_connection(self):
        """Get a database connection with proper settings."""
        if self._is_memory and self._persistent_conn:
            # For in-memory databases, always use the persistent connection
            yield self._persistent_conn
        else:
            # For file-based databases, create new connections
            conn = self._create_connection()
            try:
                yield conn
            finally:
                conn.close()

    # =========================================================================
    # Instrument Operations
    # =========================================================================

    def add_instrument(
        self,
        ticker: str,
        name: str,
        instrument_type: InstrumentType,
        description: str = "",
        currency: str = "USD",
        exchange: str = "",
        metadata: Optional[dict] = None
    ) -> Instrument:
        """
        Add a new instrument to the database.

        Args:
            ticker: Unique ticker symbol (e.g., "AAPL", "SPX")
            name: Full name of the instrument
            instrument_type: Type of instrument (stock, index, etc.)
            description: Optional description
            currency: Currency code (default: USD)
            exchange: Exchange where traded
            metadata: Additional metadata as dictionary

        Returns:
            The created Instrument object with its ID

        Raises:
            sqlite3.IntegrityError: If ticker already exists
        """
        metadata = metadata or {}

        with self._get_connection() as conn:
            cursor = conn.execute(
                """
                INSERT INTO instruments (ticker, name, instrument_type, description,
                                         currency, exchange, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (ticker, name, instrument_type.value, description,
                 currency, exchange, json.dumps(metadata))
            )
            conn.commit()

            instrument = self.get_instrument(cursor.lastrowid)
            logger.info(f"Added instrument: {ticker} (ID: {instrument.id})")
            return instrument

    def get_instrument(self, instrument_id: int) -> Optional[Instrument]:
        """Get an instrument by ID."""
        with self._get_connection() as conn:
            row = conn.execute(
                "SELECT * FROM instruments WHERE id = ?",
                (instrument_id,)
            ).fetchone()

            if row:
                return self._row_to_instrument(row)
            return None

    def get_instrument_by_ticker(self, ticker: str) -> Optional[Instrument]:
        """Get an instrument by ticker symbol."""
        with self._get_connection() as conn:
            row = conn.execute(
                "SELECT * FROM instruments WHERE ticker = ?",
                (ticker,)
            ).fetchone()

            if row:
                return self._row_to_instrument(row)
            return None

    def list_instruments(
        self,
        instrument_type: Optional[InstrumentType] = None,
        search: Optional[str] = None
    ) -> list[Instrument]:
        """
        List instruments with optional filtering.

        Args:
            instrument_type: Filter by instrument type
            search: Search in ticker and name

        Returns:
            List of matching instruments
        """
        query = "SELECT * FROM instruments WHERE 1=1"
        params = []

        if instrument_type:
            query += " AND instrument_type = ?"
            params.append(instrument_type.value)

        if search:
            query += " AND (ticker LIKE ? OR name LIKE ?)"
            params.extend([f"%{search}%", f"%{search}%"])

        query += " ORDER BY ticker"

        with self._get_connection() as conn:
            rows = conn.execute(query, params).fetchall()
            return [self._row_to_instrument(row) for row in rows]

    def update_instrument(
        self,
        instrument_id: int,
        **kwargs
    ) -> Optional[Instrument]:
        """
        Update an instrument's attributes.

        Args:
            instrument_id: ID of the instrument to update
            **kwargs: Attributes to update (ticker, name, instrument_type, etc.)

        Returns:
            Updated Instrument or None if not found
        """
        allowed_fields = {'ticker', 'name', 'instrument_type', 'description',
                          'currency', 'exchange', 'metadata'}

        updates = {k: v for k, v in kwargs.items() if k in allowed_fields}
        if not updates:
            return self.get_instrument(instrument_id)

        # Handle special conversions
        if 'instrument_type' in updates and isinstance(updates['instrument_type'], InstrumentType):
            updates['instrument_type'] = updates['instrument_type'].value
        if 'metadata' in updates and isinstance(updates['metadata'], dict):
            updates['metadata'] = json.dumps(updates['metadata'])

        set_clause = ", ".join(f"{k} = ?" for k in updates.keys())
        values = list(updates.values()) + [instrument_id]

        with self._get_connection() as conn:
            conn.execute(
                f"UPDATE instruments SET {set_clause} WHERE id = ?",
                values
            )
            conn.commit()

        logger.info(f"Updated instrument ID: {instrument_id}")
        return self.get_instrument(instrument_id)

    def delete_instrument(
        self,
        instrument_id: int,
        dry_run: bool = False
    ) -> DeletionImpact:
        """
        Delete an instrument and all related data.

        This will:
        - Delete all fields of this instrument
        - Delete all provider configs for those fields
        - Delete all time series data for those fields
        - Set alias references to NULL and warn about affected aliases

        Args:
            instrument_id: ID of the instrument to delete
            dry_run: If True, only simulate and return impact without deleting

        Returns:
            DeletionImpact object describing what was/would be deleted
        """
        impact = self._calculate_instrument_deletion_impact(instrument_id)

        if dry_run:
            logger.info(f"DRY RUN - Would delete instrument ID: {instrument_id}")
            return impact

        # Log warnings about aliases that will be broken
        for warning in impact.warnings:
            logger.warning(warning)

        with self._get_connection() as conn:
            # The CASCADE will handle fields and their provider configs
            # But we need to handle the aliases pointing TO this instrument

            # First, get all fields that are aliases pointing to this instrument's fields
            alias_fields = conn.execute(
                """
                SELECT if2.id, if2.instrument_id, i.ticker, if2.field_name
                FROM instrument_fields if2
                JOIN instruments i ON if2.instrument_id = i.id
                WHERE if2.alias_instrument_id = ?
                """,
                (instrument_id,)
            ).fetchall()

            # Delete the alias fields (or set to NULL - choosing delete here for cleaner state)
            for alias in alias_fields:
                conn.execute(
                    "DELETE FROM instrument_fields WHERE id = ?",
                    (alias['id'],)
                )
                logger.warning(
                    f"Deleted alias field: {alias['ticker']}.{alias['field_name']} "
                    f"(was pointing to deleted instrument)"
                )

            # Now delete the instrument (CASCADE will handle the rest)
            conn.execute("DELETE FROM instruments WHERE id = ?", (instrument_id,))
            conn.commit()

        logger.info(f"Deleted instrument ID: {instrument_id}")
        return impact

    def _calculate_instrument_deletion_impact(self, instrument_id: int) -> DeletionImpact:
        """Calculate the impact of deleting an instrument."""
        instrument = self.get_instrument(instrument_id)
        if not instrument:
            return DeletionImpact(
                target_type="instrument",
                target_id=instrument_id,
                target_name="NOT FOUND",
                warnings=["Instrument not found"]
            )

        impact = DeletionImpact(
            target_type="instrument",
            target_id=instrument_id,
            target_name=f"{instrument.ticker} ({instrument.name})"
        )

        with self._get_connection() as conn:
            # Get fields that will be deleted
            fields = conn.execute(
                """
                SELECT id, field_name, frequency FROM instrument_fields
                WHERE instrument_id = ?
                """,
                (instrument_id,)
            ).fetchall()

            impact.fields_to_delete = [
                {"id": f['id'], "name": f['field_name'], "frequency": f['frequency']}
                for f in fields
            ]

            field_ids = [f['id'] for f in fields]

            if field_ids:
                placeholders = ",".join("?" * len(field_ids))

                # Get provider configs that will be deleted
                configs = conn.execute(
                    f"""
                    SELECT id, provider FROM provider_configs
                    WHERE field_id IN ({placeholders})
                    """,
                    field_ids
                ).fetchall()

                impact.provider_configs_to_delete = [
                    {"id": c['id'], "provider": c['provider']}
                    for c in configs
                ]

                # Count time series points
                count = conn.execute(
                    f"""
                    SELECT COUNT(*) as cnt FROM time_series_data
                    WHERE field_id IN ({placeholders})
                    """,
                    field_ids
                ).fetchone()

                impact.time_series_points_to_delete = count['cnt']

            # Find aliases that point to this instrument
            aliases = conn.execute(
                """
                SELECT if2.id, if2.field_name, if2.frequency, i.ticker
                FROM instrument_fields if2
                JOIN instruments i ON if2.instrument_id = i.id
                WHERE if2.alias_instrument_id = ?
                """,
                (instrument_id,)
            ).fetchall()

            impact.aliases_to_delete = [
                {"id": a['id'], "ticker": a['ticker'],
                 "field": a['field_name'], "frequency": a['frequency']}
                for a in aliases
            ]

            if impact.aliases_to_delete:
                for alias in impact.aliases_to_delete:
                    impact.warnings.append(
                        f"WARNING: Alias {alias['ticker']}.{alias['field']} ({alias['frequency']}) "
                        f"points to this instrument and will be deleted"
                    )

        return impact

    # =========================================================================
    # Field Operations
    # =========================================================================

    def add_field(
        self,
        instrument_id: int,
        field_name: str,
        frequency: Frequency,
        description: str = "",
        unit: str = "",
        alias_instrument_id: Optional[int] = None,
        alias_field_id: Optional[int] = None,
        metadata: Optional[dict] = None
    ) -> InstrumentField:
        """
        Add a new field to an instrument.

        Args:
            instrument_id: ID of the parent instrument
            field_name: Name of the field (e.g., "PRICE", "EPS", "TOTAL_RETURN")
            frequency: Data frequency
            description: Optional description
            unit: Unit of measurement
            alias_instrument_id: If this is an alias, the target instrument ID
            alias_field_id: If this is an alias, the target field ID
            metadata: Additional metadata

        Returns:
            The created InstrumentField object

        Raises:
            ValueError: If alias_instrument_id is set without alias_field_id or vice versa
            sqlite3.IntegrityError: If field already exists for this instrument/frequency
        """
        if (alias_instrument_id is None) != (alias_field_id is None):
            raise ValueError(
                "Both alias_instrument_id and alias_field_id must be set together, or neither"
            )

        metadata = metadata or {}

        with self._get_connection() as conn:
            cursor = conn.execute(
                """
                INSERT INTO instrument_fields
                (instrument_id, field_name, frequency, description, unit,
                 alias_instrument_id, alias_field_id, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (instrument_id, field_name, frequency.value, description, unit,
                 alias_instrument_id, alias_field_id, json.dumps(metadata))
            )
            conn.commit()

            field = self.get_field(cursor.lastrowid)
            logger.info(
                f"Added field: {field_name} ({frequency.value}) "
                f"for instrument ID: {instrument_id}"
            )
            return field

    def add_alias_field(
        self,
        instrument_id: int,
        field_name: str,
        frequency: Frequency,
        target_instrument_id: int,
        target_field_name: str,
        target_frequency: Optional[Frequency] = None,
        description: str = ""
    ) -> InstrumentField:
        """
        Convenience method to add an alias field that points to another instrument's field.

        Args:
            instrument_id: ID of the instrument to add the alias to
            field_name: Name for the alias field
            frequency: Frequency of the alias
            target_instrument_id: ID of the target instrument
            target_field_name: Name of the target field
            target_frequency: Frequency of target field (defaults to same as alias)
            description: Optional description

        Returns:
            The created alias field

        Raises:
            ValueError: If target field not found
        """
        target_frequency = target_frequency or frequency

        # Find the target field
        target_field = self.get_field_by_name(
            target_instrument_id, target_field_name, target_frequency
        )

        if not target_field:
            raise ValueError(
                f"Target field not found: instrument_id={target_instrument_id}, "
                f"field_name={target_field_name}, frequency={target_frequency.value}"
            )

        return self.add_field(
            instrument_id=instrument_id,
            field_name=field_name,
            frequency=frequency,
            description=description or f"Alias for {target_field_name}",
            alias_instrument_id=target_instrument_id,
            alias_field_id=target_field.id
        )

    def get_field(self, field_id: int) -> Optional[InstrumentField]:
        """Get a field by ID."""
        with self._get_connection() as conn:
            row = conn.execute(
                "SELECT * FROM instrument_fields WHERE id = ?",
                (field_id,)
            ).fetchone()

            if row:
                return self._row_to_field(row)
            return None

    def get_field_by_name(
        self,
        instrument_id: int,
        field_name: str,
        frequency: Frequency
    ) -> Optional[InstrumentField]:
        """Get a field by instrument ID, name, and frequency."""
        with self._get_connection() as conn:
            row = conn.execute(
                """
                SELECT * FROM instrument_fields
                WHERE instrument_id = ? AND field_name = ? AND frequency = ?
                """,
                (instrument_id, field_name, frequency.value)
            ).fetchone()

            if row:
                return self._row_to_field(row)
            return None

    def list_fields(
        self,
        instrument_id: Optional[int] = None,
        frequency: Optional[Frequency] = None,
        include_aliases: bool = True
    ) -> list[InstrumentField]:
        """
        List fields with optional filtering.

        Args:
            instrument_id: Filter by instrument
            frequency: Filter by frequency
            include_aliases: Whether to include alias fields

        Returns:
            List of matching fields
        """
        query = "SELECT * FROM instrument_fields WHERE 1=1"
        params = []

        if instrument_id is not None:
            query += " AND instrument_id = ?"
            params.append(instrument_id)

        if frequency:
            query += " AND frequency = ?"
            params.append(frequency.value)

        if not include_aliases:
            query += " AND alias_field_id IS NULL"

        query += " ORDER BY instrument_id, field_name, frequency"

        with self._get_connection() as conn:
            rows = conn.execute(query, params).fetchall()
            return [self._row_to_field(row) for row in rows]

    def get_aliases_for_field(self, field_id: int) -> list[InstrumentField]:
        """Get all alias fields that point to a specific field."""
        with self._get_connection() as conn:
            rows = conn.execute(
                "SELECT * FROM instrument_fields WHERE alias_field_id = ?",
                (field_id,)
            ).fetchall()
            return [self._row_to_field(row) for row in rows]

    def resolve_alias(self, field_id: int) -> InstrumentField:
        """
        Resolve an alias to its target field.

        If the field is not an alias, returns the field itself.
        Handles chains of aliases by following them to the end.

        Args:
            field_id: ID of the field (possibly an alias)

        Returns:
            The resolved target field

        Raises:
            ValueError: If field not found or circular alias detected
        """
        visited = set()
        current_id = field_id

        while True:
            if current_id in visited:
                raise ValueError(f"Circular alias detected at field ID: {current_id}")

            visited.add(current_id)
            field = self.get_field(current_id)

            if not field:
                raise ValueError(f"Field not found: {current_id}")

            if field.alias_field_id is None:
                return field

            current_id = field.alias_field_id

    def update_field(
        self,
        field_id: int,
        **kwargs
    ) -> Optional[InstrumentField]:
        """
        Update a field's attributes.

        Args:
            field_id: ID of the field to update
            **kwargs: Attributes to update

        Returns:
            Updated field or None if not found
        """
        allowed_fields = {'field_name', 'frequency', 'description', 'unit',
                          'alias_instrument_id', 'alias_field_id', 'metadata'}

        updates = {k: v for k, v in kwargs.items() if k in allowed_fields}
        if not updates:
            return self.get_field(field_id)

        # Handle special conversions
        if 'frequency' in updates and isinstance(updates['frequency'], Frequency):
            updates['frequency'] = updates['frequency'].value
        if 'metadata' in updates and isinstance(updates['metadata'], dict):
            updates['metadata'] = json.dumps(updates['metadata'])

        set_clause = ", ".join(f"{k} = ?" for k in updates.keys())
        values = list(updates.values()) + [field_id]

        with self._get_connection() as conn:
            conn.execute(
                f"UPDATE instrument_fields SET {set_clause} WHERE id = ?",
                values
            )
            conn.commit()

        logger.info(f"Updated field ID: {field_id}")
        return self.get_field(field_id)

    def delete_field(
        self,
        field_id: int,
        dry_run: bool = False
    ) -> DeletionImpact:
        """
        Delete a field and all related data.

        This will:
        - Delete all provider configs for this field
        - Delete all time series data for this field
        - Warn about aliases pointing to this field

        Args:
            field_id: ID of the field to delete
            dry_run: If True, only simulate and return impact

        Returns:
            DeletionImpact describing what was/would be deleted
        """
        impact = self._calculate_field_deletion_impact(field_id)

        if dry_run:
            logger.info(f"DRY RUN - Would delete field ID: {field_id}")
            return impact

        for warning in impact.warnings:
            logger.warning(warning)

        with self._get_connection() as conn:
            # Handle aliases pointing to this field
            aliases = conn.execute(
                "SELECT id FROM instrument_fields WHERE alias_field_id = ?",
                (field_id,)
            ).fetchall()

            for alias in aliases:
                conn.execute(
                    "DELETE FROM instrument_fields WHERE id = ?",
                    (alias['id'],)
                )

            # Delete the field (CASCADE handles provider_configs and time_series_data)
            conn.execute("DELETE FROM instrument_fields WHERE id = ?", (field_id,))
            conn.commit()

        logger.info(f"Deleted field ID: {field_id}")
        return impact

    def _calculate_field_deletion_impact(self, field_id: int) -> DeletionImpact:
        """Calculate the impact of deleting a field."""
        field = self.get_field(field_id)
        if not field:
            return DeletionImpact(
                target_type="field",
                target_id=field_id,
                target_name="NOT FOUND",
                warnings=["Field not found"]
            )

        instrument = self.get_instrument(field.instrument_id)

        impact = DeletionImpact(
            target_type="field",
            target_id=field_id,
            target_name=f"{instrument.ticker}.{field.field_name} ({field.frequency.value})"
        )

        with self._get_connection() as conn:
            # Provider configs
            configs = conn.execute(
                "SELECT id, provider FROM provider_configs WHERE field_id = ?",
                (field_id,)
            ).fetchall()

            impact.provider_configs_to_delete = [
                {"id": c['id'], "provider": c['provider']}
                for c in configs
            ]

            # Time series count
            count = conn.execute(
                "SELECT COUNT(*) as cnt FROM time_series_data WHERE field_id = ?",
                (field_id,)
            ).fetchone()

            impact.time_series_points_to_delete = count['cnt']

            # Aliases
            aliases = conn.execute(
                """
                SELECT if2.id, if2.field_name, if2.frequency, i.ticker
                FROM instrument_fields if2
                JOIN instruments i ON if2.instrument_id = i.id
                WHERE if2.alias_field_id = ?
                """,
                (field_id,)
            ).fetchall()

            impact.aliases_to_delete = [
                {"id": a['id'], "ticker": a['ticker'],
                 "field": a['field_name'], "frequency": a['frequency']}
                for a in aliases
            ]

            for alias in impact.aliases_to_delete:
                impact.warnings.append(
                    f"WARNING: Alias {alias['ticker']}.{alias['field']} ({alias['frequency']}) "
                    f"points to this field and will be deleted"
                )

        return impact

    # =========================================================================
    # Provider Config Operations
    # =========================================================================

    def add_provider_config(
        self,
        field_id: int,
        provider: DataProvider,
        config: dict,
        is_active: bool = True,
        priority: int = 0
    ) -> ProviderConfig:
        """
        Add a data provider configuration for a field.

        Args:
            field_id: ID of the field
            provider: Data provider type
            config: Provider-specific configuration (e.g., Bloomberg settings)
            is_active: Whether this config is active
            priority: Priority for fetching (lower = higher priority)

        Returns:
            The created ProviderConfig

        Example config for Bloomberg:
            {
                "ticker": "AAPL US Equity",
                "field": "PX_LAST",
                "override": {"BEST_FPERIOD_OVERRIDE": "1BF"}
            }
        """
        with self._get_connection() as conn:
            cursor = conn.execute(
                """
                INSERT INTO provider_configs (field_id, provider, config, is_active, priority)
                VALUES (?, ?, ?, ?, ?)
                """,
                (field_id, provider.value, json.dumps(config), int(is_active), priority)
            )
            conn.commit()

            provider_config = self.get_provider_config(cursor.lastrowid)
            logger.info(
                f"Added provider config: {provider.value} for field ID: {field_id}"
            )
            return provider_config

    def get_provider_config(self, config_id: int) -> Optional[ProviderConfig]:
        """Get a provider config by ID."""
        with self._get_connection() as conn:
            row = conn.execute(
                "SELECT * FROM provider_configs WHERE id = ?",
                (config_id,)
            ).fetchone()

            if row:
                return self._row_to_provider_config(row)
            return None

    def get_provider_configs_for_field(
        self,
        field_id: int,
        active_only: bool = True
    ) -> list[ProviderConfig]:
        """Get all provider configs for a field, ordered by priority."""
        query = "SELECT * FROM provider_configs WHERE field_id = ?"
        params = [field_id]

        if active_only:
            query += " AND is_active = 1"

        query += " ORDER BY priority"

        with self._get_connection() as conn:
            rows = conn.execute(query, params).fetchall()
            return [self._row_to_provider_config(row) for row in rows]

    def update_provider_config(
        self,
        config_id: int,
        **kwargs
    ) -> Optional[ProviderConfig]:
        """Update a provider config."""
        allowed_fields = {'provider', 'config', 'is_active', 'priority'}

        updates = {k: v for k, v in kwargs.items() if k in allowed_fields}
        if not updates:
            return self.get_provider_config(config_id)

        if 'provider' in updates and isinstance(updates['provider'], DataProvider):
            updates['provider'] = updates['provider'].value
        if 'config' in updates and isinstance(updates['config'], dict):
            updates['config'] = json.dumps(updates['config'])
        if 'is_active' in updates:
            updates['is_active'] = int(updates['is_active'])

        set_clause = ", ".join(f"{k} = ?" for k in updates.keys())
        values = list(updates.values()) + [config_id]

        with self._get_connection() as conn:
            conn.execute(
                f"UPDATE provider_configs SET {set_clause} WHERE id = ?",
                values
            )
            conn.commit()

        logger.info(f"Updated provider config ID: {config_id}")
        return self.get_provider_config(config_id)

    def delete_provider_config(self, config_id: int) -> bool:
        """Delete a provider config."""
        with self._get_connection() as conn:
            cursor = conn.execute(
                "DELETE FROM provider_configs WHERE id = ?",
                (config_id,)
            )
            conn.commit()

            if cursor.rowcount > 0:
                logger.info(f"Deleted provider config ID: {config_id}")
                return True
            return False

    # =========================================================================
    # Time Series Data Operations
    # =========================================================================

    def add_time_series_point(
        self,
        field_id: int,
        timestamp: datetime | date,
        value: float,
        metadata: Optional[dict] = None
    ) -> TimeSeriesPoint:
        """
        Add a single time series data point.

        Args:
            field_id: ID of the field
            timestamp: Timestamp of the data point
            value: The value
            metadata: Optional additional data (e.g., volume, open, high, low)

        Returns:
            The created TimeSeriesPoint
        """
        if isinstance(timestamp, date) and not isinstance(timestamp, datetime):
            timestamp = datetime.combine(timestamp, datetime.min.time())

        metadata = metadata or {}

        with self._get_connection() as conn:
            cursor = conn.execute(
                """
                INSERT OR REPLACE INTO time_series_data (field_id, timestamp, value, metadata)
                VALUES (?, ?, ?, ?)
                """,
                (field_id, timestamp, value, json.dumps(metadata))
            )
            conn.commit()

            return TimeSeriesPoint(
                id=cursor.lastrowid,
                field_id=field_id,
                timestamp=timestamp,
                value=value,
                metadata=metadata
            )

    def add_time_series_bulk(
        self,
        field_id: int,
        data: list[tuple[datetime | date, float, Optional[dict]]]
    ) -> int:
        """
        Add multiple time series data points efficiently.

        Args:
            field_id: ID of the field
            data: List of (timestamp, value, metadata) tuples

        Returns:
            Number of points inserted
        """
        processed_data = []
        for item in data:
            ts = item[0]
            if isinstance(ts, date) and not isinstance(ts, datetime):
                ts = datetime.combine(ts, datetime.min.time())

            metadata = item[2] if len(item) > 2 and item[2] else {}
            processed_data.append((field_id, ts, item[1], json.dumps(metadata)))

        with self._get_connection() as conn:
            conn.executemany(
                """
                INSERT OR REPLACE INTO time_series_data (field_id, timestamp, value, metadata)
                VALUES (?, ?, ?, ?)
                """,
                processed_data
            )
            conn.commit()

        logger.info(f"Bulk inserted {len(processed_data)} points for field ID: {field_id}")
        return len(processed_data)

    def get_time_series(
        self,
        field_id: int,
        start_date: Optional[datetime | date] = None,
        end_date: Optional[datetime | date] = None,
        resolve_alias: bool = True
    ) -> list[TimeSeriesPoint]:
        """
        Get time series data for a field.

        Args:
            field_id: ID of the field
            start_date: Start of date range (inclusive)
            end_date: End of date range (inclusive)
            resolve_alias: If True and field is an alias, get data from target field

        Returns:
            List of TimeSeriesPoint objects
        """
        # Resolve alias if needed
        actual_field_id = field_id
        if resolve_alias:
            resolved_field = self.resolve_alias(field_id)
            actual_field_id = resolved_field.id

        query = "SELECT * FROM time_series_data WHERE field_id = ?"
        params: list[Any] = [actual_field_id]

        if start_date:
            if isinstance(start_date, date) and not isinstance(start_date, datetime):
                start_date = datetime.combine(start_date, datetime.min.time())
            query += " AND timestamp >= ?"
            params.append(start_date)

        if end_date:
            if isinstance(end_date, date) and not isinstance(end_date, datetime):
                end_date = datetime.combine(end_date, datetime.max.time())
            query += " AND timestamp <= ?"
            params.append(end_date)

        query += " ORDER BY timestamp"

        with self._get_connection() as conn:
            rows = conn.execute(query, params).fetchall()
            return [self._row_to_time_series_point(row) for row in rows]

    def get_latest_value(
        self,
        field_id: int,
        resolve_alias: bool = True
    ) -> Optional[TimeSeriesPoint]:
        """Get the most recent time series value for a field."""
        actual_field_id = field_id
        if resolve_alias:
            resolved_field = self.resolve_alias(field_id)
            actual_field_id = resolved_field.id

        with self._get_connection() as conn:
            row = conn.execute(
                """
                SELECT * FROM time_series_data
                WHERE field_id = ?
                ORDER BY timestamp DESC
                LIMIT 1
                """,
                (actual_field_id,)
            ).fetchone()

            if row:
                return self._row_to_time_series_point(row)
            return None

    def delete_time_series(
        self,
        field_id: int,
        start_date: Optional[datetime | date] = None,
        end_date: Optional[datetime | date] = None
    ) -> int:
        """
        Delete time series data for a field within a date range.

        Args:
            field_id: ID of the field
            start_date: Start of date range (if None, no lower bound)
            end_date: End of date range (if None, no upper bound)

        Returns:
            Number of points deleted
        """
        query = "DELETE FROM time_series_data WHERE field_id = ?"
        params: list[Any] = [field_id]

        if start_date:
            if isinstance(start_date, date) and not isinstance(start_date, datetime):
                start_date = datetime.combine(start_date, datetime.min.time())
            query += " AND timestamp >= ?"
            params.append(start_date)

        if end_date:
            if isinstance(end_date, date) and not isinstance(end_date, datetime):
                end_date = datetime.combine(end_date, datetime.max.time())
            query += " AND timestamp <= ?"
            params.append(end_date)

        with self._get_connection() as conn:
            cursor = conn.execute(query, params)
            conn.commit()

            logger.info(f"Deleted {cursor.rowcount} time series points for field ID: {field_id}")
            return cursor.rowcount

    # =========================================================================
    # Utility Methods
    # =========================================================================

    def preview_deletion(
        self,
        target_type: str,
        target_id: int
    ) -> DeletionImpact:
        """
        Preview what would be deleted without actually deleting.

        Args:
            target_type: "instrument" or "field"
            target_id: ID of the target

        Returns:
            DeletionImpact with full details
        """
        if target_type == "instrument":
            return self._calculate_instrument_deletion_impact(target_id)
        elif target_type == "field":
            return self._calculate_field_deletion_impact(target_id)
        else:
            raise ValueError(f"Unknown target type: {target_type}")

    def get_full_field_info(self, field_id: int) -> Optional[dict]:
        """
        Get comprehensive information about a field including its instrument,
        provider configs, and time series summary.
        """
        field = self.get_field(field_id)
        if not field:
            return None

        instrument = self.get_instrument(field.instrument_id)
        provider_configs = self.get_provider_configs_for_field(field_id, active_only=False)

        with self._get_connection() as conn:
            stats = conn.execute(
                """
                SELECT
                    COUNT(*) as count,
                    MIN(timestamp) as first_date,
                    MAX(timestamp) as last_date,
                    MIN(value) as min_value,
                    MAX(value) as max_value,
                    AVG(value) as avg_value
                FROM time_series_data
                WHERE field_id = ?
                """,
                (field_id,)
            ).fetchone()

        # Get alias info if applicable
        alias_info = None
        if field.alias_field_id:
            target_field = self.get_field(field.alias_field_id)
            if target_field:
                target_instrument = self.get_instrument(target_field.instrument_id)
                alias_info = {
                    "target_instrument": target_instrument.ticker if target_instrument else None,
                    "target_field": target_field.field_name,
                    "target_frequency": target_field.frequency.value
                }

        return {
            "field": asdict(field),
            "instrument": asdict(instrument) if instrument else None,
            "provider_configs": [asdict(pc) for pc in provider_configs],
            "alias_info": alias_info,
            "time_series_stats": {
                "count": stats['count'],
                "first_date": stats['first_date'],
                "last_date": stats['last_date'],
                "min_value": stats['min_value'],
                "max_value": stats['max_value'],
                "avg_value": stats['avg_value']
            }
        }

    def export_to_dict(self) -> dict:
        """Export the entire database to a dictionary (useful for backup/migration)."""
        with self._get_connection() as conn:
            instruments = conn.execute("SELECT * FROM instruments").fetchall()
            fields = conn.execute("SELECT * FROM instrument_fields").fetchall()
            configs = conn.execute("SELECT * FROM provider_configs").fetchall()

            # Don't export time series data by default (can be large)
            ts_count = conn.execute(
                "SELECT COUNT(*) as cnt FROM time_series_data"
            ).fetchone()['cnt']

        return {
            "instruments": [dict(row) for row in instruments],
            "fields": [dict(row) for row in fields],
            "provider_configs": [dict(row) for row in configs],
            "time_series_point_count": ts_count
        }

    def vacuum(self) -> None:
        """Optimize the database file size."""
        with self._get_connection() as conn:
            conn.execute("VACUUM")
        logger.info("Database vacuumed")

    def close(self) -> None:
        """Close the database connection (if persistent)."""
        if self._persistent_conn:
            self._persistent_conn.close()
            self._persistent_conn = None
            logger.info("Database connection closed")

    # =========================================================================
    # Private Helper Methods
    # =========================================================================

    def _row_to_instrument(self, row: sqlite3.Row) -> Instrument:
        """Convert a database row to an Instrument object."""
        return Instrument(
            id=row['id'],
            ticker=row['ticker'],
            name=row['name'],
            instrument_type=InstrumentType(row['instrument_type']),
            description=row['description'],
            currency=row['currency'],
            exchange=row['exchange'],
            metadata=json.loads(row['metadata']) if row['metadata'] else {},
            created_at=row['created_at'],
            updated_at=row['updated_at']
        )

    def _row_to_field(self, row: sqlite3.Row) -> InstrumentField:
        """Convert a database row to an InstrumentField object."""
        return InstrumentField(
            id=row['id'],
            instrument_id=row['instrument_id'],
            field_name=row['field_name'],
            frequency=Frequency(row['frequency']),
            description=row['description'],
            unit=row['unit'],
            alias_instrument_id=row['alias_instrument_id'],
            alias_field_id=row['alias_field_id'],
            metadata=json.loads(row['metadata']) if row['metadata'] else {},
            created_at=row['created_at'],
            updated_at=row['updated_at']
        )

    def _row_to_provider_config(self, row: sqlite3.Row) -> ProviderConfig:
        """Convert a database row to a ProviderConfig object."""
        return ProviderConfig(
            id=row['id'],
            field_id=row['field_id'],
            provider=DataProvider(row['provider']),
            config=json.loads(row['config']) if row['config'] else {},
            is_active=bool(row['is_active']),
            priority=row['priority'],
            created_at=row['created_at'],
            updated_at=row['updated_at']
        )

    def _row_to_time_series_point(self, row: sqlite3.Row) -> TimeSeriesPoint:
        """Convert a database row to a TimeSeriesPoint object."""
        return TimeSeriesPoint(
            id=row['id'],
            field_id=row['field_id'],
            timestamp=row['timestamp'],
            value=row['value'],
            metadata=json.loads(row['metadata']) if row['metadata'] else {}
        )


# =============================================================================
# Convenience Functions
# =============================================================================

def create_database(db_path: str | Path = ":memory:") -> FinancialTimeSeriesDB:
    """
    Create a new financial time series database.

    Args:
        db_path: Path to the database file or ":memory:" for in-memory database

    Returns:
        FinancialTimeSeriesDB instance
    """
    return FinancialTimeSeriesDB(db_path)


def print_deletion_impact(impact: DeletionImpact) -> None:
    """Pretty print a deletion impact report."""
    print("\n" + "=" * 60)
    print(f"DELETION IMPACT REPORT: {impact.target_type.upper()}")
    print("=" * 60)
    print(f"Target: {impact.target_name} (ID: {impact.target_id})")
    print("-" * 60)

    if impact.fields_to_delete:
        print(f"\nFields to be deleted ({len(impact.fields_to_delete)}):")
        for f in impact.fields_to_delete:
            print(f"  - {f['name']} ({f['frequency']})")

    if impact.aliases_to_delete:
        print(f"\nAlias fields to be deleted ({len(impact.aliases_to_delete)}):")
        for a in impact.aliases_to_delete:
            print(f"  - {a['ticker']}.{a['field']} ({a['frequency']})")

    if impact.provider_configs_to_delete:
        print(f"\nProvider configs to be deleted ({len(impact.provider_configs_to_delete)}):")
        for c in impact.provider_configs_to_delete:
            print(f"  - {c['provider']} (ID: {c['id']})")

    if impact.time_series_points_to_delete > 0:
        print(f"\nTime series points to be deleted: {impact.time_series_points_to_delete:,}")

    if impact.warnings:
        print("\n⚠️  WARNINGS:")
        for warning in impact.warnings:
            print(f"  {warning}")

    print("=" * 60 + "\n")


if __name__ == "__main__":
    # Run example usage
    from example_usage import main
    main()
