"""
Financial Time Series Database Interface

A SQLite-based database for storing and managing financial time series data,
including instruments, fields, aliases, and data provider configurations.

All public methods use string identifiers (ticker, field_name, frequency)
instead of numeric IDs for a cleaner API.

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
from typing import Optional, Any, Union
from contextlib import contextmanager

import pandas as pd

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
# Storable Fields Registry
# =============================================================================

@dataclass
class StorableFieldDef:
    """Definition of a storable field with its metadata."""
    name: str
    description: str = ""
    metadata: dict = field(default_factory=dict)


# Default storable fields that are allowed in the database.
# Users can add or remove fields via the database instance methods.
DEFAULT_STORABLE_FIELDS: dict[str, StorableFieldDef] = {
    "price": StorableFieldDef(
        name="price",
        description="Last traded price",
        metadata={"unit": "currency"}
    ),
    "pct total return": StorableFieldDef(
        name="pct total return",
        description="Percentage total return including dividends",
        metadata={"unit": "percent"}
    ),
}


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
    extra_data: dict = field(default_factory=dict)
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
    # These are string identifiers for the public API
    alias_ticker: Optional[str] = None
    alias_field_name: Optional[str] = None
    alias_frequency: Optional[str] = None
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
    extra_data TEXT DEFAULT '{}',
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

-- Storable fields table (for field name registry with descriptions and metadata)
CREATE TABLE IF NOT EXISTS storable_fields (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL UNIQUE,
    description TEXT DEFAULT '',
    metadata TEXT DEFAULT '{}',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
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
CREATE INDEX IF NOT EXISTS idx_storable_fields_name ON storable_fields(name);

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

CREATE TRIGGER IF NOT EXISTS update_storable_field_timestamp
AFTER UPDATE ON storable_fields
BEGIN
    UPDATE storable_fields SET updated_at = CURRENT_TIMESTAMP WHERE id = NEW.id;
END;
"""


# =============================================================================
# Helper function for frequency conversion
# =============================================================================

def _to_frequency(frequency: Union[str, Frequency]) -> Frequency:
    """Convert string or Frequency enum to Frequency enum."""
    if isinstance(frequency, Frequency):
        return frequency
    try:
        return Frequency(frequency.lower())
    except ValueError:
        raise ValueError(
            f"Invalid frequency '{frequency}'. Valid options: "
            f"{', '.join(f.value for f in Frequency)}"
        )


# =============================================================================
# Main Database Class
# =============================================================================

class FinancialTimeSeriesDB:
    """
    Interface for the Financial Time Series SQLite Database.

    All public methods use string identifiers (ticker, field_name, frequency)
    instead of numeric IDs for a cleaner, more intuitive API.

    This class provides methods to:
    - Manage instruments (add, update, delete, query)
    - Manage fields per instrument at different frequencies
    - Handle field aliases (pointing to other instruments' fields)
    - Configure data provider settings
    - Store and retrieve time series data
    - Preview deletion impacts before executing
    - Manage storable field definitions (persisted to database)

    Example:
        db = FinancialTimeSeriesDB("my_database.db")

        # Add a storable field definition (persisted to database)
        db.add_storable_field("price", "Last traded price", {"unit": "currency"})

        # Add an instrument
        apple = db.add_instrument(
            ticker="AAPL",
            name="Apple Inc.",
            instrument_type=InstrumentType.STOCK
        )

        # Add a field using ticker
        price_field = db.add_field(
            ticker="AAPL",
            field_name="price",
            frequency="daily"
        )

        # Configure provider using string identifiers
        db.add_provider_config(
            ticker="AAPL",
            field_name="price",
            frequency="daily",
            provider=DataProvider.BLOOMBERG,
            config={"ticker": "AAPL US Equity", "field": "PX_LAST"}
        )

        # Add time series data
        db.add_time_series_point(
            ticker="AAPL",
            field_name="price",
            frequency="daily",
            timestamp=datetime.now(),
            value=150.0
        )
    """

    def __init__(self, db_path: str | Path = ":memory:"):
        """
        Initialize the database connection.

        Args:
            db_path: Path to the SQLite database file. Use ":memory:" for in-memory database.

        Storable fields are managed via the public API (add_storable_field, remove_storable_field)
        and persisted in the database. On first initialization, default storable fields are added.
        """
        self.db_path = Path(db_path) if db_path != ":memory:" else db_path
        self._is_memory = db_path == ":memory:"
        self._persistent_conn: Optional[sqlite3.Connection] = None

        # For in-memory databases, we need a persistent connection
        if self._is_memory:
            self._persistent_conn = self._create_connection()

        # Initialize schema first
        self._initialize_db()

        # Load storable fields from database, or initialize if empty
        self._storable_fields: dict[str, StorableFieldDef] = {}
        self._load_storable_fields_from_db()

        # If database has no storable fields, populate with defaults
        if not self._storable_fields:
            for name, field_def in DEFAULT_STORABLE_FIELDS.items():
                self._persist_storable_field(field_def)

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

    def _load_storable_fields_from_db(self) -> None:
        """Load storable fields from the database into memory."""
        with self._get_connection() as conn:
            rows = conn.execute("SELECT name, description, metadata FROM storable_fields").fetchall()
            for row in rows:
                name = row['name'].lower()
                metadata = json.loads(row['metadata']) if row['metadata'] else {}
                self._storable_fields[name] = StorableFieldDef(
                    name=name,
                    description=row['description'] or "",
                    metadata=metadata
                )

    def _persist_storable_field(self, field_def: StorableFieldDef) -> None:
        """Persist a storable field to the database."""
        normalized_name = field_def.name.lower()
        with self._get_connection() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO storable_fields (name, description, metadata)
                VALUES (?, ?, ?)
                """,
                (normalized_name, field_def.description, json.dumps(field_def.metadata))
            )
            conn.commit()
        # Also update in-memory cache
        self._storable_fields[normalized_name] = StorableFieldDef(
            name=normalized_name,
            description=field_def.description,
            metadata=field_def.metadata
        )

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
    # Internal ID lookup helpers
    # =========================================================================

    def _get_instrument_id(self, ticker: str) -> int:
        """Get instrument ID by ticker, raising ValueError if not found."""
        with self._get_connection() as conn:
            row = conn.execute(
                "SELECT id FROM instruments WHERE ticker = ?",
                (ticker,)
            ).fetchone()
            if not row:
                raise ValueError(f"Instrument not found with ticker: {ticker}")
            return row['id']

    def _get_field_id(
        self,
        ticker: str,
        field_name: str,
        frequency: Union[str, Frequency]
    ) -> int:
        """Get field ID by ticker, field_name, frequency, raising ValueError if not found."""
        freq = _to_frequency(frequency)
        instrument_id = self._get_instrument_id(ticker)

        with self._get_connection() as conn:
            row = conn.execute(
                """
                SELECT id FROM instrument_fields
                WHERE instrument_id = ? AND field_name = ? AND frequency = ?
                """,
                (instrument_id, field_name, freq.value)
            ).fetchone()
            if not row:
                raise ValueError(
                    f"Field not found: ticker={ticker}, field_name={field_name}, frequency={freq.value}"
                )
            return row['id']

    def _get_field_by_id(self, field_id: int) -> Optional[InstrumentField]:
        """Get a field by its internal ID (for internal use)."""
        with self._get_connection() as conn:
            row = conn.execute(
                "SELECT * FROM instrument_fields WHERE id = ?",
                (field_id,)
            ).fetchone()
            if row:
                return self._row_to_field(row, conn)
            return None

    def _get_instrument_by_id(self, instrument_id: int) -> Optional[Instrument]:
        """Get an instrument by its internal ID (for internal use)."""
        with self._get_connection() as conn:
            row = conn.execute(
                "SELECT * FROM instruments WHERE id = ?",
                (instrument_id,)
            ).fetchone()
            if row:
                return self._row_to_instrument(row)
            return None

    # =========================================================================
    # Storable Fields Management
    # =========================================================================

    def get_storable_fields(self) -> dict[str, StorableFieldDef]:
        """
        Get the current dict of allowed storable field definitions.

        Returns:
            Dict mapping field names (lowercase) to StorableFieldDef objects
        """
        return self._storable_fields.copy()

    def get_storable_field(self, field_name: str) -> Optional[StorableFieldDef]:
        """
        Get a storable field definition by name.

        Args:
            field_name: The field name to look up (case-insensitive)

        Returns:
            StorableFieldDef if found, None otherwise
        """
        return self._storable_fields.get(field_name.lower())

    def add_storable_field(
        self,
        field_name: str,
        description: str = "",
        metadata: Optional[dict] = None,
        overwrite: bool = False
    ) -> None:
        """
        Add a field definition to the list of allowed storable fields.

        The field definition is persisted in the database.

        Args:
            field_name: The field name to allow (case-insensitive)
            description: Description of the field
            metadata: Additional metadata for the field (e.g., {"unit": "USD"})
            overwrite: If True, overwrites existing field. If False, raises error if exists.

        Raises:
            ValueError: If the field already exists and overwrite=False.

        Example:
            db.add_storable_field("dividend yield", "Annual dividend yield", {"unit": "percent"})
            db.add_storable_field("eps", "Earnings per share", {"unit": "currency"})

            # Update existing field
            db.add_storable_field("price", "Updated description", overwrite=True)
        """
        normalized = field_name.lower()
        exists = normalized in self._storable_fields

        if exists and not overwrite:
            raise ValueError(
                f"Storable field '{normalized}' already exists. "
                f"Use overwrite=True to update it."
            )

        field_def = StorableFieldDef(
            name=normalized,
            description=description,
            metadata=metadata or {}
        )
        self._persist_storable_field(field_def)

        if exists:
            logger.info(f"Updated storable field: {normalized}")
        else:
            logger.info(f"Added storable field: {normalized}")

    def remove_storable_field(self, field_name: str) -> bool:
        """
        Remove a field name from the list of allowed storable fields.

        Note: This does not delete existing fields with this name from the database.
        It only prevents new fields with this name from being added.

        Args:
            field_name: The field name to remove (case-insensitive)

        Returns:
            True if the field was removed, False if it wasn't in the list
        """
        normalized = field_name.lower()
        if normalized in self._storable_fields:
            # Remove from database
            with self._get_connection() as conn:
                conn.execute(
                    "DELETE FROM storable_fields WHERE name = ?",
                    (normalized,)
                )
                conn.commit()
            # Remove from in-memory cache
            del self._storable_fields[normalized]
            logger.info(f"Removed storable field: {normalized}")
            return True
        return False

    def is_storable_field(self, field_name: str) -> bool:
        """
        Check if a field name is in the allowed storable fields list.

        Args:
            field_name: The field name to check (case-insensitive)

        Returns:
            True if the field name is allowed, False otherwise
        """
        return field_name.lower() in self._storable_fields

    def set_storable_fields(self, field_defs: dict[str, StorableFieldDef]) -> None:
        """
        Replace the entire dict of allowed storable fields.

        This clears all existing storable fields from the database and replaces them.

        Args:
            field_defs: New dict of field definitions

        Example:
            db.set_storable_fields({
                "price": StorableFieldDef("price", "Last price", {"unit": "currency"}),
                "volume": StorableFieldDef("volume", "Trading volume", {"unit": "shares"}),
            })
        """
        # Clear existing storable fields from database and memory
        with self._get_connection() as conn:
            conn.execute("DELETE FROM storable_fields")
            conn.commit()
        self._storable_fields = {}

        # Add new storable fields
        for name, field_def in field_defs.items():
            normalized = name.lower()
            normalized_def = StorableFieldDef(
                name=normalized,
                description=field_def.description,
                metadata=field_def.metadata
            )
            self._persist_storable_field(normalized_def)

        logger.info(f"Set storable fields to: {list(self._storable_fields.keys())}")

    def validate_field_name(self, field_name: str) -> None:
        """
        Validate that a field name is in the allowed storable fields list.

        Args:
            field_name: The field name to validate

        Raises:
            ValueError: If the field name is not in the allowed list
        """
        if not self._storable_fields:
            # If storable fields is empty, validation is disabled
            return

        if not self.is_storable_field(field_name):
            allowed = ", ".join(sorted(self._storable_fields.keys()))
            raise ValueError(
                f"Field name '{field_name}' is not in the allowed storable fields list. "
                f"Allowed fields: {allowed}. "
                f"Use db.add_storable_field('{field_name.lower()}', description, metadata) to allow this field."
            )

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
        metadata: Optional[dict] = None,
        extra_data: Optional[dict] = None
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
            extra_data: Additional JSON data for flexible storage

        Returns:
            The created Instrument object with its ID

        Raises:
            sqlite3.IntegrityError: If ticker already exists
        """
        metadata = metadata or {}
        extra_data = extra_data or {}

        with self._get_connection() as conn:
            cursor = conn.execute(
                """
                INSERT INTO instruments (ticker, name, instrument_type, description,
                                         currency, exchange, metadata, extra_data)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (ticker, name, instrument_type.value, description,
                 currency, exchange, json.dumps(metadata), json.dumps(extra_data))
            )
            conn.commit()

            instrument = self.get_instrument(ticker)
            logger.info(f"Added instrument: {ticker}")
            return instrument

    def get_instrument(self, ticker: str) -> Optional[Instrument]:
        """
        Get an instrument by ticker symbol.

        Args:
            ticker: The ticker symbol

        Returns:
            Instrument if found, None otherwise
        """
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
        ticker: str,
        **kwargs
    ) -> Optional[Instrument]:
        """
        Update an instrument's attributes.

        Args:
            ticker: Ticker of the instrument to update
            **kwargs: Attributes to update (name, instrument_type,
                      description, currency, exchange, metadata, extra_data)

        Returns:
            Updated Instrument or None if not found
        """
        instrument = self.get_instrument(ticker)
        if not instrument:
            return None

        allowed_fields = {'name', 'instrument_type', 'description',
                          'currency', 'exchange', 'metadata', 'extra_data'}

        updates = {k: v for k, v in kwargs.items() if k in allowed_fields}
        if not updates:
            return instrument

        # Handle special conversions
        if 'instrument_type' in updates and isinstance(updates['instrument_type'], InstrumentType):
            updates['instrument_type'] = updates['instrument_type'].value
        if 'metadata' in updates and isinstance(updates['metadata'], dict):
            updates['metadata'] = json.dumps(updates['metadata'])
        if 'extra_data' in updates and isinstance(updates['extra_data'], dict):
            updates['extra_data'] = json.dumps(updates['extra_data'])

        set_clause = ", ".join(f"{k} = ?" for k in updates.keys())
        values = list(updates.values()) + [instrument.id]

        with self._get_connection() as conn:
            conn.execute(
                f"UPDATE instruments SET {set_clause} WHERE id = ?",
                values
            )
            conn.commit()

        logger.info(f"Updated instrument: {ticker}")
        return self.get_instrument(ticker)

    def update_instrument_extra_data(
        self,
        ticker: str,
        data: dict,
        merge: bool = True
    ) -> Optional[Instrument]:
        """
        Update an instrument's extra_data JSON field.

        Args:
            ticker: Ticker of the instrument to update
            data: Dictionary of data to add/update
            merge: If True, merge with existing data (default). If False, replace entirely.

        Returns:
            Updated Instrument or None if not found

        Example:
            db.update_instrument_extra_data("AAPL", {"sector": "Technology", "beta": 1.2})
        """
        instrument = self.get_instrument(ticker)
        if not instrument:
            return None

        if merge:
            existing = instrument.extra_data or {}
            new_data = {**existing, **data}
        else:
            new_data = data

        return self.update_instrument(ticker, extra_data=new_data)

    def get_instrument_extra_data(
        self,
        ticker: str,
        key: Optional[str] = None
    ) -> Optional[Any]:
        """
        Get an instrument's extra_data or a specific key from it.

        Args:
            ticker: Ticker of the instrument
            key: Optional specific key to retrieve. If None, returns all extra_data.

        Returns:
            The extra_data dict, a specific value, or None if instrument not found
        """
        instrument = self.get_instrument(ticker)
        if not instrument:
            return None

        if key is None:
            return instrument.extra_data

        return instrument.extra_data.get(key)

    def delete_instrument(
        self,
        ticker: str,
        dry_run: bool = False,
        print_output: bool = True
    ) -> DeletionImpact:
        """
        Delete an instrument and all related data.

        Args:
            ticker: Ticker of the instrument to delete
            dry_run: If True, only simulate and return impact without deleting
            print_output: If True and dry_run is True, print impact report to stdout

        Returns:
            DeletionImpact object describing what was/would be deleted
        """
        instrument = self.get_instrument(ticker)
        if not instrument:
            return DeletionImpact(
                target_type="instrument",
                target_name=ticker,
                warnings=["Instrument not found"]
            )

        impact = self._calculate_instrument_deletion_impact(instrument.id)

        if dry_run:
            logger.info(f"DRY RUN - Would delete instrument: {ticker}")
            if print_output:
                print_deletion_impact(impact)
            return impact

        for warning in impact.warnings:
            logger.warning(warning)

        with self._get_connection() as conn:
            # First, delete alias fields pointing to this instrument
            alias_fields = conn.execute(
                """
                SELECT if2.id, if2.instrument_id, i.ticker, if2.field_name
                FROM instrument_fields if2
                JOIN instruments i ON if2.instrument_id = i.id
                WHERE if2.alias_instrument_id = ?
                """,
                (instrument.id,)
            ).fetchall()

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
            conn.execute("DELETE FROM instruments WHERE id = ?", (instrument.id,))
            conn.commit()

        logger.info(f"Deleted instrument: {ticker}")
        return impact

    def _calculate_instrument_deletion_impact(self, instrument_id: int) -> DeletionImpact:
        """Calculate the impact of deleting an instrument."""
        instrument = self._get_instrument_by_id(instrument_id)
        if not instrument:
            return DeletionImpact(
                target_type="instrument",
                target_name="NOT FOUND",
                warnings=["Instrument not found"]
            )

        impact = DeletionImpact(
            target_type="instrument",
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
                {"name": f['field_name'], "frequency": f['frequency']}
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
                    {"provider": c['provider']}
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
                {"ticker": a['ticker'], "field": a['field_name'], "frequency": a['frequency']}
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
        ticker: str,
        field_name: str,
        frequency: Union[str, Frequency],
        unit: str = "",
        alias_ticker: Optional[str] = None,
        alias_field_name: Optional[str] = None,
        alias_frequency: Optional[Union[str, Frequency]] = None
    ) -> InstrumentField:
        """
        Associate a storable field with an instrument at a specific frequency.

        The field's description and metadata are inherited from the storable field
        registry. Use add_storable_field() to define field properties.

        Args:
            ticker: Ticker symbol of the parent instrument (e.g., "AAPL")
            field_name: Name of the field (must be in storable fields registry)
            frequency: Data frequency (string like "daily" or Frequency enum)
            unit: Unit of measurement (overrides storable field's unit if provided)
            alias_ticker: If this is an alias, the ticker of the target instrument
            alias_field_name: If this is an alias, the target field name
            alias_frequency: If this is an alias, the target frequency (defaults to same)

        Returns:
            The created InstrumentField object

        Raises:
            ValueError: If ticker not found, alias params incomplete, or field_name not allowed
        """
        freq = _to_frequency(frequency)
        instrument_id = self._get_instrument_id(ticker)

        # Validate field name
        self.validate_field_name(field_name)

        # Get description and metadata from storable field definition
        storable_field = self.get_storable_field(field_name)
        description = storable_field.description if storable_field else ""
        field_metadata = storable_field.metadata if storable_field else {}

        # Use unit from storable field metadata if not explicitly provided
        if not unit and storable_field and storable_field.metadata:
            unit = storable_field.metadata.get("unit", "")

        # Handle alias
        alias_instrument_id = None
        alias_field_id = None

        if alias_ticker or alias_field_name:
            if not (alias_ticker and alias_field_name):
                raise ValueError(
                    "Both alias_ticker and alias_field_name must be set together"
                )
            alias_freq = _to_frequency(alias_frequency) if alias_frequency else freq
            alias_instrument_id = self._get_instrument_id(alias_ticker)
            alias_field_id = self._get_field_id(alias_ticker, alias_field_name, alias_freq)

        with self._get_connection() as conn:
            cursor = conn.execute(
                """
                INSERT INTO instrument_fields
                (instrument_id, field_name, frequency, description, unit,
                 alias_instrument_id, alias_field_id, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (instrument_id, field_name, freq.value, description, unit,
                 alias_instrument_id, alias_field_id, json.dumps(field_metadata))
            )
            conn.commit()

            field = self._get_field_by_id(cursor.lastrowid)
            logger.info(f"Added field: {ticker}.{field_name} ({freq.value})")
            return field

    def add_alias_field(
        self,
        ticker: str,
        field_name: str,
        frequency: Union[str, Frequency],
        target_ticker: str,
        target_field_name: str,
        target_frequency: Optional[Union[str, Frequency]] = None
    ) -> InstrumentField:
        """
        Add an alias field that points to another instrument's field.

        Args:
            ticker: Ticker of the instrument to add the alias to
            field_name: Name for the alias field
            frequency: Frequency of the alias
            target_ticker: Ticker of the target instrument
            target_field_name: Name of the target field
            target_frequency: Frequency of target field (defaults to same as alias)

        Returns:
            The created alias field
        """
        freq = _to_frequency(frequency)
        target_freq = _to_frequency(target_frequency) if target_frequency else freq

        return self.add_field(
            ticker=ticker,
            field_name=field_name,
            frequency=freq,
            alias_ticker=target_ticker,
            alias_field_name=target_field_name,
            alias_frequency=target_freq
        )

    def get_field(
        self,
        ticker: str,
        field_name: str,
        frequency: Union[str, Frequency]
    ) -> Optional[InstrumentField]:
        """
        Get a field by ticker, field_name, and frequency.

        Args:
            ticker: Ticker symbol of the instrument
            field_name: Name of the field
            frequency: Data frequency

        Returns:
            InstrumentField if found, None otherwise
        """
        freq = _to_frequency(frequency)
        instrument = self.get_instrument(ticker)
        if not instrument:
            return None

        with self._get_connection() as conn:
            row = conn.execute(
                """
                SELECT * FROM instrument_fields
                WHERE instrument_id = ? AND field_name = ? AND frequency = ?
                """,
                (instrument.id, field_name, freq.value)
            ).fetchone()

            if row:
                return self._row_to_field(row, conn)
            return None

    def list_fields(
        self,
        ticker: Optional[str] = None,
        frequency: Optional[Union[str, Frequency]] = None,
        include_aliases: bool = True
    ) -> list[InstrumentField]:
        """
        List fields with optional filtering.

        Args:
            ticker: Filter by instrument ticker
            frequency: Filter by frequency
            include_aliases: Whether to include alias fields

        Returns:
            List of matching fields
        """
        query = "SELECT * FROM instrument_fields WHERE 1=1"
        params = []

        if ticker:
            instrument = self.get_instrument(ticker)
            if not instrument:
                return []
            query += " AND instrument_id = ?"
            params.append(instrument.id)

        if frequency:
            freq = _to_frequency(frequency)
            query += " AND frequency = ?"
            params.append(freq.value)

        if not include_aliases:
            query += " AND alias_field_id IS NULL"

        query += " ORDER BY instrument_id, field_name, frequency"

        with self._get_connection() as conn:
            rows = conn.execute(query, params).fetchall()
            return [self._row_to_field(row, conn) for row in rows]

    def get_instrument_fields(
        self,
        ticker: str,
        include_aliases: bool = True
    ) -> list[dict]:
        """
        Get a summary of all fields for an instrument with their frequencies.

        Args:
            ticker: Ticker of the instrument
            include_aliases: Whether to include alias fields

        Returns:
            List of dicts with field information:
            [
                {"field_name": "price", "frequency": "daily", "is_alias": False, ...},
                {"field_name": "price", "frequency": "weekly", "is_alias": False, ...},
                ...
            ]

        Example:
            fields = db.get_instrument_fields("AAPL")
            for f in fields:
                print(f"{f['field_name']} ({f['frequency']})")
        """
        fields = self.list_fields(ticker=ticker, include_aliases=include_aliases)

        result = []
        for f in fields:
            result.append({
                "field_name": f.field_name,
                "frequency": f.frequency.value,
                "description": f.description,
                "unit": f.unit,
                "is_alias": f.alias_ticker is not None,
                "alias_target": f"{f.alias_ticker}.{f.alias_field_name}" if f.alias_ticker else None
            })

        return result

    def get_fields_by_metadata(
        self,
        ticker: str,
        metadata_filter: dict,
        frequency: Optional[Union[str, Frequency]] = None,
        include_aliases: bool = True
    ) -> list[dict]:
        """
        Get fields of an instrument that match specific metadata criteria.

        Returns fields where the storable field's metadata contains all the
        specified key-value pairs. The match is exact for each key-value pair.

        Args:
            ticker: Ticker of the instrument
            metadata_filter: Dict of metadata key-value pairs to match.
                             All pairs must match for a field to be included.
                             Metadata is looked up from the storable field registry.
            frequency: Optional frequency filter
            include_aliases: Whether to include alias fields

        Returns:
            List of dicts with field information for matching fields

        Example:
            # First define storable fields with metadata
            db.add_storable_field("vol_25d_call", "25 delta call vol",
                                  {"type": "volatility", "delta": 25, "option_type": "call"})
            db.add_storable_field("vol_50d_atm", "ATM vol",
                                  {"type": "volatility", "delta": 50, "option_type": "atm"})

            # Then add fields to instrument
            db.add_field("AAPL", "vol_25d_call", "daily")
            db.add_field("AAPL", "vol_50d_atm", "daily")

            # Find all volatility fields
            fields = db.get_fields_by_metadata("AAPL", {"type": "volatility"})

            # Find 50 delta fields
            fields = db.get_fields_by_metadata("AAPL", {"delta": 50})
        """
        # Get all fields for the instrument
        fields = self.list_fields(
            ticker=ticker,
            frequency=frequency,
            include_aliases=include_aliases
        )

        result = []
        for f in fields:
            # Look up metadata from storable field registry (source of truth)
            storable_field = self.get_storable_field(f.field_name)
            field_metadata = storable_field.metadata if storable_field else {}

            # Check if all metadata filter criteria match
            matches = True
            for key, value in metadata_filter.items():
                if key not in field_metadata or field_metadata[key] != value:
                    matches = False
                    break

            if matches:
                result.append({
                    "field_name": f.field_name,
                    "frequency": f.frequency.value,
                    "description": f.description,
                    "unit": f.unit,
                    "metadata": field_metadata,
                    "is_alias": f.alias_ticker is not None,
                    "alias_target": f"{f.alias_ticker}.{f.alias_field_name}" if f.alias_ticker else None
                })

        return result

    def get_aliases_for_field(
        self,
        ticker: str,
        field_name: str,
        frequency: Union[str, Frequency]
    ) -> list[InstrumentField]:
        """Get all alias fields that point to a specific field."""
        field_id = self._get_field_id(ticker, field_name, frequency)

        with self._get_connection() as conn:
            rows = conn.execute(
                "SELECT * FROM instrument_fields WHERE alias_field_id = ?",
                (field_id,)
            ).fetchall()
            return [self._row_to_field(row, conn) for row in rows]

    def resolve_alias(
        self,
        ticker: str,
        field_name: str,
        frequency: Union[str, Frequency]
    ) -> InstrumentField:
        """
        Resolve an alias to its target field.

        If the field is not an alias, returns the field itself.
        Handles chains of aliases by following them to the end.

        Args:
            ticker: Ticker of the instrument
            field_name: Name of the field
            frequency: Data frequency

        Returns:
            The resolved target field

        Raises:
            ValueError: If field not found or circular alias detected
        """
        visited = set()
        current_ticker = ticker
        current_field_name = field_name
        current_frequency = _to_frequency(frequency)

        while True:
            field_key = (current_ticker, current_field_name, current_frequency.value)
            if field_key in visited:
                raise ValueError(f"Circular alias detected")

            visited.add(field_key)
            field = self.get_field(current_ticker, current_field_name, current_frequency)

            if not field:
                raise ValueError(f"Field not found during alias resolution")

            if field.alias_ticker is None:
                return field

            # Follow the alias chain
            current_ticker = field.alias_ticker
            current_field_name = field.alias_field_name
            current_frequency = Frequency(field.alias_frequency)

    def update_field(
        self,
        ticker: str,
        field_name: str,
        frequency: Union[str, Frequency],
        **kwargs
    ) -> Optional[InstrumentField]:
        """
        Update a field's attributes.

        Args:
            ticker: Ticker of the instrument
            field_name: Name of the field
            frequency: Data frequency
            **kwargs: Attributes to update (description, unit, metadata)

        Returns:
            Updated field or None if not found
        """
        field_id = self._get_field_id(ticker, field_name, frequency)

        allowed_fields = {'description', 'unit', 'metadata'}
        updates = {k: v for k, v in kwargs.items() if k in allowed_fields}

        if not updates:
            return self._get_field_by_id(field_id)

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

        logger.info(f"Updated field: {ticker}.{field_name}")
        return self._get_field_by_id(field_id)

    def delete_field(
        self,
        ticker: str,
        field_name: str,
        frequency: Union[str, Frequency],
        dry_run: bool = False,
        print_output: bool = True
    ) -> DeletionImpact:
        """
        Delete a field and all related data.

        Args:
            ticker: Ticker of the instrument
            field_name: Name of the field
            frequency: Data frequency
            dry_run: If True, only simulate and return impact
            print_output: If True and dry_run is True, print impact report

        Returns:
            DeletionImpact describing what was/would be deleted
        """
        freq = _to_frequency(frequency)
        field = self.get_field(ticker, field_name, freq)

        if not field:
            return DeletionImpact(
                target_type="field",
                target_name=f"{ticker}.{field_name} ({freq.value})",
                warnings=["Field not found"]
            )

        impact = self._calculate_field_deletion_impact(field.id)

        if dry_run:
            logger.info(f"DRY RUN - Would delete field: {ticker}.{field_name}")
            if print_output:
                print_deletion_impact(impact)
            return impact

        for warning in impact.warnings:
            logger.warning(warning)

        with self._get_connection() as conn:
            # Handle aliases pointing to this field
            aliases = conn.execute(
                "SELECT id FROM instrument_fields WHERE alias_field_id = ?",
                (field.id,)
            ).fetchall()

            for alias in aliases:
                conn.execute(
                    "DELETE FROM instrument_fields WHERE id = ?",
                    (alias['id'],)
                )

            # Delete the field
            conn.execute("DELETE FROM instrument_fields WHERE id = ?", (field.id,))
            conn.commit()

        logger.info(f"Deleted field: {ticker}.{field_name}")
        return impact

    def _calculate_field_deletion_impact(self, field_id: int) -> DeletionImpact:
        """Calculate the impact of deleting a field."""
        field = self._get_field_by_id(field_id)
        if not field:
            return DeletionImpact(
                target_type="field",
                target_name="NOT FOUND",
                warnings=["Field not found"]
            )

        instrument = self._get_instrument_by_id(field.instrument_id)

        impact = DeletionImpact(
            target_type="field",
            target_name=f"{instrument.ticker}.{field.field_name} ({field.frequency.value})"
        )

        with self._get_connection() as conn:
            # Provider configs
            configs = conn.execute(
                "SELECT id, provider FROM provider_configs WHERE field_id = ?",
                (field_id,)
            ).fetchall()

            impact.provider_configs_to_delete = [
                {"provider": c['provider']}
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
                {"ticker": a['ticker'], "field": a['field_name'], "frequency": a['frequency']}
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
        ticker: str,
        field_name: str,
        frequency: Union[str, Frequency],
        provider: DataProvider,
        config: dict,
        is_active: bool = True,
        priority: int = 0,
        pct_change: bool = False
    ) -> ProviderConfig:
        """
        Add a data provider configuration for a field.

        Args:
            ticker: Ticker of the instrument
            field_name: Name of the field
            frequency: Data frequency
            provider: Data provider type
            config: Provider-specific configuration
            is_active: Whether this config is active
            priority: Priority for fetching (lower = higher priority)
            pct_change: If True, apply percent change transformation before storing.
                       Downloaded values will be converted to percentage changes.

        Returns:
            The created ProviderConfig
        """
        field_id = self._get_field_id(ticker, field_name, frequency)

        # Merge pct_change into config
        full_config = {**config, "pct_change": pct_change}

        with self._get_connection() as conn:
            cursor = conn.execute(
                """
                INSERT INTO provider_configs (field_id, provider, config, is_active, priority)
                VALUES (?, ?, ?, ?, ?)
                """,
                (field_id, provider.value, json.dumps(full_config), int(is_active), priority)
            )
            conn.commit()

            provider_config = self._get_provider_config_by_id(cursor.lastrowid)
            logger.info(
                f"Added provider config: {provider.value} for {ticker}.{field_name} "
                f"({frequency if isinstance(frequency, str) else frequency.value}), "
                f"priority={priority}, active={is_active}"
            )
            return provider_config

    def _get_provider_config_by_id(self, config_id: int) -> Optional[ProviderConfig]:
        """Get a provider config by its internal ID."""
        with self._get_connection() as conn:
            row = conn.execute(
                "SELECT * FROM provider_configs WHERE id = ?",
                (config_id,)
            ).fetchone()

            if row:
                return self._row_to_provider_config(row)
            return None

    def get_provider_configs(
        self,
        ticker: str,
        field_name: str,
        frequency: Union[str, Frequency],
        active_only: bool = True
    ) -> list[ProviderConfig]:
        """Get all provider configs for a field, ordered by priority."""
        field_id = self._get_field_id(ticker, field_name, frequency)

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
        ticker: str,
        field_name: str,
        frequency: Union[str, Frequency],
        provider: DataProvider,
        pct_change: Optional[bool] = None,
        **kwargs
    ) -> Optional[ProviderConfig]:
        """
        Update a provider config.

        Args:
            ticker: Ticker of the instrument
            field_name: Name of the field
            frequency: Data frequency
            provider: Data provider type
            pct_change: If provided, update the percent change transformation setting
            **kwargs: Other fields to update (config, is_active, priority)

        Returns:
            Updated ProviderConfig or None if not found
        """
        field_id = self._get_field_id(ticker, field_name, frequency)

        with self._get_connection() as conn:
            row = conn.execute(
                "SELECT id, config FROM provider_configs WHERE field_id = ? AND provider = ?",
                (field_id, provider.value)
            ).fetchone()

            if not row:
                return None

            config_id = row['id']
            existing_config = json.loads(row['config']) if row['config'] else {}

        allowed_fields = {'config', 'is_active', 'priority'}
        updates = {k: v for k, v in kwargs.items() if k in allowed_fields}

        # Handle pct_change - merge into config
        if pct_change is not None:
            if 'config' in updates and isinstance(updates['config'], dict):
                updates['config']['pct_change'] = pct_change
            else:
                # Merge with existing config
                existing_config['pct_change'] = pct_change
                updates['config'] = existing_config

        if not updates:
            return self._get_provider_config_by_id(config_id)

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

        updated_fields = list(kwargs.keys())
        if pct_change is not None:
            updated_fields.append('pct_change')
        logger.info(
            f"Updated provider config: {provider.value} for {ticker}.{field_name} "
            f"({frequency if isinstance(frequency, str) else frequency.value}), "
            f"updated fields: {updated_fields}"
        )
        return self._get_provider_config_by_id(config_id)

    def delete_provider_config(
        self,
        ticker: str,
        field_name: str,
        frequency: Union[str, Frequency],
        provider: DataProvider
    ) -> bool:
        """Delete a provider config."""
        field_id = self._get_field_id(ticker, field_name, frequency)

        with self._get_connection() as conn:
            cursor = conn.execute(
                "DELETE FROM provider_configs WHERE field_id = ? AND provider = ?",
                (field_id, provider.value)
            )
            conn.commit()

            if cursor.rowcount > 0:
                logger.info(
                    f"Deleted provider config: {provider.value} for {ticker}.{field_name} "
                    f"({frequency if isinstance(frequency, str) else frequency.value})"
                )
                return True
            return False

    # =========================================================================
    # Time Series Data Operations
    # =========================================================================

    def add_time_series_point(
        self,
        ticker: str,
        field_name: str,
        frequency: Union[str, Frequency],
        timestamp: datetime | date,
        value: float,
        metadata: Optional[dict] = None
    ) -> TimeSeriesPoint:
        """
        Add a single time series data point.

        Args:
            ticker: Ticker of the instrument
            field_name: Name of the field
            frequency: Data frequency
            timestamp: Timestamp of the data point
            value: The value
            metadata: Optional additional data

        Returns:
            The created TimeSeriesPoint
        """
        field_id = self._get_field_id(ticker, field_name, frequency)

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
        ticker: str,
        field_name: str,
        frequency: Union[str, Frequency],
        data: list[tuple[datetime | date, float, Optional[dict]]]
    ) -> int:
        """
        Add multiple time series data points efficiently.

        Args:
            ticker: Ticker of the instrument
            field_name: Name of the field
            frequency: Data frequency
            data: List of (timestamp, value, metadata) tuples

        Returns:
            Number of points inserted
        """
        field_id = self._get_field_id(ticker, field_name, frequency)

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

        logger.info(f"Bulk inserted {len(processed_data)} points for {ticker}.{field_name}")
        return len(processed_data)

    def get_time_series(
        self,
        ticker: str,
        field_name: Union[str, list[str]],
        frequency: Union[str, Frequency],
        start_date: Optional[datetime | date] = None,
        end_date: Optional[datetime | date] = None,
        resolve_alias: bool = True
    ) -> pd.DataFrame:
        """
        Get time series data for one or more fields.

        Args:
            ticker: Ticker of the instrument
            field_name: Name of the field(s) - can be a single string or list of strings
            frequency: Data frequency
            start_date: Start of date range (inclusive)
            end_date: End of date range (inclusive)
            resolve_alias: If True and field is an alias, get data from target field

        Returns:
            pandas DataFrame indexed by timestamp with field names as columns
        """
        freq = _to_frequency(frequency)

        # Normalize field_name to a list
        field_names = [field_name] if isinstance(field_name, str) else field_name

        # Convert dates once (outside any loop)
        sd = None
        if start_date:
            sd = start_date
            if isinstance(sd, date) and not isinstance(sd, datetime):
                sd = datetime.combine(sd, datetime.min.time())

        ed = None
        if end_date:
            ed = end_date
            if isinstance(ed, date) and not isinstance(ed, datetime):
                ed = datetime.combine(ed, datetime.max.time())

        # Pre-resolve all field IDs upfront (maps field_id -> field_name)
        field_id_to_name: dict[int, str] = {}
        for fname in field_names:
            field_id = self._get_field_id(ticker, fname, freq)
            actual_field_id = field_id
            if resolve_alias:
                resolved_field = self.resolve_alias(ticker, fname, freq)
                actual_field_id = resolved_field.id
            field_id_to_name[actual_field_id] = fname

        if not field_id_to_name:
            return pd.DataFrame()

        # Build single query for all fields using IN clause
        field_ids = list(field_id_to_name.keys())
        placeholders = ",".join("?" * len(field_ids))
        query = f"SELECT field_id, timestamp, value FROM time_series_data WHERE field_id IN ({placeholders})"
        params: list[Any] = field_ids

        if sd:
            query += " AND timestamp >= ?"
            params.append(sd)

        if ed:
            query += " AND timestamp <= ?"
            params.append(ed)

        query += " ORDER BY timestamp"

        # Execute single query and pivot results
        with self._get_connection() as conn:
            rows = conn.execute(query, params).fetchall()

        # Group data by field_name
        data_by_field: dict[str, tuple[list, list]] = {fname: ([], []) for fname in field_names}
        for row in rows:
            fname = field_id_to_name[row['field_id']]
            data_by_field[fname][0].append(row['timestamp'])
            data_by_field[fname][1].append(row['value'])

        # Build series dict
        series_dict = {}
        for fname, (timestamps, values) in data_by_field.items():
            series_dict[fname] = pd.Series(values, index=pd.DatetimeIndex(timestamps))

        # Combine all series into a DataFrame
        df = pd.DataFrame(series_dict)
        df.index.name = 'timestamp'
        return df

    def get_all_time_series(
        self,
        ticker: str,
        frequency: Union[str, Frequency],
        start_date: Optional[datetime | date] = None,
        end_date: Optional[datetime | date] = None,
        resolve_alias: bool = True,
        include_aliases: bool = True
    ) -> pd.DataFrame:
        """
        Get time series data for all fields of an instrument at a given frequency.

        Args:
            ticker: Ticker of the instrument
            frequency: Data frequency
            start_date: Start of date range (inclusive)
            end_date: End of date range (inclusive)
            resolve_alias: If True and field is an alias, get data from target field
            include_aliases: Whether to include alias fields

        Returns:
            pandas DataFrame indexed by timestamp with all field names as columns

        Example:
            # Get all daily fields for AAPL
            df = db.get_all_time_series("AAPL", "daily")
            # Returns DataFrame with columns like 'price', 'volume', etc.
        """
        freq = _to_frequency(frequency)

        # Get all fields for this instrument at the given frequency
        fields = self.list_fields(ticker=ticker, frequency=freq, include_aliases=include_aliases)

        if not fields:
            # Return empty DataFrame if no fields found
            return pd.DataFrame()

        # Extract field names
        field_names = [f.field_name for f in fields]

        # Use get_time_series to retrieve all fields
        return self.get_time_series(
            ticker=ticker,
            field_name=field_names,
            frequency=freq,
            start_date=start_date,
            end_date=end_date,
            resolve_alias=resolve_alias
        )

    def get_latest_value(
        self,
        ticker: str,
        field_name: str,
        frequency: Union[str, Frequency],
        resolve_alias: bool = True
    ) -> Optional[TimeSeriesPoint]:
        """Get the most recent time series value for a field."""
        freq = _to_frequency(frequency)
        field_id = self._get_field_id(ticker, field_name, freq)

        actual_field_id = field_id
        if resolve_alias:
            resolved_field = self.resolve_alias(ticker, field_name, freq)
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
        ticker: str,
        field_name: str,
        frequency: Union[str, Frequency],
        start_date: Optional[datetime | date] = None,
        end_date: Optional[datetime | date] = None
    ) -> int:
        """
        Delete time series data for a field within a date range.

        Args:
            ticker: Ticker of the instrument
            field_name: Name of the field
            frequency: Data frequency
            start_date: Start of date range (if None, no lower bound)
            end_date: End of date range (if None, no upper bound)

        Returns:
            Number of points deleted
        """
        field_id = self._get_field_id(ticker, field_name, frequency)

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

            logger.info(f"Deleted {cursor.rowcount} time series points for {ticker}.{field_name}")
            return cursor.rowcount

    # =========================================================================
    # Utility Methods
    # =========================================================================

    def get_full_field_info(
        self,
        ticker: str,
        field_name: str,
        frequency: Union[str, Frequency]
    ) -> Optional[dict]:
        """
        Get comprehensive information about a field including its instrument,
        provider configs, and time series summary.
        """
        freq = _to_frequency(frequency)
        field = self.get_field(ticker, field_name, freq)
        if not field:
            return None

        instrument = self.get_instrument(ticker)
        provider_configs = self.get_provider_configs(ticker, field_name, freq, active_only=False)

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
                (field.id,)
            ).fetchone()

        # Get alias info if applicable (already resolved to strings in field object)
        alias_info = None
        if field.alias_ticker:
            alias_info = {
                "target_ticker": field.alias_ticker,
                "target_field": field.alias_field_name,
                "target_frequency": field.alias_frequency
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
            extra_data=json.loads(row['extra_data']) if row['extra_data'] else {},
            created_at=row['created_at'],
            updated_at=row['updated_at']
        )

    def _row_to_field(self, row: sqlite3.Row, conn: Optional[sqlite3.Connection] = None) -> InstrumentField:
        """Convert a database row to an InstrumentField object."""
        # Resolve alias IDs to string identifiers
        alias_ticker = None
        alias_field_name = None
        alias_frequency = None

        if row['alias_field_id'] is not None:
            # Need to look up the alias target to get string identifiers
            if conn is None:
                with self._get_connection() as c:
                    alias_row = c.execute(
                        """
                        SELECT if2.field_name, if2.frequency, i.ticker
                        FROM instrument_fields if2
                        JOIN instruments i ON if2.instrument_id = i.id
                        WHERE if2.id = ?
                        """,
                        (row['alias_field_id'],)
                    ).fetchone()
            else:
                alias_row = conn.execute(
                    """
                    SELECT if2.field_name, if2.frequency, i.ticker
                    FROM instrument_fields if2
                    JOIN instruments i ON if2.instrument_id = i.id
                    WHERE if2.id = ?
                    """,
                    (row['alias_field_id'],)
                ).fetchone()

            if alias_row:
                alias_ticker = alias_row['ticker']
                alias_field_name = alias_row['field_name']
                alias_frequency = alias_row['frequency']

        return InstrumentField(
            id=row['id'],
            instrument_id=row['instrument_id'],
            field_name=row['field_name'],
            frequency=Frequency(row['frequency']),
            description=row['description'],
            unit=row['unit'],
            alias_ticker=alias_ticker,
            alias_field_name=alias_field_name,
            alias_frequency=alias_frequency,
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

    Storable fields are managed via the public API (add_storable_field, etc.)
    and persisted in the database. Default fields are added on first initialization.
    """
    return FinancialTimeSeriesDB(db_path)


def print_deletion_impact(impact: DeletionImpact) -> None:
    """Pretty print a deletion impact report."""
    print("\n" + "=" * 60)
    print(f"DELETION IMPACT REPORT: {impact.target_type.upper()}")
    print("=" * 60)
    print(f"Target: {impact.target_name}")
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
            print(f"  - {c['provider']}")

    if impact.time_series_points_to_delete > 0:
        print(f"\nTime series points to be deleted: {impact.time_series_points_to_delete:,}")

    if impact.warnings:
        print("\nWARNINGS:")
        for warning in impact.warnings:
            print(f"  {warning}")

    print("=" * 60 + "\n")


if __name__ == "__main__":
    # Run example usage
    from example_usage import main
    main()
