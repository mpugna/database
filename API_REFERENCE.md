# Financial Time Series Database - API Reference

A SQLite-based database for storing and managing financial time series data with support for multiple instruments, fields, frequencies, and data providers.

## Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [Enums](#enums)
- [Data Classes](#data-classes)
- [Database Creation](#database-creation)
- [Storable Fields Management](#storable-fields-management)
- [Instrument Operations](#instrument-operations)
- [Field Operations](#field-operations)
- [Provider Configuration](#provider-configuration)
- [Time Series Data](#time-series-data)
- [Utility Functions](#utility-functions)

---

## Installation

```python
import pandas as pd
from financial_ts_db import (
    FinancialTimeSeriesDB,
    Frequency,
    InstrumentType,
    DataProvider,
    create_database,
)
```

---

## Quick Start

```python
from datetime import datetime
from financial_ts_db import create_database, Frequency, InstrumentType

# Create database (in-memory or file-based)
db = create_database(":memory:")  # or "financial_data.db"

# Add an instrument
db.add_instrument(
    ticker="AAPL",
    name="Apple Inc.",
    instrument_type=InstrumentType.STOCK
)

# Add a field
db.add_field(ticker="AAPL", field_name="price", frequency=Frequency.DAILY)

# Add time series data
db.add_time_series_point(
    ticker="AAPL",
    field_name="price",
    frequency="daily",
    timestamp=datetime(2024, 1, 1),
    value=185.50
)

# Retrieve data as DataFrame
df = db.get_time_series("AAPL", "price", "daily")
print(df)

db.close()
```

---

## Enums

### Frequency

Supported data frequencies for time series.

```python
class Frequency(Enum):
    TICK = "tick"
    SECOND = "second"
    MINUTE = "minute"
    HOURLY = "hourly"
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"
    QUARTERLY = "quarterly"
    YEARLY = "yearly"
```

### InstrumentType

Types of financial instruments.

```python
class InstrumentType(Enum):
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
```

### DataProvider

Supported data providers for field configurations.

```python
class DataProvider(Enum):
    BLOOMBERG = "bloomberg"
    REUTERS = "reuters"
    YAHOO_FINANCE = "yahoo_finance"
    ALPHA_VANTAGE = "alpha_vantage"
    QUANDL = "quandl"
    FRED = "fred"
    CUSTOM = "custom"
```

---

## Data Classes

### Instrument

Represents a financial instrument.

| Field | Type | Description |
|-------|------|-------------|
| `id` | `int` | Unique identifier |
| `ticker` | `str` | Ticker symbol |
| `name` | `str` | Full name |
| `instrument_type` | `InstrumentType` | Type of instrument |
| `description` | `str` | Description |
| `currency` | `str` | Currency (default: "USD") |
| `exchange` | `str` | Exchange |
| `metadata` | `dict` | Additional metadata |
| `extra_data` | `dict` | Custom extra data |

### InstrumentField

Represents a field for an instrument at a specific frequency.

| Field | Type | Description |
|-------|------|-------------|
| `id` | `int` | Unique identifier |
| `field_name` | `str` | Field name (e.g., "price") |
| `frequency` | `Frequency` | Data frequency |
| `description` | `str` | Description |
| `unit` | `str` | Unit of measurement |
| `alias_ticker` | `str` | Target ticker if alias |
| `alias_field_name` | `str` | Target field if alias |

### ProviderConfig

Configuration for retrieving data from a provider.

| Field | Type | Description |
|-------|------|-------------|
| `id` | `int` | Unique identifier |
| `provider` | `DataProvider` | Data provider |
| `config` | `dict` | Provider-specific configuration |
| `is_active` | `bool` | Whether config is active |
| `priority` | `int` | Priority (lower = higher) |

### TimeSeriesPoint

A single point in a time series.

| Field | Type | Description |
|-------|------|-------------|
| `id` | `int` | Unique identifier |
| `timestamp` | `datetime` | Timestamp |
| `value` | `float` | The value |
| `metadata` | `dict` | Additional metadata |

---

## Database Creation

### create_database

Creates and returns a new database instance.

```python
def create_database(db_path: str | Path = ":memory:") -> FinancialTimeSeriesDB
```

**Example:**
```python
# In-memory database
db = create_database(":memory:")

# File-based database
db = create_database("financial_data.db")
```

---

## Storable Fields Management

Storable fields define which field names are allowed in the database. By default, "price" and "pct total return" are allowed.

### get_storable_fields

Returns all allowed storable field definitions.

```python
def get_storable_fields(self) -> dict[str, StorableFieldDef]
```

**Example:**
```python
fields = db.get_storable_fields()
print(fields.keys())  # ['price', 'pct total return']
```

### get_storable_field

Get a specific storable field definition.

```python
def get_storable_field(self, field_name: str) -> Optional[StorableFieldDef]
```

**Example:**
```python
field_def = db.get_storable_field("price")
print(field_def.description)  # "Last traded price"
```

### add_storable_field

Add a new allowed field name. Raises `ValueError` if field already exists.

```python
def add_storable_field(
    self,
    field_name: str,
    description: str = "",
    metadata: Optional[dict] = None
) -> None
```

**Example:**
```python
db.add_storable_field("volume", "Trading volume", {"unit": "shares"})
db.add_storable_field("eps", "Earnings per share", {"unit": "currency"})
```

### update_storable_field

Update an existing storable field. Raises `ValueError` if field doesn't exist.

```python
def update_storable_field(
    self,
    field_name: str,
    description: Optional[str] = None,
    metadata: Optional[dict] = None
) -> None
```

**Example:**
```python
db.update_storable_field("price", description="Updated description")
db.update_storable_field("price", metadata={"unit": "USD", "precision": 2})
```

### remove_storable_field

Remove a field from the allowed list.

```python
def remove_storable_field(self, field_name: str) -> bool
```

**Example:**
```python
removed = db.remove_storable_field("volume")  # Returns True if removed
```

### is_storable_field

Check if a field name is allowed.

```python
def is_storable_field(self, field_name: str) -> bool
```

**Example:**
```python
if db.is_storable_field("price"):
    print("price is allowed")
```

---

## Instrument Operations

### add_instrument

Add a new financial instrument.

```python
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
) -> Instrument
```

**Example:**
```python
apple = db.add_instrument(
    ticker="AAPL",
    name="Apple Inc.",
    instrument_type=InstrumentType.STOCK,
    currency="USD",
    exchange="NASDAQ",
    metadata={"sector": "Technology"},
    extra_data={"isin": "US0378331005"}
)
```

### get_instrument

Retrieve an instrument by ticker.

```python
def get_instrument(self, ticker: str) -> Optional[Instrument]
```

**Example:**
```python
instrument = db.get_instrument("AAPL")
if instrument:
    print(f"{instrument.name} ({instrument.ticker})")
```

### list_instruments

List instruments with optional filtering.

```python
def list_instruments(
    self,
    instrument_type: Optional[InstrumentType] = None,
    search: Optional[str] = None
) -> list[Instrument]
```

**Example:**
```python
# All instruments
all_instruments = db.list_instruments()

# Only stocks
stocks = db.list_instruments(instrument_type=InstrumentType.STOCK)

# Search by name/ticker
results = db.list_instruments(search="Apple")
```

### update_instrument

Update instrument attributes.

```python
def update_instrument(
    self,
    ticker: str,
    **kwargs
) -> Optional[Instrument]
```

**Example:**
```python
updated = db.update_instrument(
    "AAPL",
    name="Apple Inc. (Updated)",
    metadata={"sector": "Technology", "industry": "Consumer Electronics"}
)
```

### update_instrument_extra_data

Update or merge extra_data for an instrument.

```python
def update_instrument_extra_data(
    self,
    ticker: str,
    extra_data: dict,
    merge: bool = True
) -> Optional[Instrument]
```

**Example:**
```python
# Merge with existing data
db.update_instrument_extra_data("AAPL", {"pe_ratio": 28.5})

# Replace entirely
db.update_instrument_extra_data("AAPL", {"new_field": "value"}, merge=False)
```

### get_instrument_extra_data

Get extra_data or a specific key.

```python
def get_instrument_extra_data(
    self,
    ticker: str,
    key: Optional[str] = None
) -> Optional[Any]
```

**Example:**
```python
# Get all extra_data
all_data = db.get_instrument_extra_data("AAPL")

# Get specific key
pe_ratio = db.get_instrument_extra_data("AAPL", "pe_ratio")
```

### delete_instrument

Delete an instrument and all related data.

```python
def delete_instrument(
    self,
    ticker: str,
    dry_run: bool = False,
    print_output: bool = True
) -> DeletionImpact
```

**Example:**
```python
# Preview deletion impact
impact = db.delete_instrument("AAPL", dry_run=True)
print(f"Would delete {len(impact.fields_to_delete)} fields")

# Actually delete
impact = db.delete_instrument("AAPL")
```

---

## Field Operations

### add_field

Add a field to an instrument.

```python
def add_field(
    self,
    ticker: str,
    field_name: str,
    frequency: Frequency | str,
    unit: str = "",
    metadata: Optional[dict] = None
) -> InstrumentField
```

**Example:**
```python
# Add daily price field
price_field = db.add_field(
    ticker="AAPL",
    field_name="price",
    frequency=Frequency.DAILY,
    unit="USD"
)

# Add quarterly EPS field
eps_field = db.add_field(
    ticker="AAPL",
    field_name="eps",
    frequency="quarterly"
)

# Add field with metadata (for filtering with get_fields_by_metadata)
vol_field = db.add_field(
    ticker="AAPL",
    field_name="vol_25d_call",
    frequency="daily",
    metadata={"type": "volatility", "delta": 25, "option_type": "call"}
)
```

### add_alias_field

Create an alias field that points to another instrument's field.

```python
def add_alias_field(
    self,
    ticker: str,
    field_name: str,
    frequency: Frequency | str,
    target_ticker: str,
    target_field_name: str
) -> InstrumentField
```

**Example:**
```python
# SPX.total_return points to SPXTR.price
alias = db.add_alias_field(
    ticker="SPX",
    field_name="total_return",
    frequency=Frequency.DAILY,
    target_ticker="SPXTR",
    target_field_name="price"
)
```

### get_field

Get a specific field.

```python
def get_field(
    self,
    ticker: str,
    field_name: str,
    frequency: Frequency | str
) -> Optional[InstrumentField]
```

**Example:**
```python
field = db.get_field("AAPL", "price", "daily")
if field:
    print(f"Field: {field.field_name}, Unit: {field.unit}")
```

### list_fields

List fields with optional filtering.

```python
def list_fields(
    self,
    ticker: Optional[str] = None,
    frequency: Optional[Frequency | str] = None,
    include_aliases: bool = True
) -> list[InstrumentField]
```

**Example:**
```python
# All fields for AAPL
aapl_fields = db.list_fields(ticker="AAPL")

# Daily fields only
daily_fields = db.list_fields(ticker="AAPL", frequency="daily")

# Exclude aliases
real_fields = db.list_fields(ticker="AAPL", include_aliases=False)
```

### get_instrument_fields

Get a summary of all fields for an instrument with their frequencies.

```python
def get_instrument_fields(
    self,
    ticker: str,
    include_aliases: bool = True
) -> list[dict]
```

**Example:**
```python
fields = db.get_instrument_fields("AAPL")
for f in fields:
    alias_info = f" -> {f['alias_target']}" if f['is_alias'] else ""
    print(f"{f['field_name']} ({f['frequency']}){alias_info}")

# Output:
#   price (daily)
#   price (weekly)
#   eps (quarterly)
#   total_return (daily) -> SPX.price
```

### get_fields_by_metadata

Get fields of an instrument that match specific metadata criteria.

```python
def get_fields_by_metadata(
    self,
    ticker: str,
    metadata_filter: dict,
    frequency: Optional[Frequency | str] = None,
    include_aliases: bool = True
) -> list[dict]
```

**Example:**
```python
# Add fields with metadata for a volatility surface
db.add_field("AAPL", "vol_25d_call", "daily",
             metadata={"type": "volatility", "delta": 25, "option_type": "call"})
db.add_field("AAPL", "vol_50d_atm", "daily",
             metadata={"type": "volatility", "delta": 50, "option_type": "atm"})
db.add_field("AAPL", "vol_25d_put", "daily",
             metadata={"type": "volatility", "delta": 25, "option_type": "put"})

# Find all volatility fields
vol_fields = db.get_fields_by_metadata("AAPL", {"type": "volatility"})
print(f"Found {len(vol_fields)} volatility fields")

# Find 25 delta call fields
call_fields = db.get_fields_by_metadata(
    "AAPL",
    {"delta": 25, "option_type": "call"}
)

# Filter by frequency as well
daily_vol = db.get_fields_by_metadata(
    "AAPL",
    {"type": "volatility"},
    frequency="daily"
)

# Returned dict structure includes metadata:
# {
#     "field_name": "vol_25d_call",
#     "frequency": "daily",
#     "description": "...",
#     "unit": "...",
#     "metadata": {"type": "volatility", "delta": 25, "option_type": "call"},
#     "is_alias": False,
#     "alias_target": None
# }
```

### resolve_alias

Resolve an alias field to its target field.

```python
def resolve_alias(
    self,
    ticker: str,
    field_name: str,
    frequency: Frequency | str
) -> InstrumentField
```

**Example:**
```python
# Get the actual field that SPX.total_return points to
resolved = db.resolve_alias("SPX", "total_return", "daily")
print(f"Points to: {resolved.field_name}")
```

### update_field

Update field attributes.

```python
def update_field(
    self,
    ticker: str,
    field_name: str,
    frequency: Frequency | str,
    **kwargs
) -> Optional[InstrumentField]
```

**Example:**
```python
updated = db.update_field(
    "AAPL", "price", "daily",
    description="Adjusted closing price",
    unit="USD"
)
```

### delete_field

Delete a field and its data.

```python
def delete_field(
    self,
    ticker: str,
    field_name: str,
    frequency: Frequency | str,
    dry_run: bool = False,
    print_output: bool = True
) -> DeletionImpact
```

**Example:**
```python
# Preview
impact = db.delete_field("AAPL", "price", "daily", dry_run=True)

# Delete
impact = db.delete_field("AAPL", "price", "daily")
```

---

## Provider Configuration

### add_provider_config

Add a data provider configuration for a field.

```python
def add_provider_config(
    self,
    ticker: str,
    field_name: str,
    frequency: Frequency | str,
    provider: DataProvider,
    config: dict,
    is_active: bool = True,
    priority: int = 0,
    pct_change: bool = False
) -> ProviderConfig
```

**Example:**
```python
# Bloomberg config
bb_config = db.add_provider_config(
    ticker="AAPL",
    field_name="price",
    frequency="daily",
    provider=DataProvider.BLOOMBERG,
    config={"ticker": "AAPL US Equity", "field": "PX_LAST"},
    priority=0  # Primary source
)

# Yahoo Finance as backup
yahoo_config = db.add_provider_config(
    ticker="AAPL",
    field_name="price",
    frequency="daily",
    provider=DataProvider.YAHOO_FINANCE,
    config={"symbol": "AAPL"},
    priority=1  # Backup
)
```

### get_provider_configs

Get all provider configs for a field.

```python
def get_provider_configs(
    self,
    ticker: str,
    field_name: str,
    frequency: Frequency | str,
    active_only: bool = True
) -> list[ProviderConfig]
```

**Example:**
```python
configs = db.get_provider_configs("AAPL", "price", "daily")
for cfg in configs:
    print(f"{cfg.provider.value}: priority {cfg.priority}")
```

### update_provider_config

Update a provider configuration.

```python
def update_provider_config(
    self,
    ticker: str,
    field_name: str,
    frequency: Frequency | str,
    provider: DataProvider,
    **kwargs
) -> Optional[ProviderConfig]
```

**Example:**
```python
updated = db.update_provider_config(
    "AAPL", "price", "daily",
    DataProvider.BLOOMBERG,
    priority=2,
    is_active=False
)
```

### delete_provider_config

Delete a provider configuration.

```python
def delete_provider_config(
    self,
    ticker: str,
    field_name: str,
    frequency: Frequency | str,
    provider: DataProvider
) -> bool
```

**Example:**
```python
deleted = db.delete_provider_config(
    "AAPL", "price", "daily",
    DataProvider.YAHOO_FINANCE
)
```

---

## Time Series Data

### add_time_series_point

Add a single data point.

```python
def add_time_series_point(
    self,
    ticker: str,
    field_name: str,
    frequency: Frequency | str,
    timestamp: datetime | date,
    value: float,
    metadata: Optional[dict] = None
) -> TimeSeriesPoint
```

**Example:**
```python
point = db.add_time_series_point(
    ticker="AAPL",
    field_name="price",
    frequency="daily",
    timestamp=datetime(2024, 1, 15),
    value=185.50,
    metadata={"volume": 50000000}
)
```

### add_time_series_bulk

Add multiple data points efficiently.

```python
def add_time_series_bulk(
    self,
    ticker: str,
    field_name: str,
    frequency: Frequency | str,
    data: list[tuple[datetime | date, float, Optional[dict]]]
) -> int
```

**Example:**
```python
data = [
    (datetime(2024, 1, 1), 180.0, {"volume": 50000000}),
    (datetime(2024, 1, 2), 181.5, {"volume": 45000000}),
    (datetime(2024, 1, 3), 182.0, {"volume": 55000000}),
]
count = db.add_time_series_bulk("AAPL", "price", "daily", data)
print(f"Inserted {count} points")
```

### get_time_series

Retrieve time series data as a DataFrame. Supports single or multiple fields.

```python
def get_time_series(
    self,
    ticker: str,
    field_name: str | list[str],
    frequency: Frequency | str,
    start_date: Optional[datetime | date] = None,
    end_date: Optional[datetime | date] = None,
    resolve_alias: bool = True
) -> pd.DataFrame
```

**Example:**
```python
# Single field - returns DataFrame with one column
df = db.get_time_series("AAPL", "price", "daily")
print(df)
#             price
# timestamp
# 2024-01-01  180.0
# 2024-01-02  181.5

# Multiple fields - returns DataFrame with multiple columns
df = db.get_time_series("AAPL", ["price", "volume"], "daily")
print(df)
#             price     volume
# timestamp
# 2024-01-01  180.0  50000000.0
# 2024-01-02  181.5  45000000.0

# With date range
df = db.get_time_series(
    "AAPL", "price", "daily",
    start_date=datetime(2024, 1, 1),
    end_date=datetime(2024, 1, 31)
)

# Iterating over data
for timestamp, row in df.iterrows():
    print(f"{timestamp.date()}: ${row['price']:.2f}")
```

### get_all_time_series

Get all fields for an instrument at a given frequency.

```python
def get_all_time_series(
    self,
    ticker: str,
    frequency: Frequency | str,
    start_date: Optional[datetime | date] = None,
    end_date: Optional[datetime | date] = None,
    resolve_alias: bool = True,
    include_aliases: bool = True
) -> pd.DataFrame
```

**Example:**
```python
# Get all daily data for AAPL
df = db.get_all_time_series("AAPL", "daily")
print(df.columns)  # ['price', 'volume', 'open', 'high', 'low']

# With date range
df = db.get_all_time_series(
    "AAPL", "daily",
    start_date=datetime(2024, 1, 1),
    end_date=datetime(2024, 6, 30)
)
```

### get_latest_value

Get the most recent value for a field.

```python
def get_latest_value(
    self,
    ticker: str,
    field_name: str,
    frequency: Frequency | str,
    resolve_alias: bool = True
) -> Optional[TimeSeriesPoint]
```

**Example:**
```python
latest = db.get_latest_value("AAPL", "price", "daily")
if latest:
    print(f"Latest price: ${latest.value:.2f} ({latest.timestamp.date()})")
```

### delete_time_series

Delete time series data within a range.

```python
def delete_time_series(
    self,
    ticker: str,
    field_name: str,
    frequency: Frequency | str,
    start_date: Optional[datetime | date] = None,
    end_date: Optional[datetime | date] = None
) -> int
```

**Example:**
```python
# Delete all data for a field
deleted = db.delete_time_series("AAPL", "price", "daily")

# Delete data in a range
deleted = db.delete_time_series(
    "AAPL", "price", "daily",
    start_date=datetime(2024, 1, 1),
    end_date=datetime(2024, 1, 31)
)
print(f"Deleted {deleted} points")
```

---

## Utility Functions

### get_full_field_info

Get comprehensive information about a field.

```python
def get_full_field_info(
    self,
    ticker: str,
    field_name: str,
    frequency: Frequency | str
) -> dict
```

**Example:**
```python
info = db.get_full_field_info("AAPL", "price", "daily")
print(f"Field: {info['field']['field_name']}")
print(f"Provider configs: {len(info['provider_configs'])}")
print(f"Data points: {info['time_series_stats']['count']}")
```

### export_to_dict

Export the entire database structure to a dictionary.

```python
def export_to_dict(self) -> dict
```

**Example:**
```python
export = db.export_to_dict()
print(f"Instruments: {len(export['instruments'])}")
print(f"Fields: {len(export['fields'])}")
print(f"Time series points: {export['time_series_point_count']}")
```

### vacuum

Reclaim unused space in the database file.

```python
def vacuum(self) -> None
```

**Example:**
```python
db.vacuum()
```

### close

Close the database connection.

```python
def close(self) -> None
```

**Example:**
```python
db.close()
```

### print_deletion_impact

Print a formatted deletion impact report.

```python
def print_deletion_impact(impact: DeletionImpact) -> None
```

**Example:**
```python
impact = db.delete_instrument("AAPL", dry_run=True, print_output=False)
print_deletion_impact(impact)
```

---

## Complete Example

```python
from datetime import datetime, timedelta
from financial_ts_db import (
    create_database,
    Frequency,
    InstrumentType,
    DataProvider,
)

# Create database
db = create_database("portfolio.db")

# Register custom fields
db.add_storable_field("volume", "Trading volume", {"unit": "shares"})
db.add_storable_field("open", "Opening price")
db.add_storable_field("high", "High price")
db.add_storable_field("low", "Low price")

# Add instruments
db.add_instrument("AAPL", "Apple Inc.", InstrumentType.STOCK, currency="USD")
db.add_instrument("GOOGL", "Alphabet Inc.", InstrumentType.STOCK, currency="USD")

# Add fields
for ticker in ["AAPL", "GOOGL"]:
    for field in ["price", "volume", "open", "high", "low"]:
        db.add_field(ticker, field, Frequency.DAILY)

# Add sample data
base_date = datetime(2024, 1, 1)
for i in range(100):
    ts = base_date + timedelta(days=i)
    db.add_time_series_point("AAPL", "price", "daily", ts, 180 + i * 0.5)
    db.add_time_series_point("AAPL", "volume", "daily", ts, 50000000 + i * 100000)

# Query data
df = db.get_time_series("AAPL", ["price", "volume"], "daily")
print(df.describe())

# Get all fields for an instrument
fields = db.get_instrument_fields("AAPL")
for f in fields:
    print(f"{f['field_name']} ({f['frequency']})")

# Clean up
db.close()
```
