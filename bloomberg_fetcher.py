"""
Bloomberg Data Fetcher for Financial Time Series Database

This module provides integration with Bloomberg via the xbbg library to fetch
financial data and store it in the FinancialTimeSeriesDB.

Requires:
    - xbbg: High-level Bloomberg API wrapper (pip install xbbg)
    - A valid Bloomberg Terminal or B-PIPE connection

Author: Claude
"""

from __future__ import annotations

import logging
from datetime import datetime, date, timedelta
from typing import Optional
from dataclasses import dataclass, field

import pandas as pd

from financial_ts_db import (
    FinancialTimeSeriesDB,
    DataProvider,
    ProviderConfig,
    InstrumentField,
)

logger = logging.getLogger(__name__)

# Try to import xbbg - it may not be installed
try:
    from xbbg import blp
    XBBG_AVAILABLE = True
except ImportError:
    XBBG_AVAILABLE = False
    logger.warning(
        "xbbg not installed. Bloomberg fetching will not be available. "
        "Install with: pip install xbbg"
    )


# =============================================================================
# Constants
# =============================================================================

# Bloomberg security type suffixes (for formatting tickers)
SECURITY_TYPE_SUFFIXES = {
    "stock": "Equity",
    "index": "Index",
    "etf": "Equity",
    "bond": "Corp",
    "commodity": "Comdty",
    "currency": "Curncy",
    "mutual_fund": "Equity",
    "crypto": "Curncy",
}

# Periodicity mapping from our format to xbbg format
PERIODICITY_MAP = {
    "DAILY": "DAILY",
    "WEEKLY": "WEEKLY",
    "MONTHLY": "MONTHLY",
    "QUARTERLY": "QUARTERLY",
    "YEARLY": "YEARLY",
}


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class BloombergRequest:
    """Represents a Bloomberg data request."""
    security: str
    fields: list[str]
    start_date: Optional[date] = None
    end_date: Optional[date] = None
    overrides: dict = field(default_factory=dict)
    periodicity: str = "DAILY"


@dataclass
class BloombergDataPoint:
    """A single data point from Bloomberg."""
    security: str
    field: str
    date: date
    value: float
    metadata: dict = field(default_factory=dict)


# =============================================================================
# Bloomberg Fetcher Class
# =============================================================================

class BloombergFetcher:
    """
    Fetches data from Bloomberg and stores it in FinancialTimeSeriesDB.

    Uses the xbbg library for a simpler, more Pythonic interface to Bloomberg.
    No explicit connection management is needed - xbbg handles this automatically.

    Features:
    - Historical data requests via blp.bdh()
    - Reference data requests via blp.bdp()
    - Automatic storage of fetched data into the database
    - Incremental fetching (only fetches data after the last date in DB)
    - Percent change transformation (configurable per field)
    - Verbose mode for debugging fetch issues

    All methods use string identifiers (ticker, field_name, frequency) instead of
    numeric IDs for a cleaner API.

    Example:
        db = FinancialTimeSeriesDB("my_db.sqlite")
        fetcher = BloombergFetcher(db, verbose=True)

        # Fetch historical prices
        fetcher.fetch_historical_data(
            ticker="AAPL",
            field_name="price",
            frequency="daily",
            start_date=date(2024, 1, 1),
            end_date=date(2024, 12, 31)
        )

        # Set up a field with percent change transformation
        setup_bloomberg_field(
            db=db,
            ticker="AAPL",
            field_name="price_change",
            frequency="daily",
            bloomberg_ticker="AAPL US Equity",
            bloomberg_field="PX_LAST",
            pct_change=True
        )
    """

    def __init__(
        self,
        db: FinancialTimeSeriesDB,
        verbose: bool = False
    ):
        """
        Initialize the Bloomberg fetcher.

        Args:
            db: The FinancialTimeSeriesDB instance to store data in
            verbose: If True, enable detailed logging for debugging
        """
        if not XBBG_AVAILABLE:
            raise ImportError(
                "xbbg is not installed. Please install it with: pip install xbbg"
            )

        self.db = db
        self.verbose = verbose

    def _log_verbose(self, message: str) -> None:
        """Log a message if verbose mode is enabled."""
        if self.verbose:
            print(f"[BloombergFetcher] {message}")

    # =========================================================================
    # Data Fetching Methods
    # =========================================================================

    def fetch_historical_data(
        self,
        ticker: str,
        field_name: str,
        frequency: str = "daily",
        start_date: Optional[date] = None,
        end_date: Optional[date] = None,
        store: bool = True
    ) -> list[BloombergDataPoint]:
        """
        Fetch historical data for a field from Bloomberg.

        Automatically checks for existing data in the database:
        - If data exists, starts from the day after the last available date
        - Only stores new data points (no duplicates)
        - Applies percent change transformation if configured in provider config

        Args:
            ticker: Instrument ticker (e.g., "AAPL", "SPX")
            field_name: Name of the field (e.g., "price")
            frequency: Data frequency (e.g., "daily", "weekly")
            start_date: Start date for historical data (used only if no data in DB)
            end_date: End date (defaults to today)
            store: If True, automatically store fetched data in the database

        Returns:
            List of BloombergDataPoint objects (after any transformations)

        Raises:
            ValueError: If no Bloomberg config found for field, or no start_date
                       when DB is empty
        """
        end_date = end_date or date.today()

        self._log_verbose(f"--- fetch_historical_data called ---")
        self._log_verbose(f"  ticker={ticker}, field_name={field_name}, frequency={frequency}")
        self._log_verbose(f"  start_date={start_date}, end_date={end_date}, store={store}")

        # Get the Bloomberg provider config for this field
        self._log_verbose(f"Looking up provider configs for {ticker}.{field_name} ({frequency})...")
        configs = self.db.get_provider_configs(ticker, field_name, frequency, active_only=True)
        self._log_verbose(f"  Found {len(configs)} active config(s)")
        bloomberg_config = None

        for config in configs:
            self._log_verbose(f"  Config: provider={config.provider}, config={config.config}")
            if config.provider == DataProvider.BLOOMBERG:
                bloomberg_config = config
                break

        if not bloomberg_config:
            self._log_verbose("  ERROR: No Bloomberg config found!")
            raise ValueError(
                f"No active Bloomberg config found for {ticker}.{field_name} ({frequency})"
            )

        # Extract Bloomberg-specific settings from config
        bb_ticker = bloomberg_config.config.get("ticker")
        bb_field = bloomberg_config.config.get("field", "PX_LAST")
        overrides = bloomberg_config.config.get("overrides", {})
        periodicity = bloomberg_config.config.get("periodicity", "DAILY")
        pct_change = bloomberg_config.config.get("pct_change", False)

        self._log_verbose(f"Bloomberg config extracted:")
        self._log_verbose(f"  bb_ticker={bb_ticker}")
        self._log_verbose(f"  bb_field={bb_field}")
        self._log_verbose(f"  overrides={overrides}")
        self._log_verbose(f"  periodicity={periodicity}")
        self._log_verbose(f"  pct_change={pct_change}")

        if not bb_ticker:
            self._log_verbose("  ERROR: Bloomberg config missing 'ticker' key!")
            raise ValueError(
                f"Bloomberg config for {ticker}.{field_name} missing 'ticker' in config"
            )

        # Check for existing data to determine actual start date
        self._log_verbose(f"Checking for existing data in DB...")
        latest_point = self.db.get_latest_value(ticker, field_name, frequency, resolve_alias=False)
        self._log_verbose(f"  latest_point={latest_point}")

        if latest_point:
            # Start from the day after the last available date
            if isinstance(latest_point.timestamp, datetime):
                last_date = latest_point.timestamp.date()
            else:
                last_date = latest_point.timestamp

            effective_start = last_date + timedelta(days=1)
            self._log_verbose(f"  Existing data found through {last_date}")
            self._log_verbose(f"  Effective start date: {effective_start}")

            # If we already have data up to or past end_date, nothing to fetch
            if effective_start > end_date:
                self._log_verbose(f"  No new data needed (effective_start > end_date)")
                logger.info(
                    f"No new data to fetch for {ticker}.{field_name}: "
                    f"DB has data through {last_date}"
                )
                return []
        else:
            # No existing data, use provided start_date
            self._log_verbose(f"  No existing data in DB")
            if not start_date:
                self._log_verbose("  ERROR: No start_date provided and no data in DB!")
                raise ValueError(
                    f"No data exists in DB for {ticker}.{field_name} ({frequency}) "
                    f"and no start_date was provided"
                )
            effective_start = start_date
            self._log_verbose(f"  Using provided start_date: {effective_start}")

        self._log_verbose(f"Will request data from {effective_start} to {end_date}")
        logger.info(
            f"Fetching {ticker}.{field_name} from {effective_start} to {end_date}"
        )

        # Make the Bloomberg request using xbbg
        data_points = self._request_historical_data(
            security=bb_ticker,
            field=bb_field,
            start_date=effective_start,
            end_date=end_date,
            overrides=overrides,
            periodicity=periodicity
        )
        self._log_verbose(f"Bloomberg returned {len(data_points)} data points")
        if data_points and self.verbose:
            self._log_verbose(f"  First point: date={data_points[0].date}, value={data_points[0].value}")
            if len(data_points) > 1:
                self._log_verbose(f"  Last point: date={data_points[-1].date}, value={data_points[-1].value}")

        # Apply percent change transformation if configured
        if pct_change and data_points:
            self._log_verbose(f"Applying pct_change transformation...")
            data_points = self._apply_pct_change(data_points, ticker, field_name, frequency)
            self._log_verbose(f"  After pct_change: {len(data_points)} data points")

        # Store in database if requested
        if store and data_points:
            self._log_verbose(f"Storing {len(data_points)} data points in DB...")
            stored_count = self._store_data_points(ticker, field_name, frequency, data_points)
            self._log_verbose(f"  Stored {stored_count} data points")
        elif not store:
            self._log_verbose(f"store=False, skipping database storage")
        elif not data_points:
            self._log_verbose(f"No data points to store")

        self._log_verbose(f"--- fetch_historical_data complete, returning {len(data_points)} points ---")
        return data_points

    def _request_historical_data(
        self,
        security: str,
        field: str,
        start_date: date,
        end_date: date,
        overrides: Optional[dict] = None,
        periodicity: str = "DAILY"
    ) -> list[BloombergDataPoint]:
        """
        Make a historical data request to Bloomberg using xbbg.

        Args:
            security: Bloomberg ticker (e.g., "AAPL US Equity")
            field: Bloomberg field (e.g., "PX_LAST")
            start_date: Start date
            end_date: End date
            overrides: Optional field overrides
            periodicity: Data frequency (DAILY, WEEKLY, etc.)

        Returns:
            List of BloombergDataPoint objects
        """
        self._log_verbose(f"--- _request_historical_data (xbbg) ---")
        self._log_verbose(f"  security={security}")
        self._log_verbose(f"  field={field}")
        self._log_verbose(f"  start_date={start_date}")
        self._log_verbose(f"  end_date={end_date}")
        self._log_verbose(f"  periodicity={periodicity}")
        self._log_verbose(f"  overrides={overrides}")

        try:
            # Build kwargs for blp.bdh
            kwargs = {
                "tickers": security,
                "flds": field,
                "start_date": start_date,
                "end_date": end_date,
                "Per": periodicity,
            }

            # Add overrides if provided (filter out invalid ones)
            if overrides and isinstance(overrides, dict):
                valid_overrides = {
                    k: v for k, v in overrides.items()
                    if isinstance(k, str) and k and v is not None
                }
                if valid_overrides:
                    self._log_verbose(f"  Applying overrides: {valid_overrides}")
                    kwargs.update(valid_overrides)

            self._log_verbose(f"Calling blp.bdh with kwargs: {kwargs}")

            # Make the request
            df = blp.bdh(**kwargs)

            self._log_verbose(f"  Response DataFrame shape: {df.shape}")
            self._log_verbose(f"  Response columns: {list(df.columns)}")

            if df.empty:
                self._log_verbose("  DataFrame is empty - no data returned")
                return []

            # Convert DataFrame to list of BloombergDataPoint
            data_points = []

            # xbbg returns a DataFrame with MultiIndex columns (ticker, field)
            # or simple columns depending on the request
            for idx, row in df.iterrows():
                # idx is the date
                if isinstance(idx, pd.Timestamp):
                    point_date = idx.date()
                else:
                    point_date = idx

                # Get the value - handle both single and multi-column responses
                if isinstance(df.columns, pd.MultiIndex):
                    # Multi-ticker or multi-field response
                    value = row[(security, field)]
                else:
                    # Single ticker/field response
                    if field in df.columns:
                        value = row[field]
                    else:
                        # Column might be named differently
                        value = row.iloc[0]

                # Skip NaN values
                if pd.isna(value):
                    self._log_verbose(f"  Skipping NaN value for {point_date}")
                    continue

                data_points.append(BloombergDataPoint(
                    security=security,
                    field=field,
                    date=point_date,
                    value=float(value)
                ))

            self._log_verbose(f"--- _request_historical_data complete: {len(data_points)} points ---")
            logger.info(f"Fetched {len(data_points)} historical data points for {security}")
            return data_points

        except Exception as e:
            self._log_verbose(f"  ERROR: {type(e).__name__}: {e}")
            logger.error(f"Bloomberg request failed: {e}")
            raise

    def _apply_pct_change(
        self,
        data_points: list[BloombergDataPoint],
        ticker: str,
        field_name: str,
        frequency: str
    ) -> list[BloombergDataPoint]:
        """
        Apply percent change transformation to data points.

        Uses the last value in the database as the base for the first point's
        percent change calculation.
        """
        self._log_verbose(f"--- _apply_pct_change ---")
        self._log_verbose(f"  Input: {len(data_points)} data points")

        if not data_points:
            self._log_verbose(f"  No data points to transform")
            return []

        # Get the last value from DB to use as base for first pct change
        latest_point = self.db.get_latest_value(ticker, field_name, frequency, resolve_alias=False)
        self._log_verbose(f"  Latest point in DB: {latest_point}")

        transformed = []
        prev_value = latest_point.value if latest_point else None
        skipped_count = 0

        for dp in data_points:
            if prev_value is not None and prev_value != 0:
                pct_change_value = ((dp.value - prev_value) / prev_value) * 100
                transformed.append(BloombergDataPoint(
                    security=dp.security,
                    field=dp.field,
                    date=dp.date,
                    value=pct_change_value
                ))
            else:
                skipped_count += 1
            prev_value = dp.value

        self._log_verbose(f"  Output: {len(transformed)} transformed points, {skipped_count} skipped")
        return transformed

    def fetch_reference_data(
        self,
        ticker: str,
        field_name: str,
        frequency: str = "daily",
        store: bool = True
    ) -> Optional[BloombergDataPoint]:
        """
        Fetch current/reference data for a field from Bloomberg.

        This fetches the latest value (not historical) for a field.

        Args:
            ticker: Instrument ticker (e.g., "AAPL", "SPX")
            field_name: Name of the field (e.g., "price")
            frequency: Data frequency (e.g., "daily", "weekly")
            store: If True, automatically store fetched data in the database

        Returns:
            BloombergDataPoint with current value, or None if not available
        """
        self._log_verbose(f"--- fetch_reference_data ---")
        self._log_verbose(f"  ticker={ticker}, field_name={field_name}")

        # Get the Bloomberg provider config for this field
        configs = self.db.get_provider_configs(ticker, field_name, frequency, active_only=True)
        bloomberg_config = None

        for config in configs:
            if config.provider == DataProvider.BLOOMBERG:
                bloomberg_config = config
                break

        if not bloomberg_config:
            raise ValueError(
                f"No active Bloomberg config found for {ticker}.{field_name} ({frequency})"
            )

        bb_ticker = bloomberg_config.config.get("ticker")
        bb_field = bloomberg_config.config.get("field", "PX_LAST")
        overrides = bloomberg_config.config.get("overrides", {})

        if not bb_ticker:
            raise ValueError(
                f"Bloomberg config for {ticker}.{field_name} missing 'ticker' in config"
            )

        # Make the reference data request using xbbg
        data_point = self._request_reference_data(
            security=bb_ticker,
            field=bb_field,
            overrides=overrides
        )

        if store and data_point:
            self._store_data_points(ticker, field_name, frequency, [data_point])

        return data_point

    def _request_reference_data(
        self,
        security: str,
        field: str,
        overrides: Optional[dict] = None
    ) -> Optional[BloombergDataPoint]:
        """Make a reference data request to Bloomberg using xbbg."""
        self._log_verbose(f"--- _request_reference_data (xbbg) ---")
        self._log_verbose(f"  security={security}, field={field}")

        try:
            # Build kwargs for blp.bdp
            kwargs = {
                "tickers": security,
                "flds": field,
            }

            # Add overrides if provided
            if overrides and isinstance(overrides, dict):
                valid_overrides = {
                    k: v for k, v in overrides.items()
                    if isinstance(k, str) and k and v is not None
                }
                if valid_overrides:
                    kwargs.update(valid_overrides)

            self._log_verbose(f"Calling blp.bdp with kwargs: {kwargs}")

            # Make the request
            df = blp.bdp(**kwargs)

            self._log_verbose(f"  Response: {df}")

            if df.empty:
                self._log_verbose("  No data returned")
                return None

            # Get the value
            value = df.loc[security, field] if field in df.columns else df.iloc[0, 0]

            if pd.isna(value):
                return None

            return BloombergDataPoint(
                security=security,
                field=field,
                date=date.today(),
                value=float(value)
            )

        except Exception as e:
            self._log_verbose(f"  ERROR: {type(e).__name__}: {e}")
            logger.error(f"Bloomberg reference request failed: {e}")
            raise

    def get_bloomberg_field_info(
        self,
        ticker: str,
        field_name: str,
        frequency: str = "daily",
    ) -> Optional[dict]:
        """
        Retrieve field info and Bloomberg config for an existing field.

        Returns:
            Dictionary with field info, or None if field/instrument not found
        """
        from financial_ts_db import Frequency

        try:
            freq_enum = Frequency(frequency.lower())
        except ValueError:
            raise ValueError(
                f"Invalid frequency '{frequency}'. Valid options: "
                f"{', '.join(f.value for f in Frequency)}"
            )

        instrument = self.db.get_instrument(ticker)
        if not instrument:
            return None

        field = self.db.get_field(ticker, field_name, freq_enum)
        if not field:
            return None

        configs = self.db.get_provider_configs(ticker, field_name, frequency, active_only=True)
        bloomberg_config = None
        for config in configs:
            if config.provider == DataProvider.BLOOMBERG:
                bloomberg_config = config
                break

        latest_point = self.db.get_latest_value(ticker, field_name, frequency, resolve_alias=False)
        latest_date = None
        if latest_point:
            if isinstance(latest_point.timestamp, datetime):
                latest_date = latest_point.timestamp.date()
            else:
                latest_date = latest_point.timestamp

        return {
            "ticker": ticker,
            "field_name": field_name,
            "frequency": frequency,
            "field": field,
            "instrument": instrument,
            "bloomberg_config": bloomberg_config,
            "latest_date": latest_date,
            "has_data": latest_date is not None,
        }

    def fetch_all_instrument_data(
        self,
        ticker: str,
        start_date: date,
        end_date: Optional[date] = None,
        store: bool = True
    ) -> dict[str, list[BloombergDataPoint]]:
        """
        Fetch data for all Bloomberg-configured fields of an instrument.

        Returns:
            Dict mapping field key (field_name:frequency) to list of BloombergDataPoint
        """
        results = {}
        fields = self.db.list_fields(ticker=ticker, include_aliases=False)

        for field in fields:
            freq_str = field.frequency.value
            configs = self.db.get_provider_configs(
                ticker, field.field_name, freq_str, active_only=True
            )
            has_bloomberg = any(c.provider == DataProvider.BLOOMBERG for c in configs)

            if has_bloomberg:
                field_key = f"{field.field_name}:{freq_str}"
                try:
                    data_points = self.fetch_historical_data(
                        ticker=ticker,
                        field_name=field.field_name,
                        frequency=freq_str,
                        start_date=start_date,
                        end_date=end_date,
                        store=store
                    )
                    results[field_key] = data_points
                    logger.info(
                        f"Fetched {len(data_points)} points for {ticker}.{field.field_name}"
                    )
                except Exception as e:
                    logger.error(f"Error fetching {ticker}.{field.field_name}: {e}")
                    results[field_key] = []

        return results

    # =========================================================================
    # Storage Methods
    # =========================================================================

    def _store_data_points(
        self,
        ticker: str,
        field_name: str,
        frequency: str,
        data_points: list[BloombergDataPoint]
    ) -> int:
        """Store fetched data points in the database."""
        self._log_verbose(f"--- _store_data_points ---")
        self._log_verbose(f"  ticker={ticker}, field_name={field_name}, frequency={frequency}")
        self._log_verbose(f"  data_points count: {len(data_points)}")

        if not data_points:
            self._log_verbose(f"  No data points to store")
            return 0

        bulk_data = [
            (
                datetime.combine(dp.date, datetime.min.time()),
                dp.value,
                {"bloomberg_field": dp.field, "security": dp.security}
            )
            for dp in data_points
        ]

        self._log_verbose(f"  Calling db.add_time_series_bulk with {len(bulk_data)} records...")
        count = self.db.add_time_series_bulk(ticker, field_name, frequency, bulk_data)
        self._log_verbose(f"  add_time_series_bulk returned: {count}")
        logger.info(f"Stored {count} data points for {ticker}.{field_name}")
        return count


# =============================================================================
# Helper Functions
# =============================================================================

def format_bloomberg_ticker(
    ticker: str,
    security_type: str = "stock",
    exchange: str = "US"
) -> str:
    """
    Format a ticker into Bloomberg format.

    Args:
        ticker: Base ticker symbol (e.g., "AAPL")
        security_type: Type of security (stock, index, bond, etc.)
        exchange: Exchange code (e.g., "US", "LN", "JP")

    Returns:
        Bloomberg-formatted ticker (e.g., "AAPL US Equity")
    """
    suffix = SECURITY_TYPE_SUFFIXES.get(security_type.lower(), "Equity")

    if security_type.lower() == "index":
        return f"{ticker} {suffix}"

    return f"{ticker} {exchange} {suffix}"


def create_bloomberg_config(
    bloomberg_ticker: str,
    bloomberg_field: str,
    overrides: Optional[dict] = None,
    pct_change: bool = False,
) -> dict:
    """
    Create a Bloomberg provider config dictionary.

    Args:
        bloomberg_ticker: Full Bloomberg ticker (e.g., "AAPL US Equity")
        bloomberg_field: Bloomberg field name (e.g., "PX_LAST")
        overrides: Optional Bloomberg field overrides
        pct_change: If True, apply percent change transformation before storing

    Returns:
        Config dict ready for use with add_provider_config()
    """
    return {
        "ticker": bloomberg_ticker,
        "field": bloomberg_field,
        "overrides": overrides or {},
        "pct_change": pct_change,
    }


# =============================================================================
# Convenience Functions for Database Integration
# =============================================================================

def get_or_setup_bloomberg_field(
    db: FinancialTimeSeriesDB,
    ticker: str,
    field_name: str,
    frequency: str = "daily",
    bloomberg_ticker: Optional[str] = None,
    bloomberg_field: Optional[str] = None,
    overrides: Optional[dict] = None,
    pct_change: bool = False,
    priority: int = 0
) -> tuple[InstrumentField, ProviderConfig, bool]:
    """
    Get an existing Bloomberg field or create it if it doesn't exist.

    Returns:
        Tuple of (InstrumentField, ProviderConfig, was_created)
    """
    from financial_ts_db import Frequency

    try:
        freq_enum = Frequency(frequency.lower())
    except ValueError:
        raise ValueError(
            f"Invalid frequency '{frequency}'. Valid options: "
            f"{', '.join(f.value for f in Frequency)}"
        )

    instrument = db.get_instrument(ticker)
    if not instrument:
        raise ValueError(f"Instrument not found with ticker: {ticker}")

    existing_field = db.get_field(ticker, field_name, freq_enum)

    if existing_field:
        configs = db.get_provider_configs(ticker, field_name, frequency, active_only=True)
        bloomberg_config = None
        for config in configs:
            if config.provider == DataProvider.BLOOMBERG:
                bloomberg_config = config
                break

        if not bloomberg_config:
            if not bloomberg_ticker or not bloomberg_field:
                raise ValueError(
                    f"Field exists but has no Bloomberg config. "
                    f"You must provide bloomberg_ticker and bloomberg_field to add one."
                )
            config_dict = {
                "ticker": bloomberg_ticker,
                "field": bloomberg_field,
                "overrides": overrides or {},
                "pct_change": pct_change,
            }
            bloomberg_config = db.add_provider_config(
                ticker=ticker,
                field_name=field_name,
                frequency=frequency,
                provider=DataProvider.BLOOMBERG,
                config=config_dict,
                priority=priority
            )

        return existing_field, bloomberg_config, False

    if not bloomberg_ticker or not bloomberg_field:
        raise ValueError(
            f"Field does not exist. "
            f"You must provide bloomberg_ticker and bloomberg_field to create it."
        )

    field, config = setup_bloomberg_field(
        db=db,
        ticker=ticker,
        field_name=field_name,
        frequency=frequency,
        bloomberg_ticker=bloomberg_ticker,
        bloomberg_field=bloomberg_field,
        overrides=overrides,
        pct_change=pct_change,
        priority=priority
    )
    return field, config, True


def setup_bloomberg_field(
    db: FinancialTimeSeriesDB,
    ticker: str,
    field_name: str,
    frequency: str,
    bloomberg_ticker: str,
    bloomberg_field: str,
    overrides: Optional[dict] = None,
    pct_change: bool = False,
    priority: int = 0
) -> tuple[InstrumentField, ProviderConfig]:
    """
    Add a field with Bloomberg provider config in one call.

    Args:
        db: FinancialTimeSeriesDB instance
        ticker: Instrument ticker in database
        field_name: Internal field name (must be in storable fields registry)
        frequency: Data frequency (e.g., "daily", "weekly")
        bloomberg_ticker: Full Bloomberg ticker (e.g., "AAPL US Equity")
        bloomberg_field: Bloomberg field name (e.g., "PX_LAST")
        overrides: Optional Bloomberg field overrides
        pct_change: If True, apply percent change transformation
        priority: Provider priority (lower = higher priority)

    Returns:
        Tuple of (InstrumentField, ProviderConfig)
    """
    from financial_ts_db import Frequency

    try:
        freq_enum = Frequency(frequency.lower())
    except ValueError:
        raise ValueError(
            f"Invalid frequency '{frequency}'. Valid options: "
            f"{', '.join(f.value for f in Frequency)}"
        )

    field = db.add_field(
        ticker=ticker,
        field_name=field_name,
        frequency=freq_enum
    )

    config_dict = {
        "ticker": bloomberg_ticker,
        "field": bloomberg_field,
        "overrides": overrides or {},
        "pct_change": pct_change,
    }

    provider_config = db.add_provider_config(
        ticker=ticker,
        field_name=field_name,
        frequency=frequency,
        provider=DataProvider.BLOOMBERG,
        config=config_dict,
        priority=priority
    )

    return field, provider_config
