"""
Bloomberg Data Fetcher for Financial Time Series Database

This module provides integration with Bloomberg's blpapi to fetch
financial data and store it in the FinancialTimeSeriesDB.

Requires:
    - blpapi: Bloomberg API Python library
    - A valid Bloomberg Terminal or B-PIPE connection

Author: Claude
"""

from __future__ import annotations

import logging
from datetime import datetime, date, timedelta
from typing import Optional, Any
from dataclasses import dataclass, field

from .financial_ts_db import (
    FinancialTimeSeriesDB,
    DataProvider,
    ProviderConfig,
    InstrumentField,
    TimeSeriesPoint,
)

logger = logging.getLogger(__name__)

# Try to import blpapi - it may not be installed
try:
    import blpapi
    BLPAPI_AVAILABLE = True
except ImportError:
    BLPAPI_AVAILABLE = False
    logger.warning(
        "blpapi not installed. Bloomberg fetching will not be available. "
        "Install with: pip install blpapi"
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
    periodicity: str = "DAILY"  # DAILY, WEEKLY, MONTHLY, QUARTERLY, YEARLY


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

    This class handles:
    - Connection management to Bloomberg API
    - Historical data requests (HistoricalDataRequest)
    - Reference data requests (ReferenceDataRequest)
    - Automatic storage of fetched data into the database

    All methods use string identifiers (ticker, field_name, frequency) instead of
    numeric IDs for a cleaner API.

    Example:
        db = FinancialTimeSeriesDB("my_db.sqlite")
        fetcher = BloombergFetcher(db)

        # Fetch historical prices for a field
        fetcher.fetch_historical_data(
            ticker="AAPL",
            field_name="price",
            frequency="daily",
            start_date=date(2024, 1, 1),
            end_date=date(2024, 12, 31)
        )

        # Or fetch for all Bloomberg-configured fields of an instrument
        fetcher.fetch_all_instrument_data(
            ticker="AAPL",
            start_date=date(2024, 1, 1)
        )
    """

    def __init__(
        self,
        db: FinancialTimeSeriesDB,
        host: str = "localhost",
        port: int = 8194,
        auto_connect: bool = False
    ):
        """
        Initialize the Bloomberg fetcher.

        Args:
            db: The FinancialTimeSeriesDB instance to store data in
            host: Bloomberg API host (default: localhost)
            port: Bloomberg API port (default: 8194 for Desktop API)
            auto_connect: If True, connect to Bloomberg immediately
        """
        if not BLPAPI_AVAILABLE:
            raise ImportError(
                "blpapi is not installed. Please install it with: pip install blpapi"
            )

        self.db = db
        self.host = host
        self.port = port
        self._session: Optional[blpapi.Session] = None
        self._ref_data_service: Optional[Any] = None

        if auto_connect:
            self.connect()

    def connect(self) -> bool:
        """
        Establish connection to Bloomberg API.

        Returns:
            True if connection successful, False otherwise
        """
        if self._session is not None:
            logger.warning("Already connected to Bloomberg")
            return True

        session_options = blpapi.SessionOptions()
        session_options.setServerHost(self.host)
        session_options.setServerPort(self.port)

        self._session = blpapi.Session(session_options)

        if not self._session.start():
            logger.error("Failed to start Bloomberg session")
            self._session = None
            return False

        if not self._session.openService("//blp/refdata"):
            logger.error("Failed to open //blp/refdata service")
            self._session.stop()
            self._session = None
            return False

        self._ref_data_service = self._session.getService("//blp/refdata")
        logger.info(f"Connected to Bloomberg API at {self.host}:{self.port}")
        return True

    def disconnect(self) -> None:
        """Disconnect from Bloomberg API."""
        if self._session is not None:
            self._session.stop()
            self._session = None
            self._ref_data_service = None
            logger.info("Disconnected from Bloomberg API")

    def is_connected(self) -> bool:
        """Check if connected to Bloomberg."""
        return self._session is not None

    def _ensure_connected(self) -> None:
        """Ensure we're connected, raise if not."""
        if not self.is_connected():
            raise RuntimeError(
                "Not connected to Bloomberg. Call connect() first or use auto_connect=True"
            )

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

        Uses the ProviderConfig associated with the field to determine
        the Bloomberg security and field to request.

        Args:
            ticker: Instrument ticker (e.g., "AAPL", "SPX")
            field_name: Name of the field (e.g., "price")
            frequency: Data frequency (e.g., "daily", "weekly")
            start_date: Start date for historical data (required if no data in DB)
            end_date: End date (defaults to today)
            store: If True, automatically store fetched data in the database

        Returns:
            List of BloombergDataPoint objects

        Raises:
            ValueError: If no Bloomberg config found for field
            RuntimeError: If not connected to Bloomberg
        """
        self._ensure_connected()

        end_date = end_date or (date.today() - timedelta(days=1))

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

        # Extract Bloomberg-specific settings from config
        bb_ticker = bloomberg_config.config.get("ticker")
        bb_field = bloomberg_config.config.get("field", "PX_LAST")
        overrides = bloomberg_config.config.get("overrides", {})
        periodicity = bloomberg_config.config.get("periodicity", "DAILY")

        if not bb_ticker:
            raise ValueError(
                f"Bloomberg config for {ticker}.{field_name} missing 'ticker' in config"
            )

        if not start_date:
            raise ValueError("start_date is required for fetch_historical_data")

        # Make the Bloomberg request
        data_points = self._request_historical_data(
            security=bb_ticker,
            fields=[bb_field],
            start_date=start_date,
            end_date=end_date,
            overrides=overrides,
            periodicity=periodicity
        )

        # Store in database if requested
        if store and data_points:
            self._store_data_points(ticker, field_name, frequency, data_points)

        return data_points

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
        self._ensure_connected()

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

        # Make the reference data request
        data_point = self._request_reference_data(
            security=bb_ticker,
            fields=[bb_field],
            overrides=overrides
        )

        if store and data_point:
            self._store_data_points(ticker, field_name, frequency, [data_point])

        return data_point

    def get_bloomberg_field_info(
        self,
        ticker: str,
        field_name: str,
        frequency: str = "daily",
    ) -> Optional[dict]:
        """
        Retrieve field info and Bloomberg config for an existing field.

        This method looks up an existing field by (ticker, field_name, frequency)
        and returns its Bloomberg configuration and the latest date available in the DB.

        Args:
            ticker: Instrument ticker (e.g., "AAPL", "SPX Index")
            field_name: Name of the field (e.g., "price", "pct total return")
            frequency: Data frequency as string (e.g., "daily", "weekly", "monthly")

        Returns:
            Dictionary with field info, or None if field/instrument not found:
            {
                "ticker": str,
                "field_name": str,
                "frequency": str,
                "field": InstrumentField,
                "instrument": Instrument,
                "bloomberg_config": ProviderConfig or None,
                "latest_date": date or None,  # Latest date in DB, None if no data
                "has_data": bool,
            }

        Example:
            info = fetcher.get_bloomberg_field_info(
                ticker="AAPL",
                field_name="price",
                frequency="daily"
            )
            if info:
                print(f"Latest date in DB: {info['latest_date']}")
                print(f"Bloomberg ticker: {info['bloomberg_config'].config['ticker']}")
        """
        from financial_ts_db import Frequency

        # Convert frequency string to enum
        try:
            freq_enum = Frequency(frequency.lower())
        except ValueError:
            raise ValueError(
                f"Invalid frequency '{frequency}'. Valid options: "
                f"{', '.join(f.value for f in Frequency)}"
            )

        # Look up instrument by ticker
        instrument = self.db.get_instrument(ticker)
        if not instrument:
            return None

        # Get the field
        field = self.db.get_field(ticker, field_name, freq_enum)
        if not field:
            return None

        # Get Bloomberg config
        configs = self.db.get_provider_configs(ticker, field_name, frequency, active_only=True)
        bloomberg_config = None
        for config in configs:
            if config.provider == DataProvider.BLOOMBERG:
                bloomberg_config = config
                break

        # Get latest date in DB
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

    def fetch_incremental_data(
        self,
        ticker: str,
        field_name: str,
        frequency: str = "daily",
        default_start_date: Optional[date] = None,
        end_date: Optional[date] = None,
        store: bool = True
    ) -> tuple[list[BloombergDataPoint], dict]:
        """
        Fetch data incrementally, starting from the latest date in DB or a default start date.

        This is a convenience method that:
        1. Looks up the existing field by (ticker, field_name, frequency)
        2. Determines the start date (latest date in DB + 1 day, or default_start_date)
        3. Fetches and optionally stores the data

        Args:
            ticker: Instrument ticker (e.g., "AAPL", "SPX Index")
            field_name: Name of the field (e.g., "price", "pct total return")
            frequency: Data frequency as string (e.g., "daily", "weekly", "monthly")
            default_start_date: Start date to use if no data exists in DB.
                               If None and no data exists, raises ValueError.
            end_date: End date for fetching (defaults to today)
            store: If True, automatically store fetched data in the database

        Returns:
            Tuple of (list of BloombergDataPoint, info dict with field details)

        Raises:
            ValueError: If field not found, no Bloomberg config, or no default_start_date
                       when DB is empty

        Example:
            # Fetch price data, starting from 2020-01-01 if no data exists
            data, info = fetcher.fetch_incremental_data(
                ticker="AAPL",
                field_name="price",
                frequency="daily",
                default_start_date=date(2020, 1, 1)
            )
            print(f"Fetched {len(data)} new points")
            print(f"Started from: {info['start_date_used']}")
        """
        from datetime import timedelta

        # Get field info
        info = self.get_bloomberg_field_info(ticker, field_name, frequency)
        if not info:
            raise ValueError(
                f"Field not found: ticker={ticker}, "
                f"field_name={field_name}, frequency={frequency}"
            )

        if not info["bloomberg_config"]:
            raise ValueError(
                f"No active Bloomberg config found for {ticker}.{field_name} ({frequency})"
            )

        # Determine start date
        if info["has_data"]:
            # Start from the day after the latest data point
            start_date = info["latest_date"] + timedelta(days=1)
        elif default_start_date:
            start_date = default_start_date
        else:
            raise ValueError(
                f"No data exists in DB for {ticker}.{field_name} ({frequency}) and no "
                f"default_start_date was provided"
            )

        end_date = end_date or date.today()

        # Check if we actually need to fetch
        if start_date > end_date:
            logger.info(
                f"No new data to fetch for {ticker}.{field_name}: "
                f"DB is up to date through {info['latest_date']}"
            )
            return [], {
                **info,
                "start_date_used": start_date,
                "end_date_used": end_date,
                "skipped": True,
            }

        # Fetch the data
        data_points = self.fetch_historical_data(
            ticker=ticker,
            field_name=field_name,
            frequency=frequency,
            start_date=start_date,
            end_date=end_date,
            store=store
        )

        return data_points, {
            **info,
            "start_date_used": start_date,
            "end_date_used": end_date,
            "skipped": False,
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

        Args:
            ticker: Ticker of the instrument (e.g., "AAPL", "SPX")
            start_date: Start date for historical data
            end_date: End date (defaults to today)
            store: If True, automatically store fetched data

        Returns:
            Dict mapping field key (field_name:frequency) to list of BloombergDataPoint
        """
        results = {}

        # Get all fields for this instrument
        fields = self.db.list_fields(ticker=ticker, include_aliases=False)

        for field in fields:
            freq_str = field.frequency.value

            # Check if this field has a Bloomberg config
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
    # Bloomberg API Request Methods
    # =========================================================================

    def _request_historical_data(
        self,
        security: str,
        fields: list[str],
        start_date: date,
        end_date: date,
        overrides: Optional[dict] = None,
        periodicity: str = "DAILY"
    ) -> list[BloombergDataPoint]:
        """Make a HistoricalDataRequest to Bloomberg."""
        request = self._ref_data_service.createRequest("HistoricalDataRequest")

        request.getElement("securities").appendValue(security)

        for field in fields:
            request.getElement("fields").appendValue(field)

        request.set("periodicitySelection", periodicity)
        request.set("startDate", start_date.strftime("%Y%m%d"))
        request.set("endDate", end_date.strftime("%Y%m%d"))

        # Apply overrides if any
        if overrides:
            override_element = request.getElement("overrides")
            for key, value in overrides.items():
                override = override_element.appendElement()
                override.setElement("fieldId", key)
                override.setElement("value", str(value))

        self._session.sendRequest(request)

        data_points = []

        while True:
            event = self._session.nextEvent(500)

            for msg in event:
                if msg.hasElement("securityData"):
                    security_data = msg.getElement("securityData")

                    if security_data.hasElement("fieldData"):
                        field_data_array = security_data.getElement("fieldData")

                        for i in range(field_data_array.numValues()):
                            field_data = field_data_array.getValueAsElement(i)

                            point_date = field_data.getElementAsDatetime("date")
                            point_date = date(
                                point_date.year, point_date.month, point_date.day
                            )

                            for field_name in fields:
                                if field_data.hasElement(field_name):
                                    value = field_data.getElementAsFloat(field_name)

                                    data_points.append(BloombergDataPoint(
                                        security=security,
                                        field=field_name,
                                        date=point_date,
                                        value=value
                                    ))

            if event.eventType() == blpapi.Event.RESPONSE:
                break

        logger.info(
            f"Fetched {len(data_points)} historical data points for {security}"
        )
        return data_points

    def _request_reference_data(
        self,
        security: str,
        fields: list[str],
        overrides: Optional[dict] = None
    ) -> Optional[BloombergDataPoint]:
        """Make a ReferenceDataRequest to Bloomberg."""
        request = self._ref_data_service.createRequest("ReferenceDataRequest")

        request.getElement("securities").appendValue(security)

        for field in fields:
            request.getElement("fields").appendValue(field)

        # Apply overrides if any
        if overrides:
            override_element = request.getElement("overrides")
            for key, value in overrides.items():
                override = override_element.appendElement()
                override.setElement("fieldId", key)
                override.setElement("value", str(value))

        self._session.sendRequest(request)

        result = None

        while True:
            event = self._session.nextEvent(500)

            for msg in event:
                if msg.hasElement("securityData"):
                    security_data_array = msg.getElement("securityData")

                    for i in range(security_data_array.numValues()):
                        security_data = security_data_array.getValueAsElement(i)

                        if security_data.hasElement("fieldData"):
                            field_data = security_data.getElement("fieldData")

                            for field_name in fields:
                                if field_data.hasElement(field_name):
                                    value = field_data.getElementAsFloat(field_name)

                                    result = BloombergDataPoint(
                                        security=security,
                                        field=field_name,
                                        date=date.today(),
                                        value=value
                                    )

            if event.eventType() == blpapi.Event.RESPONSE:
                break

        return result

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
        if not data_points:
            return 0

        # Convert to format expected by bulk insert
        bulk_data = [
            (
                datetime.combine(dp.date, datetime.min.time()),
                dp.value,
                {"bloomberg_field": dp.field, "security": dp.security}
            )
            for dp in data_points
        ]

        count = self.db.add_time_series_bulk(ticker, field_name, frequency, bulk_data)
        logger.info(f"Stored {count} data points for {ticker}.{field_name}")
        return count

    # =========================================================================
    # Context Manager Support
    # =========================================================================

    def __enter__(self) -> "BloombergFetcher":
        """Support context manager pattern."""
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Disconnect when exiting context."""
        self.disconnect()


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

    Example:
        >>> format_bloomberg_ticker("AAPL", "stock", "US")
        'AAPL US Equity'
        >>> format_bloomberg_ticker("SPX", "index")
        'SPX Index'
    """
    suffix = SECURITY_TYPE_SUFFIXES.get(security_type.lower(), "Equity")

    if security_type.lower() == "index":
        return f"{ticker} {suffix}"

    return f"{ticker} {exchange} {suffix}"


def create_bloomberg_config(
    bloomberg_ticker: str,
    bloomberg_field: str,
    overrides: Optional[dict] = None,
) -> dict:
    """
    Create a Bloomberg provider config dictionary.

    This helper creates the config dict to be stored in provider_configs table.
    All Bloomberg connection details should be explicitly provided.

    Args:
        bloomberg_ticker: Full Bloomberg ticker (e.g., "AAPL US Equity", "SPX Index")
        bloomberg_field: Bloomberg field name (e.g., "PX_LAST", "TOT_RETURN_INDEX_GROSS_DVDS")
        overrides: Optional Bloomberg field overrides (e.g., {"BEST_FPERIOD_OVERRIDE": "1BF"})

    Returns:
        Config dict ready for use with add_provider_config()

    Example:
        config = create_bloomberg_config(
            bloomberg_ticker="AAPL US Equity",
            bloomberg_field="PX_LAST",
            overrides={"BEST_FPERIOD_OVERRIDE": "1BF"}
        )
        db.add_provider_config(
            ticker="AAPL",
            field_name="price",
            frequency="daily",
            provider=DataProvider.BLOOMBERG,
            config=config
        )
    """
    return {
        "ticker": bloomberg_ticker,
        "field": bloomberg_field,
        "overrides": overrides or {},
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
    priority: int = 0
) -> tuple[InstrumentField, ProviderConfig, bool]:
    """
    Get an existing Bloomberg field or create it if it doesn't exist.

    This function handles the common pattern of:
    - Check if a field already exists for this (ticker, field_name, frequency)
    - If it exists, return it along with its Bloomberg config from the database
    - If it doesn't exist, create it with the provided Bloomberg settings

    The Bloomberg connection details are stored in the database's provider_configs
    table and retrieved when fetching data. Field description is automatically
    retrieved from the storable fields registry.

    Args:
        db: FinancialTimeSeriesDB instance
        ticker: Instrument ticker in database (e.g., "AAPL", "SPX")
        field_name: Internal field name (e.g., "price") - must be in storable fields registry
        frequency: Data frequency as string (e.g., "daily", "weekly", "monthly")
        bloomberg_ticker: Full Bloomberg ticker (e.g., "AAPL US Equity") - required for new fields
        bloomberg_field: Bloomberg field name (e.g., "PX_LAST") - required for new fields
        overrides: Optional Bloomberg field overrides
        priority: Provider priority (used only if creating new field)

    Returns:
        Tuple of (InstrumentField, ProviderConfig, was_created)
        - was_created is True if the field was newly created, False if it already existed

    Raises:
        ValueError: If instrument not found, or if creating new field without bloomberg_ticker/bloomberg_field

    Example:
        # Get existing field (bloomberg params not needed if field exists)
        field, config, created = get_or_setup_bloomberg_field(
            db=db,
            ticker="AAPL",
            field_name="price",
            frequency="daily"
        )

        # Create new field (bloomberg params required)
        field, config, created = get_or_setup_bloomberg_field(
            db=db,
            ticker="AAPL",
            field_name="price",
            frequency="daily",
            bloomberg_ticker="AAPL US Equity",
            bloomberg_field="PX_LAST"
        )
    """
    from financial_ts_db import Frequency

    # Convert frequency string to enum
    try:
        freq_enum = Frequency(frequency.lower())
    except ValueError:
        raise ValueError(
            f"Invalid frequency '{frequency}'. Valid options: "
            f"{', '.join(f.value for f in Frequency)}"
        )

    # Look up instrument by ticker
    instrument = db.get_instrument(ticker)
    if not instrument:
        raise ValueError(f"Instrument not found with ticker: {ticker}")

    # Check if field already exists
    existing_field = db.get_field(ticker, field_name, freq_enum)

    if existing_field:
        # Field exists, get its Bloomberg config
        configs = db.get_provider_configs(ticker, field_name, frequency, active_only=True)
        bloomberg_config = None
        for config in configs:
            if config.provider == DataProvider.BLOOMBERG:
                bloomberg_config = config
                break

        if not bloomberg_config:
            # Field exists but has no Bloomberg config - add one
            if not bloomberg_ticker or not bloomberg_field:
                raise ValueError(
                    f"Field exists but has no Bloomberg config. "
                    f"You must provide bloomberg_ticker and bloomberg_field to add one."
                )
            config_dict = {
                "ticker": bloomberg_ticker,
                "field": bloomberg_field,
                "overrides": overrides or {},
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

    # Field doesn't exist, create it
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
    priority: int = 0
) -> tuple[InstrumentField, ProviderConfig]:
    """
    Add a field with Bloomberg provider config in one call.

    The Bloomberg connection details (ticker, field, overrides) are stored
    in the database and used when fetching data. The field description is
    automatically retrieved from the storable fields registry.

    Args:
        db: FinancialTimeSeriesDB instance
        ticker: Instrument ticker in database (e.g., "AAPL", "SPX")
        field_name: Internal field name (e.g., "price") - must be in storable fields registry
        frequency: Data frequency as string (e.g., "daily", "weekly")
        bloomberg_ticker: Full Bloomberg ticker (e.g., "AAPL US Equity", "SPX Index")
        bloomberg_field: Bloomberg field name (e.g., "PX_LAST", "TOT_RETURN_INDEX_GROSS_DVDS")
        overrides: Optional Bloomberg field overrides (e.g., {"BEST_FPERIOD_OVERRIDE": "1BF"})
        priority: Provider priority (lower = higher priority)

    Returns:
        Tuple of (InstrumentField, ProviderConfig)

    Example:
        field, config = setup_bloomberg_field(
            db=db,
            ticker="AAPL",
            field_name="price",
            frequency="daily",
            bloomberg_ticker="AAPL US Equity",
            bloomberg_field="PX_LAST"
        )
    """
    from financial_ts_db import Frequency

    # Convert frequency string to enum
    try:
        freq_enum = Frequency(frequency.lower())
    except ValueError:
        raise ValueError(
            f"Invalid frequency '{frequency}'. Valid options: "
            f"{', '.join(f.value for f in Frequency)}"
        )

    # Add the field using ticker string directly
    # Description comes from storable fields registry
    field = db.add_field(
        ticker=ticker,
        field_name=field_name,
        frequency=freq_enum
    )

    # Create provider config with Bloomberg connection details
    config_dict = {
        "ticker": bloomberg_ticker,
        "field": bloomberg_field,
        "overrides": overrides or {},
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
