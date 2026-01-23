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
from datetime import datetime, date
from typing import Optional, Any
from dataclasses import dataclass, field

from financial_ts_db import (
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
# Constants and Field Mappings
# =============================================================================

# Common Bloomberg field mappings
BLOOMBERG_FIELD_MAPPINGS = {
    # Price fields
    "price": "PX_LAST",
    "open": "PX_OPEN",
    "high": "PX_HIGH",
    "low": "PX_LOW",
    "close": "PX_LAST",
    "volume": "PX_VOLUME",
    "vwap": "EQY_WEIGHTED_AVG_PX",

    # Return fields
    "pct total return": "DAY_TO_DAY_TOT_RETURN_GROSS_DVDS",
    "total return": "TOT_RETURN_INDEX_GROSS_DVDS",

    # Fundamental fields
    "eps": "BEST_EPS",
    "pe ratio": "PE_RATIO",
    "market cap": "CUR_MKT_CAP",
    "dividend yield": "EQY_DVD_YLD_IND",

    # Fixed income
    "yield": "YLD_YTM_MID",
    "duration": "DUR_ADJ_MID",
    "spread": "YAS_BOND_SPREAD_MID",
}

# Reverse mapping for Bloomberg to internal field names
BLOOMBERG_TO_INTERNAL = {v: k for k, v in BLOOMBERG_FIELD_MAPPINGS.items()}

# Bloomberg security type suffixes
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

    Example:
        db = FinancialTimeSeriesDB("my_db.sqlite")
        fetcher = BloombergFetcher(db)

        # Fetch historical prices for a field
        fetcher.fetch_historical_data(
            field_id=price_field.id,
            start_date=date(2024, 1, 1),
            end_date=date(2024, 12, 31)
        )

        # Or fetch for all Bloomberg-configured fields of an instrument
        fetcher.fetch_all_instrument_data(
            instrument_id=apple.id,
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
        field_id: int,
        start_date: date,
        end_date: Optional[date] = None,
        store: bool = True
    ) -> list[BloombergDataPoint]:
        """
        Fetch historical data for a field from Bloomberg.

        Uses the ProviderConfig associated with the field to determine
        the Bloomberg security and field to request.

        Args:
            field_id: ID of the InstrumentField to fetch data for
            start_date: Start date for historical data
            end_date: End date (defaults to today)
            store: If True, automatically store fetched data in the database

        Returns:
            List of BloombergDataPoint objects

        Raises:
            ValueError: If no Bloomberg config found for field
            RuntimeError: If not connected to Bloomberg
        """
        self._ensure_connected()

        end_date = end_date or date.today()

        # Get the Bloomberg provider config for this field
        configs = self.db.get_provider_configs_for_field(field_id, active_only=True)
        bloomberg_config = None

        for config in configs:
            if config.provider == DataProvider.BLOOMBERG:
                bloomberg_config = config
                break

        if not bloomberg_config:
            raise ValueError(f"No active Bloomberg config found for field ID {field_id}")

        # Extract Bloomberg-specific settings from config
        bb_ticker = bloomberg_config.config.get("ticker")
        bb_field = bloomberg_config.config.get("field", "PX_LAST")
        overrides = bloomberg_config.config.get("overrides", {})
        periodicity = bloomberg_config.config.get("periodicity", "DAILY")

        if not bb_ticker:
            raise ValueError(
                f"Bloomberg config for field {field_id} missing 'ticker' in config"
            )

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
            self._store_data_points(field_id, data_points)

        return data_points

    def fetch_reference_data(
        self,
        field_id: int,
        store: bool = True
    ) -> Optional[BloombergDataPoint]:
        """
        Fetch current/reference data for a field from Bloomberg.

        This fetches the latest value (not historical) for a field.

        Args:
            field_id: ID of the InstrumentField to fetch data for
            store: If True, automatically store fetched data in the database

        Returns:
            BloombergDataPoint with current value, or None if not available
        """
        self._ensure_connected()

        # Get the Bloomberg provider config for this field
        configs = self.db.get_provider_configs_for_field(field_id, active_only=True)
        bloomberg_config = None

        for config in configs:
            if config.provider == DataProvider.BLOOMBERG:
                bloomberg_config = config
                break

        if not bloomberg_config:
            raise ValueError(f"No active Bloomberg config found for field ID {field_id}")

        bb_ticker = bloomberg_config.config.get("ticker")
        bb_field = bloomberg_config.config.get("field", "PX_LAST")
        overrides = bloomberg_config.config.get("overrides", {})

        if not bb_ticker:
            raise ValueError(
                f"Bloomberg config for field {field_id} missing 'ticker' in config"
            )

        # Make the reference data request
        data_point = self._request_reference_data(
            security=bb_ticker,
            fields=[bb_field],
            overrides=overrides
        )

        if store and data_point:
            self._store_data_points(field_id, [data_point])

        return data_point

    def fetch_all_instrument_data(
        self,
        instrument_id: int,
        start_date: date,
        end_date: Optional[date] = None,
        store: bool = True
    ) -> dict[int, list[BloombergDataPoint]]:
        """
        Fetch data for all Bloomberg-configured fields of an instrument.

        Args:
            instrument_id: ID of the instrument
            start_date: Start date for historical data
            end_date: End date (defaults to today)
            store: If True, automatically store fetched data

        Returns:
            Dict mapping field_id to list of BloombergDataPoint
        """
        results = {}

        # Get all fields for this instrument
        fields = self.db.list_fields(instrument_id=instrument_id, include_aliases=False)

        for field in fields:
            # Check if this field has a Bloomberg config
            configs = self.db.get_provider_configs_for_field(field.id, active_only=True)

            has_bloomberg = any(c.provider == DataProvider.BLOOMBERG for c in configs)

            if has_bloomberg:
                try:
                    data_points = self.fetch_historical_data(
                        field_id=field.id,
                        start_date=start_date,
                        end_date=end_date,
                        store=store
                    )
                    results[field.id] = data_points
                    logger.info(
                        f"Fetched {len(data_points)} points for field {field.field_name}"
                    )
                except Exception as e:
                    logger.error(f"Error fetching field {field.id}: {e}")
                    results[field.id] = []

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
        field_id: int,
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

        count = self.db.add_time_series_bulk(field_id, bulk_data)
        logger.info(f"Stored {count} data points for field ID {field_id}")
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

def get_bloomberg_field(internal_field: str) -> str:
    """
    Convert an internal field name to Bloomberg field name.

    Args:
        internal_field: Internal field name (e.g., "price", "pct total return")

    Returns:
        Bloomberg field name (e.g., "PX_LAST", "DAY_TO_DAY_TOT_RETURN_GROSS_DVDS")
    """
    return BLOOMBERG_FIELD_MAPPINGS.get(
        internal_field.lower(),
        internal_field.upper()
    )


def get_internal_field(bloomberg_field: str) -> str:
    """
    Convert a Bloomberg field name to internal field name.

    Args:
        bloomberg_field: Bloomberg field name (e.g., "PX_LAST")

    Returns:
        Internal field name (e.g., "price")
    """
    return BLOOMBERG_TO_INTERNAL.get(bloomberg_field, bloomberg_field.lower())


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
    ticker: str,
    field: str = "PX_LAST",
    security_type: str = "stock",
    exchange: str = "US",
    overrides: Optional[dict] = None,
    periodicity: str = "DAILY"
) -> dict:
    """
    Create a Bloomberg provider config dictionary.

    This helper creates the config dict needed for ProviderConfig.

    Args:
        ticker: Base ticker symbol
        field: Bloomberg field name (default: PX_LAST)
        security_type: Type of security
        exchange: Exchange code
        overrides: Bloomberg field overrides
        periodicity: Data periodicity (DAILY, WEEKLY, MONTHLY, etc.)

    Returns:
        Config dict ready for use with add_provider_config()

    Example:
        config = create_bloomberg_config(
            ticker="AAPL",
            field="PX_LAST",
            security_type="stock",
            exchange="US"
        )
        db.add_provider_config(field_id, DataProvider.BLOOMBERG, config)
    """
    return {
        "ticker": format_bloomberg_ticker(ticker, security_type, exchange),
        "field": field,
        "overrides": overrides or {},
        "periodicity": periodicity
    }


# =============================================================================
# Convenience Functions for Database Integration
# =============================================================================

def setup_bloomberg_field(
    db: FinancialTimeSeriesDB,
    instrument_id: int,
    field_name: str,
    frequency,  # Frequency enum
    ticker: str,
    bloomberg_field: Optional[str] = None,
    security_type: str = "stock",
    exchange: str = "US",
    description: str = "",
    priority: int = 0
) -> tuple[InstrumentField, ProviderConfig]:
    """
    Convenience function to add a field with Bloomberg config in one call.

    Args:
        db: FinancialTimeSeriesDB instance
        instrument_id: ID of the instrument
        field_name: Name for the field (e.g., "price")
        frequency: Data frequency (Frequency enum)
        ticker: Base ticker symbol
        bloomberg_field: Bloomberg field name (auto-mapped if not provided)
        security_type: Type of security
        exchange: Exchange code
        description: Field description
        priority: Provider priority

    Returns:
        Tuple of (InstrumentField, ProviderConfig)

    Example:
        field, config = setup_bloomberg_field(
            db=db,
            instrument_id=apple.id,
            field_name="price",
            frequency=Frequency.DAILY,
            ticker="AAPL",
            security_type="stock",
            exchange="US"
        )
    """
    # Add the field
    field = db.add_field(
        instrument_id=instrument_id,
        field_name=field_name,
        frequency=frequency,
        description=description
    )

    # Determine Bloomberg field name
    bb_field = bloomberg_field or get_bloomberg_field(field_name)

    # Create and add provider config
    config_dict = create_bloomberg_config(
        ticker=ticker,
        field=bb_field,
        security_type=security_type,
        exchange=exchange
    )

    provider_config = db.add_provider_config(
        field_id=field.id,
        provider=DataProvider.BLOOMBERG,
        config=config_dict,
        priority=priority
    )

    return field, provider_config
