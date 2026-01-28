"""
Generic Data Fetcher for Financial Time Series Database

This module provides a unified interface for fetching financial data from
various providers. It automatically determines which provider to use based
on the configuration stored in the database.

Author: Claude
"""

from __future__ import annotations

import logging
from datetime import date, datetime, timedelta
from typing import Optional, Protocol, Any
from dataclasses import dataclass

from financial_ts_db import (
    FinancialTimeSeriesDB,
    DataProvider,
    ProviderConfig,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Data Point Class
# =============================================================================

@dataclass
class DataPoint:
    """A single data point from any provider."""
    date: date
    value: float
    provider: DataProvider
    metadata: dict = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


# =============================================================================
# Provider Fetcher Protocol
# =============================================================================

class ProviderFetcher(Protocol):
    """Protocol that all provider-specific fetchers must implement."""

    def fetch_historical(
        self,
        config: ProviderConfig,
        start_date: date,
        end_date: date,
    ) -> list[DataPoint]:
        """Fetch historical data using the provider config."""
        ...

    def fetch_current(
        self,
        config: ProviderConfig,
    ) -> Optional[DataPoint]:
        """Fetch current/reference data using the provider config."""
        ...


# =============================================================================
# Provider Implementations
# =============================================================================

class BloombergProviderFetcher:
    """Fetcher implementation for Bloomberg via xbbg."""

    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self._blp = None

    def _get_blp(self):
        """Lazy load xbbg."""
        if self._blp is None:
            try:
                from xbbg import blp
                self._blp = blp
            except ImportError:
                raise ImportError(
                    "xbbg is not installed. Install with: pip install xbbg"
                )
        return self._blp

    def _log(self, message: str):
        if self.verbose:
            print(f"[Bloomberg] {message}")

    def fetch_historical(
        self,
        config: ProviderConfig,
        start_date: date,
        end_date: date,
    ) -> list[DataPoint]:
        """Fetch historical data from Bloomberg."""
        import pandas as pd
        blp = self._get_blp()

        bb_ticker = config.config.get("ticker")
        bb_field = config.config.get("field", "PX_LAST")
        overrides = config.config.get("overrides", {})
        periodicity = config.config.get("periodicity", "DAILY")

        if not bb_ticker:
            raise ValueError("Bloomberg config missing 'ticker'")

        self._log(f"Fetching {bb_ticker} {bb_field} from {start_date} to {end_date}")

        kwargs = {
            "tickers": bb_ticker,
            "flds": bb_field,
            "start_date": start_date,
            "end_date": end_date,
            "Per": periodicity,
        }

        # Add valid overrides
        if overrides and isinstance(overrides, dict):
            valid_overrides = {
                k: v for k, v in overrides.items()
                if isinstance(k, str) and k and v is not None
            }
            if valid_overrides:
                kwargs.update(valid_overrides)

        self._log(f"Calling blp.bdh with: {kwargs}")
        df = blp.bdh(**kwargs)

        self._log(f"Response shape: {df.shape}")

        if df.empty:
            return []

        data_points = []
        for idx, row in df.iterrows():
            point_date = idx.date() if isinstance(idx, pd.Timestamp) else idx

            # Handle multi-index columns
            if isinstance(df.columns, pd.MultiIndex):
                value = row[(bb_ticker, bb_field)]
            else:
                value = row[bb_field] if bb_field in df.columns else row.iloc[0]

            if pd.isna(value):
                continue

            data_points.append(DataPoint(
                date=point_date,
                value=float(value),
                provider=DataProvider.BLOOMBERG,
                metadata={"security": bb_ticker, "field": bb_field}
            ))

        self._log(f"Fetched {len(data_points)} points")
        return data_points

    def fetch_current(
        self,
        config: ProviderConfig,
    ) -> Optional[DataPoint]:
        """Fetch current data from Bloomberg."""
        import pandas as pd
        blp = self._get_blp()

        bb_ticker = config.config.get("ticker")
        bb_field = config.config.get("field", "PX_LAST")

        if not bb_ticker:
            raise ValueError("Bloomberg config missing 'ticker'")

        self._log(f"Fetching current {bb_ticker} {bb_field}")

        df = blp.bdp(tickers=bb_ticker, flds=bb_field)

        if df.empty:
            return None

        value = df.loc[bb_ticker, bb_field] if bb_field in df.columns else df.iloc[0, 0]

        if pd.isna(value):
            return None

        return DataPoint(
            date=date.today(),
            value=float(value),
            provider=DataProvider.BLOOMBERG,
            metadata={"security": bb_ticker, "field": bb_field}
        )


class YahooFinanceProviderFetcher:
    """Fetcher implementation for Yahoo Finance via yfinance."""

    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self._yf = None

    def _get_yf(self):
        """Lazy load yfinance."""
        if self._yf is None:
            try:
                import yfinance as yf
                self._yf = yf
            except ImportError:
                raise ImportError(
                    "yfinance is not installed. Install with: pip install yfinance"
                )
        return self._yf

    def _log(self, message: str):
        if self.verbose:
            print(f"[YahooFinance] {message}")

    def fetch_historical(
        self,
        config: ProviderConfig,
        start_date: date,
        end_date: date,
    ) -> list[DataPoint]:
        """Fetch historical data from Yahoo Finance."""
        import pandas as pd
        yf = self._get_yf()

        yf_ticker = config.config.get("ticker")
        yf_field = config.config.get("field", "Close")  # Open, High, Low, Close, Adj Close, Volume

        if not yf_ticker:
            raise ValueError("Yahoo Finance config missing 'ticker'")

        self._log(f"Fetching {yf_ticker} {yf_field} from {start_date} to {end_date}")

        ticker = yf.Ticker(yf_ticker)
        # Add one day to end_date because yfinance end is exclusive
        df = ticker.history(start=start_date, end=end_date + timedelta(days=1))

        self._log(f"Response shape: {df.shape}")

        if df.empty:
            return []

        if yf_field not in df.columns:
            raise ValueError(f"Field '{yf_field}' not in Yahoo Finance response. "
                           f"Available: {list(df.columns)}")

        data_points = []
        for idx, row in df.iterrows():
            point_date = idx.date() if isinstance(idx, pd.Timestamp) else idx
            value = row[yf_field]

            if pd.isna(value):
                continue

            data_points.append(DataPoint(
                date=point_date,
                value=float(value),
                provider=DataProvider.YAHOO_FINANCE,
                metadata={"ticker": yf_ticker, "field": yf_field}
            ))

        self._log(f"Fetched {len(data_points)} points")
        return data_points

    def fetch_current(
        self,
        config: ProviderConfig,
    ) -> Optional[DataPoint]:
        """Fetch current data from Yahoo Finance."""
        yf = self._get_yf()

        yf_ticker = config.config.get("ticker")
        yf_field = config.config.get("field", "Close")

        if not yf_ticker:
            raise ValueError("Yahoo Finance config missing 'ticker'")

        self._log(f"Fetching current {yf_ticker}")

        ticker = yf.Ticker(yf_ticker)
        info = ticker.info

        # Map field names to Yahoo Finance info keys
        field_map = {
            "Close": "regularMarketPrice",
            "Open": "regularMarketOpen",
            "High": "regularMarketDayHigh",
            "Low": "regularMarketDayLow",
            "Volume": "regularMarketVolume",
            "Adj Close": "regularMarketPrice",
        }

        info_key = field_map.get(yf_field, yf_field)
        value = info.get(info_key)

        if value is None:
            return None

        return DataPoint(
            date=date.today(),
            value=float(value),
            provider=DataProvider.YAHOO_FINANCE,
            metadata={"ticker": yf_ticker, "field": yf_field}
        )


# =============================================================================
# Main Data Fetcher Class
# =============================================================================

class DataFetcher:
    """
    Unified data fetcher that automatically uses the correct provider.

    This class looks up the provider configuration from the database and
    dispatches to the appropriate provider-specific fetcher.

    Features:
    - Automatic provider selection based on database config
    - Incremental fetching (only fetches data after last date in DB)
    - Percent change transformation support
    - Priority-based provider selection (uses highest priority active config)
    - Verbose mode for debugging

    Example:
        db = FinancialTimeSeriesDB("my_db.sqlite")
        fetcher = DataFetcher(db, verbose=True)

        # Fetch data - provider is determined automatically from DB config
        data = fetcher.fetch_historical_data(
            ticker="AAPL",
            field_name="price",
            frequency="daily",
            start_date=date(2024, 1, 1)
        )
    """

    def __init__(
        self,
        db: FinancialTimeSeriesDB,
        verbose: bool = False
    ):
        """
        Initialize the data fetcher.

        Args:
            db: The FinancialTimeSeriesDB instance
            verbose: If True, enable detailed logging for debugging
        """
        self.db = db
        self.verbose = verbose

        # Initialize provider fetchers (lazy-loaded)
        self._provider_fetchers: dict[DataProvider, ProviderFetcher] = {}

    def _log(self, message: str) -> None:
        """Log a message if verbose mode is enabled."""
        if self.verbose:
            print(f"[DataFetcher] {message}")

    def _get_provider_fetcher(self, provider: DataProvider) -> ProviderFetcher:
        """Get or create a fetcher for the specified provider."""
        if provider not in self._provider_fetchers:
            if provider == DataProvider.BLOOMBERG:
                self._provider_fetchers[provider] = BloombergProviderFetcher(self.verbose)
            elif provider == DataProvider.YAHOO_FINANCE:
                self._provider_fetchers[provider] = YahooFinanceProviderFetcher(self.verbose)
            else:
                raise NotImplementedError(
                    f"Provider '{provider.value}' is not yet implemented. "
                    f"Supported providers: bloomberg, yahoo_finance"
                )
        return self._provider_fetchers[provider]

    def _get_active_config(
        self,
        ticker: str,
        field_name: str,
        frequency: str
    ) -> ProviderConfig:
        """Get the highest priority active provider config for a field."""
        configs = self.db.get_provider_configs(
            ticker, field_name, frequency, active_only=True
        )

        if not configs:
            raise ValueError(
                f"No active provider config found for {ticker}.{field_name} ({frequency}). "
                f"Use add_provider_config() to configure a data provider."
            )

        # Return highest priority (lowest priority number)
        return min(configs, key=lambda c: c.priority)

    def fetch_historical_data(
        self,
        ticker: str,
        field_name: str,
        frequency: str = "daily",
        start_date: Optional[date] = None,
        end_date: Optional[date] = None,
        store: bool = True
    ) -> list[DataPoint]:
        """
        Fetch historical data for a field.

        Automatically determines which provider to use based on the
        configuration stored in the database.

        Args:
            ticker: Instrument ticker (e.g., "AAPL", "SPX")
            field_name: Name of the field (e.g., "price")
            frequency: Data frequency (e.g., "daily", "weekly")
            start_date: Start date (used only if no data in DB)
            end_date: End date (defaults to today)
            store: If True, automatically store fetched data in the database

        Returns:
            List of DataPoint objects

        Raises:
            ValueError: If no provider config found or no start_date when DB is empty
        """
        end_date = end_date or date.today()

        self._log(f"--- fetch_historical_data ---")
        self._log(f"  {ticker}.{field_name} ({frequency})")
        self._log(f"  start_date={start_date}, end_date={end_date}")

        # Get the active provider config
        config = self._get_active_config(ticker, field_name, frequency)
        self._log(f"  Using provider: {config.provider.value}")
        self._log(f"  Config: {config.config}")

        # Check for pct_change setting
        pct_change = config.config.get("pct_change", False)

        # Check for existing data to determine actual start date
        self._log(f"Checking for existing data...")
        latest_point = self.db.get_latest_value(
            ticker, field_name, frequency, resolve_alias=False
        )

        if latest_point:
            if isinstance(latest_point.timestamp, datetime):
                last_date = latest_point.timestamp.date()
            else:
                last_date = latest_point.timestamp

            effective_start = last_date + timedelta(days=1)
            self._log(f"  Data exists through {last_date}, starting from {effective_start}")

            if effective_start > end_date:
                self._log(f"  No new data needed")
                return []
        else:
            if not start_date:
                raise ValueError(
                    f"No data exists in DB for {ticker}.{field_name} ({frequency}) "
                    f"and no start_date was provided"
                )
            effective_start = start_date
            self._log(f"  No existing data, using start_date={effective_start}")

        # Get the appropriate fetcher and fetch data
        fetcher = self._get_provider_fetcher(config.provider)
        data_points = fetcher.fetch_historical(config, effective_start, end_date)
        self._log(f"  Fetched {len(data_points)} points from {config.provider.value}")

        # Apply percent change transformation if configured
        if pct_change and data_points:
            self._log(f"  Applying pct_change transformation...")
            data_points = self._apply_pct_change(
                data_points, ticker, field_name, frequency
            )

        # Store in database
        if store and data_points:
            self._log(f"  Storing {len(data_points)} points...")
            count = self._store_data_points(ticker, field_name, frequency, data_points)
            self._log(f"  Stored {count} points")

        return data_points

    def fetch_current_data(
        self,
        ticker: str,
        field_name: str,
        frequency: str = "daily",
        store: bool = True
    ) -> Optional[DataPoint]:
        """
        Fetch current/reference data for a field.

        Args:
            ticker: Instrument ticker
            field_name: Name of the field
            frequency: Data frequency
            store: If True, store in database

        Returns:
            DataPoint with current value, or None if not available
        """
        self._log(f"--- fetch_current_data ---")
        self._log(f"  {ticker}.{field_name}")

        config = self._get_active_config(ticker, field_name, frequency)
        fetcher = self._get_provider_fetcher(config.provider)
        data_point = fetcher.fetch_current(config)

        if store and data_point:
            self._store_data_points(ticker, field_name, frequency, [data_point])

        return data_point

    def fetch_all_instrument_data(
        self,
        ticker: str,
        start_date: date,
        end_date: Optional[date] = None,
        store: bool = True
    ) -> dict[str, list[DataPoint]]:
        """
        Fetch data for all configured fields of an instrument.

        Returns:
            Dict mapping field key (field_name:frequency) to list of DataPoint
        """
        results = {}
        fields = self.db.list_fields(ticker=ticker, include_aliases=False)

        for field in fields:
            freq_str = field.frequency.value
            field_key = f"{field.field_name}:{freq_str}"

            try:
                # Check if there's any provider config
                configs = self.db.get_provider_configs(
                    ticker, field.field_name, freq_str, active_only=True
                )
                if configs:
                    data_points = self.fetch_historical_data(
                        ticker=ticker,
                        field_name=field.field_name,
                        frequency=freq_str,
                        start_date=start_date,
                        end_date=end_date,
                        store=store
                    )
                    results[field_key] = data_points
                    self._log(f"Fetched {len(data_points)} points for {field_key}")
            except Exception as e:
                logger.error(f"Error fetching {ticker}.{field.field_name}: {e}")
                results[field_key] = []

        return results

    def _apply_pct_change(
        self,
        data_points: list[DataPoint],
        ticker: str,
        field_name: str,
        frequency: str
    ) -> list[DataPoint]:
        """Apply percent change transformation to data points."""
        if not data_points:
            return []

        latest_point = self.db.get_latest_value(
            ticker, field_name, frequency, resolve_alias=False
        )

        transformed = []
        prev_value = latest_point.value if latest_point else None

        for dp in data_points:
            if prev_value is not None and prev_value != 0:
                pct_change_value = ((dp.value - prev_value) / prev_value) * 100
                transformed.append(DataPoint(
                    date=dp.date,
                    value=pct_change_value,
                    provider=dp.provider,
                    metadata=dp.metadata
                ))
            prev_value = dp.value

        return transformed

    def _store_data_points(
        self,
        ticker: str,
        field_name: str,
        frequency: str,
        data_points: list[DataPoint]
    ) -> int:
        """Store data points in the database."""
        if not data_points:
            return 0

        bulk_data = [
            (
                datetime.combine(dp.date, datetime.min.time()),
                dp.value,
                dp.metadata
            )
            for dp in data_points
        ]

        return self.db.add_time_series_bulk(ticker, field_name, frequency, bulk_data)


# =============================================================================
# Helper Functions
# =============================================================================

def create_provider_config(
    provider: DataProvider,
    ticker: str,
    field: str,
    pct_change: bool = False,
    **kwargs
) -> dict:
    """
    Create a provider config dictionary.

    Args:
        provider: The data provider
        ticker: Provider-specific ticker (e.g., "AAPL US Equity" for Bloomberg)
        field: Provider-specific field (e.g., "PX_LAST" for Bloomberg)
        pct_change: If True, apply percent change transformation
        **kwargs: Additional provider-specific settings

    Returns:
        Config dict ready for use with add_provider_config()

    Example:
        # Bloomberg config
        config = create_provider_config(
            DataProvider.BLOOMBERG,
            ticker="AAPL US Equity",
            field="PX_LAST"
        )

        # Yahoo Finance config
        config = create_provider_config(
            DataProvider.YAHOO_FINANCE,
            ticker="AAPL",
            field="Close"
        )
    """
    return {
        "ticker": ticker,
        "field": field,
        "pct_change": pct_change,
        **kwargs
    }
