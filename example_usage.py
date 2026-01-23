"""
Example Usage of the Financial Time Series Database

This script demonstrates all major features of the FinancialTimeSeriesDB:
- Creating instruments
- Adding fields at different frequencies
- Creating alias fields
- Configuring data providers
- Storing and retrieving time series data
- Previewing and executing deletions
- Bloomberg data fetching integration
"""

from datetime import datetime, date, timedelta
from financial_ts_db import (
    FinancialTimeSeriesDB,
    Frequency,
    InstrumentType,
    DataProvider,
    print_deletion_impact,
    create_database
)

# Bloomberg integration imports (optional - only if blpapi is installed)
try:
    from bloomberg_fetcher import (
        BloombergFetcher,
        setup_bloomberg_field,
        get_or_setup_bloomberg_field,
        create_bloomberg_config,
        BLPAPI_AVAILABLE
    )
except ImportError:
    BLPAPI_AVAILABLE = False


def main():
    """Demonstrate the full functionality of the Financial Time Series Database."""

    print("=" * 70)
    print("FINANCIAL TIME SERIES DATABASE - EXAMPLE USAGE")
    print("=" * 70)

    # =========================================================================
    # 1. Create Database
    # =========================================================================
    print("\n📁 Creating in-memory database...")
    db = create_database(":memory:")  # Use a file path for persistence
    # db = create_database("financial_data.db")  # Persistent database

    # =========================================================================
    # 2. Add Instruments
    # =========================================================================
    print("\n📊 Adding instruments...")

    # Add Apple stock with metadata and extra_data
    apple = db.add_instrument(
        ticker="AAPL",
        name="Apple Inc.",
        instrument_type=InstrumentType.STOCK,
        description="American multinational technology company",
        currency="USD",
        exchange="NASDAQ",
        metadata={"sector": "Technology", "industry": "Consumer Electronics"},
        extra_data={"isin": "US0378331005", "cusip": "037833100", "sedol": "2046251"}
    )
    print(f"  Added: {apple.ticker} - {apple.name} (ID: {apple.id})")
    print(f"    Extra data: {apple.extra_data}")

    # Add S&P 500 Index
    spx = db.add_instrument(
        ticker="SPX",
        name="S&P 500 Index",
        instrument_type=InstrumentType.INDEX,
        description="Market-cap weighted index of 500 leading US companies",
        currency="USD",
        exchange="INDEX"
    )
    print(f"  Added: {spx.ticker} - {spx.name} (ID: {spx.id})")

    # Add S&P 500 Total Return Index (for alias example)
    spxtr = db.add_instrument(
        ticker="SPXTR",
        name="S&P 500 Total Return Index",
        instrument_type=InstrumentType.INDEX,
        description="S&P 500 with dividends reinvested",
        currency="USD"
    )
    print(f"  Added: {spxtr.ticker} - {spxtr.name} (ID: {spxtr.id})")

    # Add Unemployment Rate (economic indicator)
    unrate = db.add_instrument(
        ticker="UNRATE",
        name="US Unemployment Rate",
        instrument_type=InstrumentType.ECONOMIC_INDICATOR,
        description="Civilian unemployment rate, seasonally adjusted",
        currency="USD",
        metadata={"source": "Bureau of Labor Statistics"}
    )
    print(f"  Added: {unrate.ticker} - {unrate.name} (ID: {unrate.id})")

    # =========================================================================
    # 3. Add Fields to Instruments
    # =========================================================================
    print("\n📋 Adding fields to instruments...")

    # Apple fields
    aapl_price_daily = db.add_field(
        instrument_id=apple.id,
        field_name="PRICE",
        frequency=Frequency.DAILY,
        description="Last traded price",
        unit="USD"
    )
    print(f"  Added: AAPL.PRICE (daily) - ID: {aapl_price_daily.id}")

    aapl_price_weekly = db.add_field(
        instrument_id=apple.id,
        field_name="PRICE",
        frequency=Frequency.WEEKLY,
        description="Weekly closing price",
        unit="USD"
    )
    print(f"  Added: AAPL.PRICE (weekly) - ID: {aapl_price_weekly.id}")

    aapl_eps = db.add_field(
        instrument_id=apple.id,
        field_name="EPS",
        frequency=Frequency.QUARTERLY,
        description="Earnings per share",
        unit="USD"
    )
    print(f"  Added: AAPL.EPS (quarterly) - ID: {aapl_eps.id}")

    # S&P 500 fields
    spx_price = db.add_field(
        instrument_id=spx.id,
        field_name="PRICE",
        frequency=Frequency.DAILY,
        description="Index level",
        unit="points"
    )
    print(f"  Added: SPX.PRICE (daily) - ID: {spx_price.id}")

    # S&P 500 Total Return - actual price field
    spxtr_price = db.add_field(
        instrument_id=spxtr.id,
        field_name="PRICE",
        frequency=Frequency.DAILY,
        description="Total return index level",
        unit="points"
    )
    print(f"  Added: SPXTR.PRICE (daily) - ID: {spxtr_price.id}")

    # Unemployment rate field
    unrate_value = db.add_field(
        instrument_id=unrate.id,
        field_name="VALUE",
        frequency=Frequency.MONTHLY,
        description="Unemployment rate value",
        unit="percent"
    )
    print(f"  Added: UNRATE.VALUE (monthly) - ID: {unrate_value.id}")

    # =========================================================================
    # 4. Create Alias Field
    # =========================================================================
    print("\n🔗 Creating alias field...")

    # Create TOTAL_RETURN field on SPX that points to SPXTR.PRICE
    # This is the alias example from the requirements
    spx_total_return = db.add_alias_field(
        instrument_id=spx.id,
        field_name="TOTAL_RETURN",
        frequency=Frequency.DAILY,
        target_instrument_id=spxtr.id,
        target_field_name="PRICE",
        description="Alias for S&P 500 Total Return Index"
    )
    print(f"  Added alias: SPX.TOTAL_RETURN -> SPXTR.PRICE (ID: {spx_total_return.id})")

    # =========================================================================
    # 5. Configure Data Providers
    # =========================================================================
    print("\n🔌 Configuring data providers...")

    # Bloomberg config for AAPL price
    aapl_bloomberg = db.add_provider_config(
        field_id=aapl_price_daily.id,
        provider=DataProvider.BLOOMBERG,
        config={
            "ticker": "AAPL US Equity",
            "field": "PX_LAST",
            "periodicity": "DAILY"
        },
        priority=0  # Primary source
    )
    print(f"  Added Bloomberg config for AAPL.PRICE: ID {aapl_bloomberg.id}")

    # Yahoo Finance as backup
    aapl_yahoo = db.add_provider_config(
        field_id=aapl_price_daily.id,
        provider=DataProvider.YAHOO_FINANCE,
        config={
            "symbol": "AAPL",
            "data_type": "adj_close"
        },
        priority=1  # Backup source
    )
    print(f"  Added Yahoo Finance config for AAPL.PRICE: ID {aapl_yahoo.id}")

    # FRED config for unemployment rate
    unrate_fred = db.add_provider_config(
        field_id=unrate_value.id,
        provider=DataProvider.FRED,
        config={
            "series_id": "UNRATE",
            "api_key_env": "FRED_API_KEY"
        }
    )
    print(f"  Added FRED config for UNRATE.VALUE: ID {unrate_fred.id}")

    # =========================================================================
    # 6. Add Time Series Data
    # =========================================================================
    print("\n📈 Adding time series data...")

    # Add some sample price data for AAPL
    base_date = datetime(2024, 1, 2)
    aapl_prices = [
        (base_date + timedelta(days=i), 180.0 + i * 0.5, {"volume": 50000000 + i * 1000000})
        for i in range(10)
    ]
    count = db.add_time_series_bulk(aapl_price_daily.id, aapl_prices)
    print(f"  Added {count} daily price points for AAPL")

    # Add data for SPXTR (which SPX.TOTAL_RETURN aliases to)
    spxtr_prices = [
        (base_date + timedelta(days=i), 4800.0 + i * 10, None)
        for i in range(10)
    ]
    count = db.add_time_series_bulk(spxtr_price.id, spxtr_prices)
    print(f"  Added {count} daily price points for SPXTR")

    # Add unemployment data
    for i in range(6):
        date = datetime(2024, i + 1, 1)
        db.add_time_series_point(
            field_id=unrate_value.id,
            timestamp=date,
            value=3.5 + i * 0.1
        )
    print(f"  Added 6 monthly unemployment rate values")

    # =========================================================================
    # 7. Query Data
    # =========================================================================
    print("\n🔍 Querying data...")

    # Get latest AAPL price
    latest_aapl = db.get_latest_value(aapl_price_daily.id)
    print(f"  Latest AAPL price: ${latest_aapl.value:.2f} ({latest_aapl.timestamp.date()})")

    # Get time series with date range
    start = datetime(2024, 1, 4)
    end = datetime(2024, 1, 8)
    aapl_series = db.get_time_series(aapl_price_daily.id, start_date=start, end_date=end)
    print(f"  AAPL prices from {start.date()} to {end.date()}:")
    for point in aapl_series:
        print(f"    {point.timestamp.date()}: ${point.value:.2f}")

    # Get data through alias (SPX.TOTAL_RETURN -> SPXTR.PRICE)
    print(f"\n  Accessing SPX.TOTAL_RETURN (alias for SPXTR.PRICE):")
    tr_data = db.get_time_series(spx_total_return.id, resolve_alias=True)
    for point in tr_data[:3]:  # Show first 3
        print(f"    {point.timestamp.date()}: {point.value:.2f}")
    print(f"    ... ({len(tr_data)} total points)")

    # =========================================================================
    # 8. Get Full Field Information
    # =========================================================================
    print("\n📊 Getting full field information...")

    info = db.get_full_field_info(aapl_price_daily.id)
    print(f"  Field: {info['instrument']['ticker']}.{info['field']['field_name']}")
    print(f"  Frequency: {info['field']['frequency']}")
    print(f"  Provider configs: {len(info['provider_configs'])}")
    print(f"  Time series stats:")
    stats = info['time_series_stats']
    print(f"    - Count: {stats['count']} points")
    print(f"    - Date range: {stats['first_date']} to {stats['last_date']}")
    print(f"    - Value range: ${stats['min_value']:.2f} to ${stats['max_value']:.2f}")

    # =========================================================================
    # 9. Preview Deletion Impact
    # =========================================================================
    print("\n⚠️  Previewing deletion impacts...")

    # Preview what would happen if we delete SPXTR
    print("\n  Previewing deletion of SPXTR (has alias pointing to it):")
    impact = db.preview_deletion("instrument", spxtr.id)
    print_deletion_impact(impact)

    # Preview field deletion
    print("  Previewing deletion of AAPL.PRICE (daily):")
    field_impact = db.preview_deletion("field", aapl_price_daily.id)
    print_deletion_impact(field_impact)

    # =========================================================================
    # 10. Demonstrate Safe Deletion with Dry Run
    # =========================================================================
    print("\n🗑️  Demonstrating dry-run deletion...")

    # Dry run with automatic output - prints impact report to stdout
    print("  Dry run with print_output=True (default):")
    dry_impact = db.delete_instrument(spxtr.id, dry_run=True)
    print(f"  Dry run completed. Would delete {len(dry_impact.fields_to_delete)} fields")
    print(f"  Would delete {len(dry_impact.aliases_to_delete)} alias references")

    # Dry run without output - silent mode for programmatic use
    print("\n  Dry run with print_output=False (silent mode):")
    silent_impact = db.delete_instrument(spxtr.id, dry_run=True, print_output=False)
    print(f"  Silent dry run completed. Impact available programmatically.")

    # Verify nothing was deleted
    spxtr_check = db.get_instrument(spxtr.id)
    print(f"  SPXTR still exists: {spxtr_check is not None}")

    # =========================================================================
    # 11. List and Search Operations
    # =========================================================================
    print("\n📋 List and search operations...")

    # List all instruments
    all_instruments = db.list_instruments()
    print(f"  Total instruments: {len(all_instruments)}")

    # Filter by type
    stocks = db.list_instruments(instrument_type=InstrumentType.STOCK)
    print(f"  Stocks: {len(stocks)}")

    # Search
    sp_instruments = db.list_instruments(search="S&P")
    print(f"  Instruments matching 'S&P': {len(sp_instruments)}")
    for inst in sp_instruments:
        print(f"    - {inst.ticker}: {inst.name}")

    # List fields for an instrument
    apple_fields = db.list_fields(instrument_id=apple.id)
    print(f"\n  Fields for AAPL:")
    for f in apple_fields:
        print(f"    - {f.field_name} ({f.frequency.value})")

    # =========================================================================
    # 12. Update Operations
    # =========================================================================
    print("\n✏️  Update operations...")

    # Update instrument
    db.update_instrument(
        apple.id,
        metadata={"sector": "Technology", "industry": "Consumer Electronics", "updated": True}
    )
    updated_apple = db.get_instrument(apple.id)
    print(f"  Updated AAPL metadata: {updated_apple.metadata}")

    # Update extra_data with merge (adds new fields while preserving existing)
    db.update_instrument_extra_data(
        apple.id,
        {"pe_ratio": 28.5, "dividend_yield": 0.5, "market_cap": 2800000000000}
    )
    updated_apple = db.get_instrument(apple.id)
    print(f"  Updated AAPL extra_data: {updated_apple.extra_data}")

    # Add more extra_data (merges with existing)
    db.update_instrument_extra_data(apple.id, {"beta": 1.25, "analyst_rating": "Buy"})
    print(f"  Merged AAPL extra_data: {db.get_instrument_extra_data(apple.id)}")

    # Get a specific extra_data field
    pe_ratio = db.get_instrument_extra_data(apple.id, "pe_ratio")
    print(f"  AAPL P/E ratio: {pe_ratio}")

    # Update field
    db.update_field(aapl_price_daily.id, description="Last traded closing price (updated)")
    updated_field = db.get_field(aapl_price_daily.id)
    print(f"  Updated field description: {updated_field.description}")

    # Update provider config
    db.update_provider_config(aapl_bloomberg.id, priority=2)  # Demote Bloomberg
    db.update_provider_config(aapl_yahoo.id, priority=0)  # Promote Yahoo
    configs = db.get_provider_configs_for_field(aapl_price_daily.id)
    print(f"  Updated provider priorities:")
    for c in configs:
        print(f"    - {c.provider.value}: priority {c.priority}")

    # =========================================================================
    # 13. Export Database
    # =========================================================================
    print("\n💾 Exporting database...")

    export = db.export_to_dict()
    print(f"  Instruments: {len(export['instruments'])}")
    print(f"  Fields: {len(export['fields'])}")
    print(f"  Provider configs: {len(export['provider_configs'])}")
    print(f"  Time series points: {export['time_series_point_count']}")

    # =========================================================================
    # 14. Demonstrate Actual Deletion
    # =========================================================================
    print("\n🗑️  Demonstrating actual deletion...")

    # Delete a provider config (simple deletion)
    deleted = db.delete_provider_config(aapl_yahoo.id)
    print(f"  Deleted Yahoo config: {deleted}")

    # Delete UNRATE and all its data
    unrate_impact = db.delete_instrument(unrate.id, dry_run=False)
    print(f"  Deleted UNRATE: {unrate_impact.time_series_points_to_delete} data points removed")

    # Verify deletion
    unrate_check = db.get_instrument_by_ticker("UNRATE")
    print(f"  UNRATE exists: {unrate_check is not None}")

    # =========================================================================
    # Summary
    # =========================================================================
    print("\n" + "=" * 70)
    print("EXAMPLE COMPLETED SUCCESSFULLY")
    print("=" * 70)

    # Final stats
    final_instruments = db.list_instruments()
    final_fields = db.list_fields()
    print(f"\nFinal database state:")
    print(f"  - Instruments: {len(final_instruments)}")
    print(f"  - Fields: {len(final_fields)}")

    print("\nKey features demonstrated:")
    print("  ✓ Instrument management (CRUD)")
    print("  ✓ Multi-frequency field support")
    print("  ✓ Alias fields")
    print("  ✓ Data provider configuration")
    print("  ✓ Time series storage and retrieval")
    print("  ✓ Deletion impact preview")
    print("  ✓ Cascade deletions with warnings")
    print("  ✓ Search and filtering")
    print("  ✓ Database export")

    db.close()


def bloomberg_example():
    """
    Demonstrate Bloomberg data fetching integration.

    This example shows how to:
    - Set up fields with Bloomberg provider configs
    - Use the BloombergFetcher to retrieve historical data
    - Store Bloomberg data in the time series database

    NOTE: This requires:
    1. blpapi Python package installed (pip install blpapi)
    2. Bloomberg Terminal or B-PIPE connection available
    """
    if not BLPAPI_AVAILABLE:
        print("\n" + "=" * 70)
        print("BLOOMBERG EXAMPLE - SKIPPED")
        print("=" * 70)
        print("\nBloomberg API (blpapi) is not installed.")
        print("To enable Bloomberg integration:")
        print("  1. Install blpapi: pip install blpapi")
        print("  2. Ensure Bloomberg Terminal is running, or")
        print("  3. Configure B-PIPE connection")
        print("\nShowing example code structure instead...\n")
        _show_bloomberg_example_code()
        return

    print("\n" + "=" * 70)
    print("BLOOMBERG DATA FETCHING EXAMPLE")
    print("=" * 70)

    # Create database
    db = create_database(":memory:")

    # =========================================================================
    # 1. Set up instruments and fields with Bloomberg configs
    # =========================================================================
    print("\n📊 Setting up instruments with Bloomberg configs...")

    # Add Apple stock
    apple = db.add_instrument(
        ticker="AAPL",
        name="Apple Inc.",
        instrument_type=InstrumentType.STOCK,
        currency="USD",
        exchange="NASDAQ"
    )
    print(f"  Added: {apple.ticker} (ID: {apple.id})")

    # Use setup_bloomberg_field to add field with Bloomberg config
    # Bloomberg connection details are explicitly provided and stored in DB
    price_field, price_config = setup_bloomberg_field(
        db=db,
        instrument_id=apple.id,
        field_name="price",
        frequency=Frequency.DAILY,
        bloomberg_ticker="AAPL US Equity",  # Full Bloomberg ticker
        bloomberg_field="PX_LAST",           # Bloomberg field name
        description="Daily closing price from Bloomberg"
    )
    print(f"  Added field: {price_field.field_name} with Bloomberg config")
    print(f"    Config stored in DB: {price_config.config}")

    # Add Microsoft with manual config for more control
    msft = db.add_instrument(
        ticker="MSFT",
        name="Microsoft Corporation",
        instrument_type=InstrumentType.STOCK,
        currency="USD",
        exchange="NASDAQ"
    )

    msft_price = db.add_field(
        instrument_id=msft.id,
        field_name="price",
        frequency=Frequency.DAILY,
        description="Daily closing price"
    )

    # Create Bloomberg config and store in DB
    # All connection details are stored in provider_configs table
    msft_bb_config = create_bloomberg_config(
        bloomberg_ticker="MSFT US Equity",
        bloomberg_field="PX_LAST",
        overrides={"BEST_FPERIOD_OVERRIDE": "1BF"}
    )

    db.add_provider_config(
        field_id=msft_price.id,
        provider=DataProvider.BLOOMBERG,
        config=msft_bb_config,
        priority=0
    )
    print(f"  Added: {msft.ticker} with Bloomberg config: {msft_bb_config}")

    # =========================================================================
    # 2. Fetch data from Bloomberg
    # =========================================================================
    print("\n📡 Fetching data from Bloomberg...")

    try:
        # Create fetcher with context manager for automatic cleanup
        with BloombergFetcher(db, auto_connect=True) as fetcher:

            # Fetch historical data for Apple
            print(f"\n  Fetching AAPL historical prices...")
            aapl_data = fetcher.fetch_historical_data(
                field_id=price_field.id,
                start_date=date(2024, 1, 1),
                end_date=date(2024, 1, 31),
                store=True  # Automatically store in database
            )
            print(f"    Fetched {len(aapl_data)} data points")

            # Fetch current/reference data
            print(f"\n  Fetching AAPL current price...")
            current = fetcher.fetch_reference_data(
                field_id=price_field.id,
                store=True
            )
            if current:
                print(f"    Current price: ${current.value:.2f}")

            # Fetch all Bloomberg-configured fields for an instrument
            print(f"\n  Fetching all MSFT Bloomberg data...")
            all_msft_data = fetcher.fetch_all_instrument_data(
                instrument_id=msft.id,
                start_date=date(2024, 1, 1),
                store=True
            )
            for field_id, points in all_msft_data.items():
                field = db.get_field(field_id)
                print(f"    {field.field_name}: {len(points)} points")

    except RuntimeError as e:
        print(f"\n  ⚠️  Bloomberg connection failed: {e}")
        print("  Make sure Bloomberg Terminal is running.")

    # =========================================================================
    # 3. Query the stored data
    # =========================================================================
    print("\n📈 Querying stored Bloomberg data...")

    # Get time series from database
    aapl_series = db.get_time_series(
        field_id=price_field.id,
        start_date=date(2024, 1, 1),
        end_date=date(2024, 1, 10)
    )

    if aapl_series:
        print(f"  AAPL prices (first 5):")
        for point in aapl_series[:5]:
            print(f"    {point.timestamp.date()}: ${point.value:.2f}")
    else:
        print("  No data stored (Bloomberg connection may have failed)")

    # =========================================================================
    # 4. Show stored provider config
    # =========================================================================
    print("\n🔧 Retrieving Bloomberg config from database:")

    # The Bloomberg connection details are stored in provider_configs
    configs = db.get_provider_configs_for_field(price_field.id)
    for cfg in configs:
        if cfg.provider == DataProvider.BLOOMBERG:
            print(f"  Field ID: {cfg.field_id}")
            print(f"  Provider: {cfg.provider.value}")
            print(f"  Config from DB: {cfg.config}")
            print(f"    - ticker: {cfg.config.get('ticker')}")
            print(f"    - field: {cfg.config.get('field')}")
            print(f"    - overrides: {cfg.config.get('overrides')}")

    print("\n" + "=" * 70)
    print("BLOOMBERG EXAMPLE COMPLETED")
    print("=" * 70)

    db.close()


def _show_bloomberg_example_code():
    """Show example code when blpapi is not available."""
    example_code = '''
# Example: Bloomberg integration with database-stored config
#
# Key concept: All Bloomberg connection details (ticker, field, overrides)
# are stored in the database's provider_configs table. When fetching data,
# the fetcher reads the config from the DB - nothing is hardcoded.

from financial_ts_db import (
    FinancialTimeSeriesDB, Frequency, InstrumentType, DataProvider
)
from bloomberg_fetcher import (
    BloombergFetcher,
    setup_bloomberg_field,
    get_or_setup_bloomberg_field,
    create_bloomberg_config
)
from datetime import date

# =============================================================================
# Setup: Store Bloomberg connection details in database
# =============================================================================

db = FinancialTimeSeriesDB("financial_data.db")

# Add instrument
apple = db.add_instrument(
    ticker="AAPL",
    name="Apple Inc.",
    instrument_type=InstrumentType.STOCK
)

# Option 1: Use setup_bloomberg_field
# Bloomberg config is stored in provider_configs table
field, config = setup_bloomberg_field(
    db=db,
    instrument_id=apple.id,
    field_name="price",
    frequency=Frequency.DAILY,
    bloomberg_ticker="AAPL US Equity",  # Stored in DB
    bloomberg_field="PX_LAST",           # Stored in DB
    overrides={"BEST_FPERIOD_OVERRIDE": "1BF"}  # Stored in DB
)

# Option 2: Manual configuration
price_field = db.add_field(
    instrument_id=apple.id,
    field_name="total_return",
    frequency=Frequency.DAILY
)

# Create config dict and store in database
bb_config = create_bloomberg_config(
    bloomberg_ticker="AAPL US Equity",
    bloomberg_field="TOT_RETURN_INDEX_GROSS_DVDS",
    overrides={}
)

db.add_provider_config(
    field_id=price_field.id,
    provider=DataProvider.BLOOMBERG,
    config=bb_config  # Stored in DB
)

# =============================================================================
# Fetch: Read config from database and call Bloomberg
# =============================================================================

with BloombergFetcher(db) as fetcher:

    # The fetcher reads Bloomberg config from the database
    # No hardcoded field mappings - everything comes from provider_configs

    # Method 1: Fetch by field_id (reads config from DB)
    data = fetcher.fetch_historical_data(
        field_id=field.id,
        start_date=date(2024, 1, 1),
        end_date=date(2024, 12, 31),
        store=True
    )
    print(f"Fetched {len(data)} data points")

    # Method 2: Incremental fetch by ticker/field_name/frequency
    # Looks up field in DB, gets Bloomberg config, fetches only new data
    data, info = fetcher.fetch_incremental_data(
        ticker="AAPL",
        field_name="price",
        frequency="daily",
        default_start_date=date(2020, 1, 1)
    )
    print(f"Start date used: {info['start_date_used']}")
    print(f"Bloomberg config from DB: {info['bloomberg_config'].config}")

# =============================================================================
# Query: Data is stored with reference to the field
# =============================================================================

series = db.get_time_series(field.id)
for point in series[-5:]:
    print(f"{point.timestamp.date()}: ${point.value:.2f}")

# You can also check what config is stored
configs = db.get_provider_configs_for_field(field.id)
for cfg in configs:
    print(f"Provider: {cfg.provider.value}, Config: {cfg.config}")
'''
    print(example_code)


if __name__ == "__main__":
    main()

    # Run Bloomberg example if user wants to see it
    print("\n" + "-" * 70)
    print("Run bloomberg_example() to see Bloomberg integration demo")
    print("-" * 70)
