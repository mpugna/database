"""
Unit tests for the Financial Time Series Database.

Run with: python -m pytest test_financial_ts_db.py -v
Or simply: python test_financial_ts_db.py
"""

import unittest
from datetime import datetime, date, timedelta

from financial_ts_db import (
    FinancialTimeSeriesDB,
    Frequency,
    InstrumentType,
    DataProvider,
    create_database,
    DEFAULT_STORABLE_FIELDS,
)


# Helper to create database with validation disabled for tests
def create_test_database():
    """Create a test database with storable field validation disabled."""
    return create_database(":memory:", storable_fields=set())


class TestInstrumentOperations(unittest.TestCase):
    """Test instrument CRUD operations."""

    def setUp(self):
        self.db = create_test_database()

    def tearDown(self):
        self.db.close()

    def test_add_instrument(self):
        """Test adding a new instrument."""
        instrument = self.db.add_instrument(
            ticker="AAPL",
            name="Apple Inc.",
            instrument_type=InstrumentType.STOCK,
            currency="USD",
            exchange="NASDAQ"
        )

        self.assertIsNotNone(instrument.id)
        self.assertEqual(instrument.ticker, "AAPL")
        self.assertEqual(instrument.name, "Apple Inc.")
        self.assertEqual(instrument.instrument_type, InstrumentType.STOCK)

    def test_get_instrument_by_ticker(self):
        """Test retrieving instrument by ticker."""
        self.db.add_instrument(
            ticker="SPX",
            name="S&P 500",
            instrument_type=InstrumentType.INDEX
        )

        instrument = self.db.get_instrument_by_ticker("SPX")
        self.assertIsNotNone(instrument)
        self.assertEqual(instrument.name, "S&P 500")

    def test_update_instrument(self):
        """Test updating instrument attributes."""
        instrument = self.db.add_instrument(
            ticker="TEST",
            name="Test Instrument",
            instrument_type=InstrumentType.OTHER
        )

        updated = self.db.update_instrument(
            instrument.id,
            name="Updated Name",
            description="New description"
        )

        self.assertEqual(updated.name, "Updated Name")
        self.assertEqual(updated.description, "New description")

    def test_list_instruments_with_filter(self):
        """Test listing instruments with filters."""
        self.db.add_instrument("AAPL", "Apple", InstrumentType.STOCK)
        self.db.add_instrument("GOOGL", "Google", InstrumentType.STOCK)
        self.db.add_instrument("SPX", "S&P 500", InstrumentType.INDEX)

        stocks = self.db.list_instruments(instrument_type=InstrumentType.STOCK)
        self.assertEqual(len(stocks), 2)

        indices = self.db.list_instruments(instrument_type=InstrumentType.INDEX)
        self.assertEqual(len(indices), 1)

    def test_delete_instrument(self):
        """Test deleting an instrument."""
        instrument = self.db.add_instrument(
            ticker="DELETE_ME",
            name="To Delete",
            instrument_type=InstrumentType.OTHER
        )

        impact = self.db.delete_instrument(instrument.id)
        self.assertEqual(impact.target_id, instrument.id)

        result = self.db.get_instrument(instrument.id)
        self.assertIsNone(result)

    def test_add_instrument_with_extra_data(self):
        """Test adding an instrument with extra_data JSON field."""
        extra_data = {
            "sector": "Technology",
            "industry": "Consumer Electronics",
            "beta": 1.2,
            "market_cap": 2500000000000
        }
        instrument = self.db.add_instrument(
            ticker="AAPL",
            name="Apple Inc.",
            instrument_type=InstrumentType.STOCK,
            extra_data=extra_data
        )

        self.assertIsNotNone(instrument.id)
        self.assertEqual(instrument.extra_data["sector"], "Technology")
        self.assertEqual(instrument.extra_data["beta"], 1.2)
        self.assertEqual(instrument.extra_data["market_cap"], 2500000000000)

    def test_update_instrument_extra_data_merge(self):
        """Test merging extra_data with existing data."""
        instrument = self.db.add_instrument(
            ticker="TEST",
            name="Test Instrument",
            instrument_type=InstrumentType.STOCK,
            extra_data={"field1": "value1", "field2": "value2"}
        )

        # Merge new data
        updated = self.db.update_instrument_extra_data(
            instrument.id,
            {"field3": "value3", "field2": "updated"}
        )

        self.assertEqual(updated.extra_data["field1"], "value1")
        self.assertEqual(updated.extra_data["field2"], "updated")
        self.assertEqual(updated.extra_data["field3"], "value3")

    def test_update_instrument_extra_data_replace(self):
        """Test replacing extra_data entirely."""
        instrument = self.db.add_instrument(
            ticker="TEST",
            name="Test Instrument",
            instrument_type=InstrumentType.STOCK,
            extra_data={"old_field": "old_value"}
        )

        # Replace entirely
        updated = self.db.update_instrument_extra_data(
            instrument.id,
            {"new_field": "new_value"},
            merge=False
        )

        self.assertNotIn("old_field", updated.extra_data)
        self.assertEqual(updated.extra_data["new_field"], "new_value")

    def test_get_instrument_extra_data(self):
        """Test getting extra_data and specific keys."""
        instrument = self.db.add_instrument(
            ticker="TEST",
            name="Test Instrument",
            instrument_type=InstrumentType.STOCK,
            extra_data={"key1": "value1", "key2": 42}
        )

        # Get all extra_data
        all_data = self.db.get_instrument_extra_data(instrument.id)
        self.assertEqual(all_data, {"key1": "value1", "key2": 42})

        # Get specific key
        value = self.db.get_instrument_extra_data(instrument.id, "key1")
        self.assertEqual(value, "value1")

        # Get non-existent key
        missing = self.db.get_instrument_extra_data(instrument.id, "missing")
        self.assertIsNone(missing)

    def test_update_instrument_with_extra_data(self):
        """Test updating extra_data through update_instrument method."""
        instrument = self.db.add_instrument(
            ticker="TEST",
            name="Test Instrument",
            instrument_type=InstrumentType.STOCK
        )

        updated = self.db.update_instrument(
            instrument.id,
            extra_data={"custom_field": "custom_value"}
        )

        self.assertEqual(updated.extra_data["custom_field"], "custom_value")


class TestFieldOperations(unittest.TestCase):
    """Test field CRUD operations."""

    def setUp(self):
        self.db = create_test_database()
        self.instrument = self.db.add_instrument(
            ticker="TEST",
            name="Test Instrument",
            instrument_type=InstrumentType.STOCK
        )

    def tearDown(self):
        self.db.close()

    def test_add_field(self):
        """Test adding a field to an instrument."""
        field = self.db.add_field(
            instrument_id=self.instrument.id,
            field_name="PRICE",
            frequency=Frequency.DAILY,
            unit="USD"
        )

        self.assertIsNotNone(field.id)
        self.assertEqual(field.field_name, "PRICE")
        self.assertEqual(field.frequency, Frequency.DAILY)

    def test_add_multiple_frequencies(self):
        """Test adding same field at different frequencies."""
        daily = self.db.add_field(
            self.instrument.id, "PRICE", Frequency.DAILY
        )
        weekly = self.db.add_field(
            self.instrument.id, "PRICE", Frequency.WEEKLY
        )

        self.assertNotEqual(daily.id, weekly.id)

        fields = self.db.list_fields(instrument_id=self.instrument.id)
        self.assertEqual(len(fields), 2)

    def test_get_field_by_name(self):
        """Test retrieving field by name and frequency."""
        self.db.add_field(
            self.instrument.id, "EPS", Frequency.QUARTERLY
        )

        field = self.db.get_field_by_name(
            self.instrument.id, "EPS", Frequency.QUARTERLY
        )
        self.assertIsNotNone(field)
        self.assertEqual(field.field_name, "EPS")


class TestAliasFields(unittest.TestCase):
    """Test alias field functionality."""

    def setUp(self):
        self.db = create_test_database()

        # Create two instruments
        self.spx = self.db.add_instrument(
            ticker="SPX",
            name="S&P 500",
            instrument_type=InstrumentType.INDEX
        )
        self.spxtr = self.db.add_instrument(
            ticker="SPXTR",
            name="S&P 500 Total Return",
            instrument_type=InstrumentType.INDEX
        )

        # Add price field to SPXTR
        self.spxtr_price = self.db.add_field(
            self.spxtr.id, "PRICE", Frequency.DAILY
        )

    def tearDown(self):
        self.db.close()

    def test_create_alias_field(self):
        """Test creating an alias field."""
        alias = self.db.add_alias_field(
            instrument_id=self.spx.id,
            field_name="TOTAL_RETURN",
            frequency=Frequency.DAILY,
            target_instrument_id=self.spxtr.id,
            target_field_name="PRICE"
        )

        self.assertIsNotNone(alias.id)
        self.assertEqual(alias.alias_field_id, self.spxtr_price.id)

    def test_resolve_alias(self):
        """Test resolving alias to target field."""
        alias = self.db.add_alias_field(
            instrument_id=self.spx.id,
            field_name="TOTAL_RETURN",
            frequency=Frequency.DAILY,
            target_instrument_id=self.spxtr.id,
            target_field_name="PRICE"
        )

        resolved = self.db.resolve_alias(alias.id)
        self.assertEqual(resolved.id, self.spxtr_price.id)

    def test_alias_data_access(self):
        """Test accessing data through alias."""
        # Add data to target field
        self.db.add_time_series_point(
            self.spxtr_price.id,
            datetime(2024, 1, 1),
            4800.0
        )

        # Create alias
        alias = self.db.add_alias_field(
            instrument_id=self.spx.id,
            field_name="TOTAL_RETURN",
            frequency=Frequency.DAILY,
            target_instrument_id=self.spxtr.id,
            target_field_name="PRICE"
        )

        # Access through alias
        data = self.db.get_time_series(alias.id, resolve_alias=True)
        self.assertEqual(len(data), 1)
        self.assertEqual(data.iloc[0], 4800.0)

    def test_delete_target_removes_alias(self):
        """Test that deleting target instrument removes aliases."""
        alias = self.db.add_alias_field(
            instrument_id=self.spx.id,
            field_name="TOTAL_RETURN",
            frequency=Frequency.DAILY,
            target_instrument_id=self.spxtr.id,
            target_field_name="PRICE"
        )

        # Preview deletion
        impact = self.db.preview_deletion("instrument", self.spxtr.id)
        self.assertEqual(len(impact.aliases_to_delete), 1)

        # Actually delete
        self.db.delete_instrument(self.spxtr.id)

        # Verify alias is gone
        result = self.db.get_field(alias.id)
        self.assertIsNone(result)


class TestProviderConfig(unittest.TestCase):
    """Test data provider configuration."""

    def setUp(self):
        self.db = create_test_database()
        self.instrument = self.db.add_instrument(
            "AAPL", "Apple", InstrumentType.STOCK
        )
        self.field = self.db.add_field(
            self.instrument.id, "PRICE", Frequency.DAILY
        )

    def tearDown(self):
        self.db.close()

    def test_add_provider_config(self):
        """Test adding provider configuration."""
        config = self.db.add_provider_config(
            field_id=self.field.id,
            provider=DataProvider.BLOOMBERG,
            config={"ticker": "AAPL US Equity", "field": "PX_LAST"}
        )

        self.assertIsNotNone(config.id)
        self.assertEqual(config.provider, DataProvider.BLOOMBERG)
        self.assertEqual(config.config["ticker"], "AAPL US Equity")

    def test_multiple_providers_with_priority(self):
        """Test multiple providers with priority ordering."""
        bloomberg = self.db.add_provider_config(
            self.field.id, DataProvider.BLOOMBERG, {"ticker": "AAPL US"}, priority=0
        )
        yahoo = self.db.add_provider_config(
            self.field.id, DataProvider.YAHOO_FINANCE, {"symbol": "AAPL"}, priority=1
        )

        configs = self.db.get_provider_configs_for_field(self.field.id)
        self.assertEqual(len(configs), 2)
        self.assertEqual(configs[0].provider, DataProvider.BLOOMBERG)  # Priority 0 first

    def test_cascade_delete_with_field(self):
        """Test that provider configs are deleted with field."""
        self.db.add_provider_config(
            self.field.id, DataProvider.BLOOMBERG, {}
        )

        self.db.delete_field(self.field.id)

        # Field and its configs should be gone
        configs = self.db.get_provider_configs_for_field(self.field.id)
        self.assertEqual(len(configs), 0)


class TestTimeSeries(unittest.TestCase):
    """Test time series data operations."""

    def setUp(self):
        self.db = create_test_database()
        self.instrument = self.db.add_instrument(
            "AAPL", "Apple", InstrumentType.STOCK
        )
        self.field = self.db.add_field(
            self.instrument.id, "PRICE", Frequency.DAILY
        )

    def tearDown(self):
        self.db.close()

    def test_add_single_point(self):
        """Test adding a single data point."""
        point = self.db.add_time_series_point(
            self.field.id,
            datetime(2024, 1, 1),
            180.0
        )

        self.assertIsNotNone(point.id)
        self.assertEqual(point.value, 180.0)

    def test_add_bulk_points(self):
        """Test bulk insert of data points."""
        data = [
            (datetime(2024, 1, i), 180.0 + i, None)
            for i in range(1, 11)
        ]

        count = self.db.add_time_series_bulk(self.field.id, data)
        self.assertEqual(count, 10)

        series = self.db.get_time_series(self.field.id)
        self.assertEqual(len(series), 10)

    def test_get_time_series_with_range(self):
        """Test retrieving time series with date range."""
        for i in range(10):
            self.db.add_time_series_point(
                self.field.id,
                datetime(2024, 1, i + 1),
                100.0 + i
            )

        start = datetime(2024, 1, 3)
        end = datetime(2024, 1, 7)
        series = self.db.get_time_series(self.field.id, start_date=start, end_date=end)

        self.assertEqual(len(series), 5)  # Days 3, 4, 5, 6, 7

    def test_get_latest_value(self):
        """Test getting most recent value."""
        self.db.add_time_series_point(self.field.id, datetime(2024, 1, 1), 100.0)
        self.db.add_time_series_point(self.field.id, datetime(2024, 1, 10), 110.0)
        self.db.add_time_series_point(self.field.id, datetime(2024, 1, 5), 105.0)

        latest = self.db.get_latest_value(self.field.id)
        self.assertEqual(latest.value, 110.0)

    def test_delete_time_series_range(self):
        """Test deleting time series within a range."""
        for i in range(10):
            self.db.add_time_series_point(
                self.field.id,
                datetime(2024, 1, i + 1),
                100.0 + i
            )

        # Delete middle range
        deleted = self.db.delete_time_series(
            self.field.id,
            start_date=datetime(2024, 1, 4),
            end_date=datetime(2024, 1, 6)
        )

        self.assertEqual(deleted, 3)

        remaining = self.db.get_time_series(self.field.id)
        self.assertEqual(len(remaining), 7)

    def test_upsert_behavior(self):
        """Test that duplicate timestamps are updated (upsert)."""
        self.db.add_time_series_point(self.field.id, datetime(2024, 1, 1), 100.0)
        self.db.add_time_series_point(self.field.id, datetime(2024, 1, 1), 200.0)

        series = self.db.get_time_series(self.field.id)
        self.assertEqual(len(series), 1)
        self.assertEqual(series.iloc[0], 200.0)


class TestDeletionPreview(unittest.TestCase):
    """Test deletion preview/simulation functionality."""

    def setUp(self):
        self.db = create_test_database()

    def tearDown(self):
        self.db.close()

    def test_preview_instrument_deletion(self):
        """Test previewing instrument deletion impact."""
        instrument = self.db.add_instrument(
            "TEST", "Test", InstrumentType.STOCK
        )
        field = self.db.add_field(instrument.id, "PRICE", Frequency.DAILY)
        self.db.add_provider_config(field.id, DataProvider.BLOOMBERG, {})
        self.db.add_time_series_point(field.id, datetime(2024, 1, 1), 100.0)

        impact = self.db.preview_deletion("instrument", instrument.id)

        self.assertEqual(len(impact.fields_to_delete), 1)
        self.assertEqual(len(impact.provider_configs_to_delete), 1)
        self.assertEqual(impact.time_series_points_to_delete, 1)

        # Verify nothing was actually deleted
        self.assertIsNotNone(self.db.get_instrument(instrument.id))

    def test_dry_run_deletion(self):
        """Test dry run deletion."""
        instrument = self.db.add_instrument(
            "TEST", "Test", InstrumentType.STOCK
        )

        impact = self.db.delete_instrument(instrument.id, dry_run=True, print_output=False)

        # Still exists after dry run
        self.assertIsNotNone(self.db.get_instrument(instrument.id))

    def test_dry_run_deletion_prints_output(self):
        """Test that dry run deletion prints impact to stdout."""
        import io
        import sys

        instrument = self.db.add_instrument(
            "TEST", "Test Instrument", InstrumentType.STOCK
        )
        field = self.db.add_field(instrument.id, "PRICE", Frequency.DAILY)

        # Capture stdout
        captured_output = io.StringIO()
        sys.stdout = captured_output

        try:
            self.db.delete_instrument(instrument.id, dry_run=True, print_output=True)
        finally:
            sys.stdout = sys.__stdout__

        output = captured_output.getvalue()
        self.assertIn("DELETION IMPACT REPORT", output)
        self.assertIn("Test Instrument", output)
        self.assertIn("PRICE", output)

    def test_dry_run_field_deletion_prints_output(self):
        """Test that dry run field deletion prints impact to stdout."""
        import io
        import sys

        instrument = self.db.add_instrument(
            "TEST", "Test Instrument", InstrumentType.STOCK
        )
        field = self.db.add_field(instrument.id, "PRICE", Frequency.DAILY)

        # Capture stdout
        captured_output = io.StringIO()
        sys.stdout = captured_output

        try:
            self.db.delete_field(field.id, dry_run=True, print_output=True)
        finally:
            sys.stdout = sys.__stdout__

        output = captured_output.getvalue()
        self.assertIn("DELETION IMPACT REPORT", output)
        self.assertIn("PRICE", output)


class TestCascadeDeletes(unittest.TestCase):
    """Test cascade deletion behavior."""

    def setUp(self):
        self.db = create_test_database()

    def tearDown(self):
        self.db.close()

    def test_instrument_deletion_cascades(self):
        """Test that deleting instrument cascades to fields, configs, and data."""
        instrument = self.db.add_instrument("TEST", "Test", InstrumentType.STOCK)
        field1 = self.db.add_field(instrument.id, "PRICE", Frequency.DAILY)
        field2 = self.db.add_field(instrument.id, "VOLUME", Frequency.DAILY)
        self.db.add_provider_config(field1.id, DataProvider.BLOOMBERG, {})
        self.db.add_time_series_point(field1.id, datetime(2024, 1, 1), 100.0)

        self.db.delete_instrument(instrument.id)

        # All related data should be gone
        self.assertIsNone(self.db.get_instrument(instrument.id))
        self.assertIsNone(self.db.get_field(field1.id))
        self.assertIsNone(self.db.get_field(field2.id))

    def test_field_deletion_cascades(self):
        """Test that deleting field cascades to configs and data."""
        instrument = self.db.add_instrument("TEST", "Test", InstrumentType.STOCK)
        field = self.db.add_field(instrument.id, "PRICE", Frequency.DAILY)
        config = self.db.add_provider_config(field.id, DataProvider.BLOOMBERG, {})
        self.db.add_time_series_point(field.id, datetime(2024, 1, 1), 100.0)

        self.db.delete_field(field.id)

        # Field and related data gone, but instrument remains
        self.assertIsNone(self.db.get_field(field.id))
        self.assertIsNotNone(self.db.get_instrument(instrument.id))


class TestStorableFields(unittest.TestCase):
    """Test storable fields validation functionality."""

    def test_default_storable_fields(self):
        """Test that default storable fields are set correctly."""
        db = create_database()
        storable = db.get_storable_fields()

        self.assertIn("price", storable)
        self.assertIn("pct total return", storable)
        db.close()

    def test_add_field_with_valid_name(self):
        """Test adding a field with a valid storable field name."""
        db = create_database()
        instrument = db.add_instrument("TEST", "Test", InstrumentType.STOCK)

        # "price" is in the default storable fields
        field = db.add_field(instrument.id, "price", Frequency.DAILY)
        self.assertIsNotNone(field.id)

        # Case insensitive - "PRICE" should also work
        field2 = db.add_field(instrument.id, "PRICE", Frequency.WEEKLY)
        self.assertIsNotNone(field2.id)
        db.close()

    def test_add_field_with_invalid_name_raises(self):
        """Test that adding a field with an invalid name raises ValueError."""
        db = create_database()
        instrument = db.add_instrument("TEST", "Test", InstrumentType.STOCK)

        with self.assertRaises(ValueError) as context:
            db.add_field(instrument.id, "INVALID_FIELD", Frequency.DAILY)

        self.assertIn("not in the allowed storable fields", str(context.exception))
        db.close()

    def test_add_storable_field(self):
        """Test adding a new storable field."""
        db = create_database()
        instrument = db.add_instrument("TEST", "Test", InstrumentType.STOCK)

        # Initially should fail
        with self.assertRaises(ValueError):
            db.add_field(instrument.id, "volume", Frequency.DAILY)

        # Add to storable fields
        db.add_storable_field("volume")
        self.assertTrue(db.is_storable_field("volume"))

        # Now it should work
        field = db.add_field(instrument.id, "volume", Frequency.DAILY)
        self.assertIsNotNone(field.id)
        db.close()

    def test_remove_storable_field(self):
        """Test removing a storable field."""
        db = create_database()

        self.assertTrue(db.is_storable_field("price"))
        result = db.remove_storable_field("price")
        self.assertTrue(result)
        self.assertFalse(db.is_storable_field("price"))

        # Removing again returns False
        result = db.remove_storable_field("price")
        self.assertFalse(result)
        db.close()

    def test_set_storable_fields(self):
        """Test replacing all storable fields."""
        db = create_database()

        new_fields = {"open", "high", "low", "close", "volume"}
        db.set_storable_fields(new_fields)

        storable = db.get_storable_fields()
        self.assertEqual(storable, new_fields)
        self.assertFalse(db.is_storable_field("price"))
        self.assertTrue(db.is_storable_field("volume"))
        db.close()

    def test_empty_storable_fields_disables_validation(self):
        """Test that empty storable fields set disables validation."""
        db = create_database(storable_fields=set())
        instrument = db.add_instrument("TEST", "Test", InstrumentType.STOCK)

        # Any field name should work
        field = db.add_field(instrument.id, "ANY_RANDOM_NAME", Frequency.DAILY)
        self.assertIsNotNone(field.id)
        db.close()

    def test_custom_storable_fields_on_init(self):
        """Test passing custom storable fields at initialization."""
        custom_fields = {"open", "high", "low", "close"}
        db = create_database(storable_fields=custom_fields)
        instrument = db.add_instrument("TEST", "Test", InstrumentType.STOCK)

        # Custom field should work
        field = db.add_field(instrument.id, "open", Frequency.DAILY)
        self.assertIsNotNone(field.id)

        # Default field should not work
        with self.assertRaises(ValueError):
            db.add_field(instrument.id, "price", Frequency.DAILY)
        db.close()

    def test_case_insensitive_matching(self):
        """Test that storable field matching is case-insensitive."""
        db = create_database()
        instrument = db.add_instrument("TEST", "Test", InstrumentType.STOCK)

        # All these should work for "price"
        db.add_field(instrument.id, "price", Frequency.DAILY)
        db.add_field(instrument.id, "PRICE", Frequency.WEEKLY)
        db.add_field(instrument.id, "Price", Frequency.MONTHLY)
        db.add_field(instrument.id, "PrIcE", Frequency.YEARLY)

        fields = db.list_fields(instrument_id=instrument.id)
        self.assertEqual(len(fields), 4)
        db.close()


if __name__ == "__main__":
    unittest.main(verbosity=2)
