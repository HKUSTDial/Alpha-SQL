import sqlite3
import tempfile
import unittest
from pathlib import Path

from alphasql.database.utils import load_value_examples


class LoadValueExamplesTest(unittest.TestCase):
    def test_default_max_example_length_filters_long_values(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db_dir = Path(tmpdir) / "sample_db"
            db_dir.mkdir()
            db_path = db_dir / "sample_db.sqlite"
            long_value = "x" * 101

            with sqlite3.connect(db_path) as conn:
                conn.execute("CREATE TABLE `items` (`name` TEXT)")
                conn.executemany(
                    "INSERT INTO `items` (`name`) VALUES (?)",
                    [("short",), (long_value,), ("",), (None,)],
                )

            examples = load_value_examples(
                "sample_db",
                tmpdir,
                "items",
                "name",
                max_num_examples=10,
            )

            self.assertEqual(examples, ["short"])

    def test_custom_max_example_length_allows_longer_values(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db_dir = Path(tmpdir) / "sample_db"
            db_dir.mkdir()
            db_path = db_dir / "sample_db.sqlite"
            long_value = "x" * 101

            with sqlite3.connect(db_path) as conn:
                conn.execute("CREATE TABLE `items` (`name` TEXT)")
                conn.executemany(
                    "INSERT INTO `items` (`name`) VALUES (?)",
                    [("short",), (long_value,)],
                )

            examples = load_value_examples(
                "sample_db",
                tmpdir,
                "items",
                "name",
                max_num_examples=10,
                max_example_length=101,
            )

            self.assertCountEqual(examples, ["short", long_value])


if __name__ == "__main__":
    unittest.main()
