import sqlite3
import tempfile
import unittest
from pathlib import Path

from alphasql.database.lsh_index import LSHIndex
from alphasql.database.schema import ColumnSchema, DatabaseSchema, TableSchema


class LSHIndexTest(unittest.TestCase):
    def test_unique_database_values_filter_empty_text(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db_dir = Path(tmpdir) / "sample_db"
            db_dir.mkdir()
            db_path = db_dir / "sample_db.sqlite"

            with sqlite3.connect(db_path) as conn:
                conn.execute("CREATE TABLE `items` (`name` TEXT)")
                conn.executemany(
                    "INSERT INTO `items` (`name`) VALUES (?)",
                    [("short",), ("",), ("   ",), (None,)],
                )

            database_schema = DatabaseSchema(
                db_id="sample_db",
                db_directory=str(db_dir),
                tables={
                    "items": TableSchema(
                        table_name="items",
                        columns={
                            "name": ColumnSchema(
                                original_column_name="name",
                                column_type="TEXT",
                            )
                        },
                    )
                },
            )

            unique_values = LSHIndex.get_unique_database_values(database_schema)

            self.assertEqual(unique_values["items"]["name"], ["short"])


if __name__ == "__main__":
    unittest.main()
