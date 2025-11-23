from typing import Dict, Any, List, Tuple
from pathlib import Path
import chardet
import pandas as pd
from loguru import logger

from alphasql.database.sql_execution import execute_sql_without_timeout

MAX_LENGTH_COLUMN_DESCRIPTION = 100

def lower_str_list(str_list: List[Any]) -> List[Any]:
    """
    Convert a list of strings or nested lists to a list of lowercase strings or nested lists.
    
    Args:
        str_list: List of strings or nested lists.
    Returns:
        List of lowercase strings or nested lists.
    """
    return [s.lower() if isinstance(s, str) else lower_str_list(s) for s in str_list]

def load_table_names(db_id: str, database_root_dir: str) -> List[str]:
    """
    Load table names from database.
    
    Args:
        db_id: Database ID.
        database_root_dir: Root directory of databases.
    Returns:
        List of table names.
    """
    db_path = Path(database_root_dir) / db_id / f"{db_id}.sqlite"
    table_names = execute_sql_without_timeout(db_path, "SELECT name FROM sqlite_master WHERE type='table' AND name != 'sqlite_sequence';").result
    return [row[0].strip() for row in table_names]

def load_column_names_and_types(db_id: str, database_root_dir: str, table_name: str) -> List[Tuple[str, str]]:
    """
    Load column names and types from a table.
    
    Args:
        db_id: Database ID.
        database_root_dir: Root directory of databases.
        table_name: Table name.
    Returns:
        List of tuples, each containing a column name and its type.
    """
    db_path = Path(database_root_dir) / db_id / f"{db_id}.sqlite"
    table_info = execute_sql_without_timeout(db_path, f"PRAGMA table_info(`{table_name}`);").result
    return [(row[1].strip(), row[2].strip()) for row in table_info]

def load_primary_keys(db_id: str, database_root_dir: str, table_name: str) -> List[str]:
    """
    Load primary keys from a table.
    
    Args:
        db_id: Database ID.
        database_root_dir: Root directory of databases.
        table_name: Table name.
    Returns:
        List of primary key column names.
    """
    db_path = Path(database_root_dir) / db_id / f"{db_id}.sqlite"
    table_info = execute_sql_without_timeout(db_path, f"PRAGMA table_info(`{table_name}`);").result
    return [row[1].strip() for row in table_info if row[5] != 0]

def load_foreign_keys(db_id: str, database_root_dir: str, table_name: str) -> List[Tuple[str, str, str, str]]:
    """
    Load foreign keys from a table.
    
    Args:
        db_id: Database ID.
        database_root_dir: Root directory of databases.
        table_name: Table name.
    Returns:
        List of tuples, each containing a source table name, source column name, target table name, and target column name.
    """
    db_path = Path(database_root_dir) / db_id / f"{db_id}.sqlite"
    foreign_keys_list = execute_sql_without_timeout(db_path, f"PRAGMA foreign_key_list(`{table_name}`);").result
    foreign_keys = []
    for foreign_key in foreign_keys_list:
        source_table_name = table_name.strip()
        source_column_name = foreign_key[3].strip()
        target_table_name = foreign_key[2].strip()
        target_column_name = None
        if foreign_key[4] is not None:
            target_column_name = foreign_key[4].strip()
        else:
            # Try to fix target column is None by searching primary keys of target table
            target_table_primary_keys = load_primary_keys(db_id, database_root_dir, target_table_name)
            if len(target_table_primary_keys) >= 1:
                target_column_name = target_table_primary_keys[0].strip()
                logger.warning(f"Target column is None and has been fixed by primary keys of target table {target_table_name} - {target_column_name}, source table: {source_table_name}, source column: {source_column_name}")
            else:
                raise ValueError(f"Target column is None and cannot be fixed by primary keys of target table {target_table_name}, source table: {source_table_name}, source column: {source_column_name}")
        foreign_keys.append((source_table_name, source_column_name, target_table_name, target_column_name))
    return foreign_keys

def _normalize_description_string(description: str) -> str:
    """
    Normalize the description string.
    """
    description = description.replace("\r", "").replace("\n", "").replace("commonsense evidence:", "").strip()
    while "  " in description:
        description = description.replace("  ", " ")
    return description

def load_database_description(db_id: str, database_root_dir: str) -> Dict[str, Dict[str, Dict[str, Any]]]:
    """
    Load database description from database.
    
    Args:
        db_id: Database ID.
        database_root_dir: Root directory of databases.
    Returns:
        A dictionary with lowercased table names as keys and table descriptions as values.
        The table description is a dictionary with lowercased original column names as keys and column descriptions as values.
    """
    db_description_dir = Path(database_root_dir) / db_id / "database_description"
    if not db_description_dir.exists():
        logger.warning(f"Database description for database {db_id} does not exist, skipping...")
        return {}
    database_description = {}
    for csv_file in db_description_dir.glob("*.csv"):
        table_name_lower = csv_file.stem.lower().strip()
        encoding_type = chardet.detect(csv_file.read_bytes())["encoding"]
        table_description = {}
        table_description_df = pd.read_csv(csv_file, encoding=encoding_type, index_col=False)
        for _, row in table_description_df.iterrows():
            if pd.isna(row["original_column_name"]):
                continue
            original_column_name_lower = row["original_column_name"].strip().lower()
            expanded_column_name = row["column_name"].strip() if pd.notna(row["column_name"]) else ""
            column_description = _normalize_description_string(row["column_description"]) if pd.notna(row["column_description"]) else ""
            data_format = row["data_format"].strip() if pd.notna(row["data_format"]) else ""
            value_description = _normalize_description_string(row["value_description"]) if pd.notna(row["value_description"]) else ""
            if value_description.lower().startswith("not useful"):
                value_description = value_description[len("not useful"):].strip()
            table_description[original_column_name_lower] = {
                "original_column_name_lower": original_column_name_lower,
                "expanded_column_name": expanded_column_name,
                "column_description": column_description,
                "data_format": data_format,
                "value_description": value_description
            }
        database_description[table_name_lower] = table_description
    return database_description

def load_value_examples(db_id: str, database_root_dir: str, table_name: str, column_name: str, max_num_examples: int = 3) -> List[str]:
    """
    Load value examples from database.
    
    Args:
        db_id: Database ID.
        database_root_dir: Root directory of databases.
        table_name: Table name.
        column_name: Column name.
        max_num_examples: Maximum number of value examples to load.
    Returns:
        List of value examples.
    """
    db_path = Path(database_root_dir) / db_id / f"{db_id}.sqlite"
    examples = execute_sql_without_timeout(db_path, f"SELECT DISTINCT `{column_name}` FROM `{table_name}` WHERE `{column_name}` IS NOT NULL AND `{column_name}` != '' LIMIT {max_num_examples};").result
    res = [example[0] for example in examples]
    # print(res, flush=True)
    if str(res) > MAX_LENGTH_COLUMN_DESCRIPTION:
        return [res[0]]
    else:
        return res

def load_database_schema_dict(db_id: str, database_root_dir: str) -> Dict[str, Dict[str, Dict[str, Any]]]:
    """
    Load the database schema as a dictionary.

    Args:
        db_id: Database ID.
        database_root_dir: Root directory of databases.
    Returns:
        A dictionary representing the database schema.
    """
    database_schema_dict = {"db_id": db_id, "db_directory": Path(database_root_dir) / db_id, "tables": {}}
    table_names = load_table_names(db_id, database_root_dir)
    for table_name in table_names:
        table_schema_dict = {}
        table_schema_dict["table_name"] = table_name
        table_schema_dict["columns"] = {}
        
        # Load primary keys
        primary_keys = load_primary_keys(db_id, database_root_dir, table_name)
        
        # Load foreign keys
        foreign_keys = load_foreign_keys(db_id, database_root_dir, table_name)
        
        # Load database description
        database_description = load_database_description(db_id, database_root_dir)
        
        # Load column names and types, and add other useful information
        column_names_and_types = load_column_names_and_types(db_id, database_root_dir, table_name)
        for column_name, column_type in column_names_and_types:
            column_schema_dict = {}
            column_schema_dict["original_column_name"] = column_name
            column_schema_dict["column_type"] = column_type
            column_schema_dict["foreign_keys"] = []
            column_schema_dict["referenced_by"] = []
            
            # Set primary key
            if column_name.lower() in lower_str_list(primary_keys):
                column_schema_dict["primary_key"] = True
            else:
                column_schema_dict["primary_key"] = False
            
            # Set foreign keys and referenced by
            for source_table_name, source_column_name, target_table_name, target_column_name in foreign_keys:
                if source_table_name.lower() == table_name.lower() and source_column_name.lower() == column_name.lower():
                    column_schema_dict["foreign_keys"].append((target_table_name, target_column_name))
                if target_table_name.lower() == table_name.lower() and target_column_name.lower() == column_name.lower():
                    column_schema_dict["referenced_by"].append((source_table_name, source_column_name))
                    
            # Set description-related information
            if table_name.lower() in database_description and column_name.lower() in database_description[table_name.lower()]:
                column_schema_dict["expanded_column_name"] = database_description[table_name.lower()][column_name.lower()]["expanded_column_name"]
                column_schema_dict["column_description"] = database_description[table_name.lower()][column_name.lower()]["column_description"]
                column_schema_dict["value_description"] = database_description[table_name.lower()][column_name.lower()]["value_description"]
            
            # Set value examples
            if column_type.upper() != "BLOB":
                column_schema_dict["value_examples"] = load_value_examples(db_id, database_root_dir, table_name, column_name)
            else:
                column_schema_dict["value_examples"] = []
            
            table_schema_dict["columns"][column_name] = column_schema_dict
        database_schema_dict["tables"][table_name] = table_schema_dict
    return database_schema_dict

def build_table_ddl_statement(table_schema_dict: Dict[str, Dict[str, Any]],
                              add_expanded_column_name: bool = True,
                              add_column_description: bool = True,
                              add_value_description: bool = True,
                              add_value_examples: bool = True) -> str:
    """
    Build the DDL statement for a table.
    
    Args:
        table_schema_dict: The table schema dictionary.
        add_expanded_column_name: Whether to add the expanded column name.
        add_column_description: Whether to add the column description.
        add_value_description: Whether to add the value description.
        add_value_examples: Whether to add the value examples.
    Returns:
        The DDL statement for the table.
    """
    statement = f"CREATE TABLE `{table_schema_dict['table_name']}` (\n"
    foreign_keys = []
    primary_keys = []
    for column_name, column_schema in table_schema_dict["columns"].items():
        column_statement = f"\t`{column_name}` {column_schema['column_type']},"
        comment_parts = []
        if add_expanded_column_name and column_schema["expanded_column_name"].strip() != "" and column_schema["expanded_column_name"].strip().lower() != column_name.lower():
            comment_parts.append(f"Column Meaning: {column_schema['expanded_column_name']}")
        if add_column_description and column_schema["column_description"].strip() != "" and column_schema["column_description"].strip().lower() != column_name.lower() and column_schema["column_description"].strip().lower() != column_schema["expanded_column_name"].strip().lower():
            comment_parts.append(f"Column Description: {column_schema['column_description']}")
        if add_value_description and column_schema["value_description"].strip() != "":
            comment_parts.append(f"Value Description: {column_schema['value_description']}")
        if add_value_examples and len(column_schema["value_examples"]) > 0 and column_schema["column_type"].upper() == "TEXT":
            comment_parts.append(f"Value Examples: {', '.join([f'`{value}`' for value in column_schema['value_examples']])}")
        if len(comment_parts) > 0:
            column_statement += f" -- {' | '.join(comment_parts)}"
        statement += column_statement + "\n"
            
        if column_schema["primary_key"]:
            primary_keys.append(column_name)
        for foreign_key in column_schema["foreign_keys"]:
            foreign_keys.append((column_name, *foreign_key))
    statement += "\tPRIMARY KEY (" + ", ".join([f"`{primary_key}`" for primary_key in primary_keys]) + "),\n"
    for source_column_name, target_table_name, target_column_name in foreign_keys:
        statement += f"\tFOREIGN KEY (`{source_column_name}`) REFERENCES `{target_table_name}`(`{target_column_name}`),\n"
    if statement[-2:] == ",\n":
        statement = statement[:-2]
    statement += "\n);"
    return statement
