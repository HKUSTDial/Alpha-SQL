import pickle
from datasketch import MinHash, MinHashLSH
from pathlib import Path
from typing import Dict, List
from collections import defaultdict
from nltk.util import ngrams
from tqdm import tqdm
from typing import Tuple, Any
import shutil

from alphasql.database.schema import DatabaseSchema
from alphasql.database.sql_execution import execute_sql_without_timeout


class LSHIndex:
    """
    A class for creating and querying a LSH index for a database schema.
    
    Attributes:
        QUERY_DISTINCT_VALUES_SQL (str): The SQL query to get the unique values for a column.
        CACHED_LSH_INDEX (Dict[str, Tuple[MinHashLSH, Dict[str, Tuple[MinHash, str, str, int, str]]]]): A dictionary mapping database ids to LSH indexes.
    """
    QUERY_DISTINCT_VALUES_SQL = "SELECT DISTINCT `{column_name}` FROM `{table_name}` WHERE `{column_name}` IS NOT NULL AND trim(cast(`{column_name}` as text)) != ''"
    
    CACHED_LSH_INDEX: Dict[str, Tuple[MinHashLSH, Dict[str, Tuple[MinHash, str, str, int, str]]]] = {}
    
    @classmethod
    def get_unique_database_values(cls, database_schema: DatabaseSchema, ignore_primary_keys: bool = True, ignore_non_text_columns: bool = True) -> Dict[str, Dict[str, List[str]]]:
        """
        Get the unique values for each column in the database schema.
        
        Args:
            database_schema (DatabaseSchema): The database schema to get the unique values for.
            ignore_primary_keys (bool): Whether to ignore primary keys.
            ignore_non_text_columns (bool): Whether to ignore non-text columns.
        Returns:
            Dict[str, Dict[str, List[str]]]: A dictionary containing the unique values for each column.
        """
        unique_values: Dict[str, Dict[str, List[str]]] = defaultdict(lambda: defaultdict(list))
        db_path = Path(database_schema.db_directory) / f"{database_schema.db_id}.sqlite"
        
        for table_name, table_schema in database_schema.tables.items():
            for column_name, column_schema in table_schema.columns.items():
                if ignore_primary_keys and column_schema.primary_key:
                    continue
                if ignore_non_text_columns and column_schema.column_type.lower() != "text":
                    continue
                query = cls.QUERY_DISTINCT_VALUES_SQL.format(column_name=column_name, table_name=table_name)
                execution_result = execute_sql_without_timeout(db_path, query)
                unique_values[table_name][column_name] = [
                    str(row[0])
                    for row in execution_result.result
                    if str(row[0]).strip() != ""
                ]
                
        return unique_values
        
    @classmethod
    def create_minhash(cls, string: str, signature_size: int = 128, n_gram: int = 3) -> MinHash:
        """
        Create a MinHash for a string.
        
        Args:
            string (str): The string to create a MinHash for.
            signature_size (int): The size of the signature, defaults to 128.
            n_gram (int): The size of the n-gram, defaults to 5.
        Returns:
            MinHash: A MinHash for the string.
        """
        minhash = MinHash(num_perm=signature_size)
        for d in ngrams(string, n_gram):
            minhash.update("".join(d).encode('utf8'))
        return minhash
    
    @classmethod
    def create_lsh_index(cls, database_schema: DatabaseSchema, threshold: float = 0.5, signature_size: int = 128, n_gram: int = 3) -> None:
        """
        Create a LSH index for the database schema.
        
        Args:
            database_schema (DatabaseSchema): The database schema to create the LSH index for.
            threshold (float): The threshold for the LSH index, defaults to 0.5.
            signature_size (int): The size of the signature, defaults to 128.
            n_gram (int): The size of the n-gram, defaults to 3.
        """
        unique_values = cls.get_unique_database_values(database_schema)
        print(unique_values)
        lsh_index = MinHashLSH(threshold=threshold, num_perm=signature_size)
        minhashes = {}
        total_unique_values_count = sum(len(column_values) for table_values in unique_values.values() for column_values in table_values.values())
        pbar = tqdm(total=total_unique_values_count, desc=f"Creating LSH index for database: {database_schema.db_id}")
        for table_name, table_values in unique_values.items():
            for column_name, column_values in table_values.items():
                for value_idx, value in enumerate(column_values):
                    minhash = cls.create_minhash(value, signature_size, n_gram)
                    minhash_key = f"{table_name}_{column_name}_{value_idx}"
                    minhashes[minhash_key] = (minhash, table_name, column_name, value)
                    lsh_index.insert(minhash_key, minhash)
                    pbar.update(1)
        pbar.close()
        
        lsh_index_dir_path = Path(database_schema.db_directory) / "lsh_index"
        if lsh_index_dir_path.exists():
            shutil.rmtree(lsh_index_dir_path)
        lsh_index_dir_path.mkdir(parents=True)
        
        lsh_index_path = lsh_index_dir_path / f"lsh_index.pkl"
        minhashes_path = lsh_index_dir_path / f"minhashes.pkl"
        with open(lsh_index_path, "wb") as f:
            pickle.dump(lsh_index, f)
        with open(minhashes_path, "wb") as f:
            pickle.dump(minhashes, f)

    @classmethod
    def query_lsh_index(cls, database_schema: DatabaseSchema, query: str, top_k: int = 10, signature_size: int = 128, n_gram: int = 3) -> List[Tuple[float, Dict[str, Any]]]:
        """
        Query the LSH index for the database schema.
        
        Args:
            database_schema (DatabaseSchema): The database schema to query.
            query (str): The query to search for.
            top_k (int): The number of results to return, defaults to 10.
            signature_size (int): The size of the signature, defaults to 128.
            n_gram (int): The size of the n-gram, defaults to 3.
        Returns:
            A list of tuples containing the score and the metadata.
        """
        lsh_index_dir_path = Path(database_schema.db_directory) / "lsh_index"
        lsh_index_path = lsh_index_dir_path / f"lsh_index.pkl"
        minhashes_path = lsh_index_dir_path / f"minhashes.pkl"
        if database_schema.db_id not in cls.CACHED_LSH_INDEX:
            with open(lsh_index_path, "rb") as f:
                lsh_index = pickle.load(f)
            with open(minhashes_path, "rb") as f:
                minhashes = pickle.load(f)
            cls.CACHED_LSH_INDEX[database_schema.db_id] = (lsh_index, minhashes)
        lsh_index, minhashes = cls.CACHED_LSH_INDEX[database_schema.db_id]
        
        query_minhash = cls.create_minhash(query, signature_size, n_gram)
        results = lsh_index.query(query_minhash)
        similar_items = [(result_key, minhashes[result_key][0].jaccard(query_minhash)) for result_key in results]
        similar_items = sorted(similar_items, key=lambda x: x[1], reverse=True)[:top_k]
        return [
            {
                "query": query,
                "lsh_score": score,
                "table_name": minhashes[result_key][1],
                "column_name": minhashes[result_key][2],
                "value": minhashes[result_key][3]
            }
            for result_key, score in similar_items
        ]
