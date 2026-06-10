import json
import pickle
import re
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
from dotenv import load_dotenv
from loguru import logger
from tqdm import tqdm

from alphasql.database.database_manager import DatabaseManager
from alphasql.database.lsh_index import LSHIndex
from alphasql.database.sql_parse import extract_db_values_from_sql
from alphasql.database.utils import build_table_ddl_statement
from alphasql.llm_call.cost_recoder import CostRecorder
from alphasql.llm_call.embedding_utils import get_embedding_model
from alphasql.llm_call.openai_llm import call_openai
from alphasql.llm_call.prompt_factory import get_prompt
from alphasql.runner.task import Task

load_dotenv(override=True)

# Initialize embedding model from environment configuration
EMBEDDING_MODEL_CALLABLE = get_embedding_model()

COST_RECORDER = CostRecorder(model="gpt-3.5-turbo")
MODEL_NAME = "gpt-4o-mini"
TEMPERATURE = 0.2

class Preprocessor:
    """
    A class that preprocesses the dataset.
    """
    
    def __init__(self, 
                 data_file_path: str, 
                 database_root_dir: str, 
                 lsh_threshold: float, 
                 lsh_signature_size: int, 
                 lsh_n_gram: int, 
                 lsh_top_k: int,
                 edit_similarity_threshold: float,
                 embedding_similarity_threshold: float,
                 data_split: str = "dev",
                 save_root_dir: str = "data/preprocessed",
                 n_parallel_processes: int = 8,
                 max_dataset_samples: int = -1):
        """
        Initialize the Preprocessor.
        
        Args:
            data_file_path (str): The path to the data file.
            database_root_dir (str): The root directory of the databases.
            lsh_threshold (float): The threshold of the LSH index.
            lsh_signature_size (int): The signature size of the LSH index.
            lsh_n_gram (int): The n-gram of the LSH index.
            lsh_top_k (int): The top k of the LSH index.
            edit_similarity_threshold (float): The threshold of the edit similarity.
            embedding_similarity_threshold (float): The threshold of the embedding similarity.
            n_parallel_processes (int): The number of parallel processes.
            max_dataset_samples (int): The maximum number of samples in the dataset, -1 means no limit.
        """
        self.data_file_path = data_file_path
        self.database_root_dir = database_root_dir
        self.lsh_threshold = lsh_threshold
        self.lsh_signature_size = lsh_signature_size
        self.lsh_n_gram = lsh_n_gram
        self.lsh_top_k = lsh_top_k
        self.edit_similarity_threshold = edit_similarity_threshold
        self.embedding_similarity_threshold = embedding_similarity_threshold
        self.data_split = data_split
        self.save_root_dir = save_root_dir
        self.n_parallel_processes = n_parallel_processes
        self.data = json.load(open(self.data_file_path, "r", encoding="utf-8"))
        if max_dataset_samples != -1:
            self.data = self.data[:max_dataset_samples]
        self.all_db_ids = list(set([data_item["db_id"] for data_item in self.data]))
        self.tasks = [Task(
            question_id=data_item.get("question_id", question_id),
            db_id=data_item["db_id"],
            question=data_item["question"],
            evidence=data_item["evidence"],
            sql=data_item.get("SQL", None),
            difficulty=data_item.get("difficulty", None)
        ) for question_id, data_item in enumerate(self.data)]
        self.save_dir = Path(self.save_root_dir) / self.data_split
        self.save_dir.mkdir(parents=True, exist_ok=True)
    
    def preprocess_lsh_index_for_one_db(self,db_id: str) -> None:
        """
        Preprocess the LSH index for a given database.
        
        Args:
            db_id (str): The id of the database.
        """
        db_schema = DatabaseManager.get_database_schema(db_id, self.database_root_dir)
        lsh_index_dir_path = Path(db_schema.db_directory) / "lsh_index"
        if lsh_index_dir_path.exists():
            logger.info(f"LSH index for database {db_id} already exists, skipping...")
            return
        LSHIndex.create_lsh_index(db_schema, self.lsh_threshold, self.lsh_signature_size, self.lsh_n_gram)

    def preprocess_lsh_index(self) -> None:
        """
        Preprocess the LSH index for all databases.
        """
        with ThreadPoolExecutor(max_workers=self.n_parallel_processes) as executor:
            executor.map(self.preprocess_lsh_index_for_one_db, self.all_db_ids)
        logger.info(f"Preprocessed LSH index for {len(self.all_db_ids)} databases")
    
    def get_keywords_for_task(self, task: Task) -> List[str]:
        """
        Get the keywords for a task.
        
        Args:
            task (Task): The task.
        
        Returns:
            The keywords list for the task.
        """
        max_retries = 10
        retry_count = 0
        
        while retry_count < max_retries:
            try:
                raw_keywords_str = call_openai(
                    get_prompt("keywords_extraction", {"QUESTION": task.question, "HINT": task.evidence}),
                    MODEL_NAME,
                    TEMPERATURE,
                    cost_recorder=COST_RECORDER
                )[0]
                
                # Use regex to extract content between ```python ``` tags
                pattern = r"```python\s*\[(.*?)\]\s*```"
                match = re.search(pattern, raw_keywords_str, re.DOTALL)
                
                if match:
                    raw_keywords_str = f"[{match.group(1)}]"
                    raw_keywords = eval(raw_keywords_str)
                    keywords = []
                    for keyword in raw_keywords:
                        keyword: str
                        keywords.append(keyword.strip())
                        # for date type, e.g. 1999/06/22
                        keywords.append(keyword.replace("/", "-").strip("\"").strip("'"))
                        # split keyword
                        keywords.extend(keyword.replace("=", " ").replace("(", " ").replace(")", " ").replace("_", " ").split(" "))
                        keywords.extend(keyword.strip("'").replace("=", " ").replace("(", " ").replace(")", " ").replace("_", " ").split(" "))
                        keywords.extend(keyword.strip("\"").replace("=", " ").replace("(", " ").replace(")", " ").replace("_", " ").split(" "))
                    # remove duplicate keywords
                    keywords = list(set(keyword.strip() for keyword in keywords if keyword.strip() != ""))
                    return keywords
                else:
                    logger.warning(f"Failed to extract Python list from response, attempt {retry_count + 1}/{max_retries}")
                    retry_count += 1
                    continue
                    
            except Exception as e:
                logger.warning(f"Error processing keywords, attempt {retry_count + 1}/{max_retries}: {str(e)}")
                retry_count += 1
                continue
        
        # If all retries failed, return empty list
        raise Exception("Failed to extract keywords")
        
    def get_gold_relevant_values_for_task(self, task: Task) -> Dict[Tuple[str, str], List[str]]:
        """
        Get the gold relevant values for a task.
        
        Args:
            task (Task): The task.
        
        Returns:
            The gold relevant values for the task.
            A dictionary with tuple of (table name, column name) as key, and a list of relevant values as value.
        """
        return extract_db_values_from_sql(task.sql, database_schema=DatabaseManager.get_database_schema(task.db_id, self.database_root_dir))
    
    def filter_candidate_values_by_edit_similarity(self, 
                                                   candidate_values: List[Dict[str, Any]],
                                                   edit_similarity_threshold: float) -> List[Dict[str, Any]]:
        """
        Filter the candidate values by edit similarity.
        
        Args:
            candidate_values (List[Dict[str, Any]]): The candidate values.
            edit_similarity_threshold (float): The threshold of the edit similarity.
        
        Returns:
            The filtered candidate values.
        """
        filtered_candidate_values = []
        for candidate_value in candidate_values:
            table_name = candidate_value["table_name"]
            column_name = candidate_value["column_name"]
            query = candidate_value["query"]
            value = candidate_value["value"]
            edit_similarity = SequenceMatcher(None, value, query).ratio()
            if edit_similarity >= edit_similarity_threshold:
                filtered_candidate_values.append({
                    "query": query,
                    "table_name": table_name,
                    "column_name": column_name,
                    "value": value,
                    "edit_similarity": edit_similarity
                })
        return filtered_candidate_values

    def filter_candidate_values_by_embedding_similarity(self, 
                                                       candidate_values: List[Dict[str, Any]],
                                                       embedding_similarity_threshold: float) -> List[Dict[str, Any]]:
        """
        Filter the candidate values by embedding similarity.
        
        Args:
            candidate_values (List[Dict[str, Any]]): The candidate values.
            embedding_similarity_threshold (float): The threshold of the embedding similarity.
        
        Returns:
            The filtered candidate values.
        """
        candidate_values = [
            candidate_value
            for candidate_value in candidate_values
            if str(candidate_value["value"]).strip() != ""
            and str(candidate_value["query"]).strip() != ""
        ]
        if len(candidate_values) == 0:
            return []

        cosine_similarity = lambda x, y: np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y))
        to_embeded_list = [str(candidate_value["value"]) for candidate_value in candidate_values]
        to_embeded_list += [str(candidate_value["query"]) for candidate_value in candidate_values]
        to_embeded_list = list(set(to_embeded_list))
        embeddings = EMBEDDING_MODEL_CALLABLE.embed_documents(to_embeded_list)
        embeddings = {to_embeded_list[i]: embeddings[i] for i in range(len(to_embeded_list))}
        
        filtered_candidate_values = []
        for candidate_value in candidate_values:
            table_name = candidate_value["table_name"]
            column_name = candidate_value["column_name"]
            value = candidate_value["value"]
            query = candidate_value["query"]
            edit_similarity = candidate_value["edit_similarity"]
            value_embedding = embeddings[value]
            query_embedding = embeddings[query]
            embedding_similarity = cosine_similarity(value_embedding, query_embedding)
            if embedding_similarity >= embedding_similarity_threshold:
                filtered_candidate_values.append({
                    "query": query,
                    "table_name": table_name,
                    "column_name": column_name,
                    "value": value,
                    "edit_similarity": edit_similarity,
                    "embedding_similarity": embedding_similarity
                })
        return filtered_candidate_values
    
    def get_relevant_values_for_task(self, task: Task) -> Dict[Tuple[str, str], List[str]]:
        """
        Get the relevant values for a task.
        
        Args:
            task (Task): The task.
        
        Returns:
            The relevant values for the task.
            A dictionary with tuple of (table name, column name) as key, and a list of relevant values as value.
        """
        keywords = self.get_keywords_for_task(task)
        print(keywords)
        # Step 1: Use keywords to query the LSH index to get the candidate values.
        lsh_candidate_values = []
        for keyword in keywords:
            results = LSHIndex.query_lsh_index(
                DatabaseManager.get_database_schema(task.db_id, self.database_root_dir), 
                keyword, 
                top_k=self.lsh_top_k, 
                signature_size=self.lsh_signature_size, 
                n_gram=self.lsh_n_gram
            )
            lsh_candidate_values.extend(results)
        print(lsh_candidate_values)
        # Step 2: Use edit distance to filter the candidate values.
        edit_similarity_candidate_values = self.filter_candidate_values_by_edit_similarity(lsh_candidate_values, self.edit_similarity_threshold)
        # Step 3: Use embedding similarity to filter the candidate values.
        embedding_similarity_candidate_values = self.filter_candidate_values_by_embedding_similarity(edit_similarity_candidate_values, self.embedding_similarity_threshold)
        
        final_candidate_values = defaultdict(list)
        for value in embedding_similarity_candidate_values:
            final_candidate_values[(value["table_name"], value["column_name"])].append(value)
        
        # Step 4: Filter the candidate values with lower than COEFFICIENT * max_similarity_score
        # COEFFICIENT = 0.0 means no filtering
        COEFFICIENT = 0.0
        for table_name, column_name in final_candidate_values:
            values = final_candidate_values[(table_name, column_name)]
            max_edit_similarity = max([value["edit_similarity"] for value in values])
            values = [value for value in values if value["edit_similarity"] >= COEFFICIENT * max_edit_similarity]
            max_embedding_similarity = max([value["embedding_similarity"] for value in values])
            values = [value for value in values if value["embedding_similarity"] >= COEFFICIENT * max_embedding_similarity]
            value_embedding_similarity_map = defaultdict(float)
            for value in values:
                value_embedding_similarity_map[value["value"]] = max(value["embedding_similarity"], value_embedding_similarity_map[value["value"]])
            values = sorted(value_embedding_similarity_map.keys(), key=lambda x: value_embedding_similarity_map[x], reverse=True)
            final_candidate_values[(table_name, column_name)] = values
        
        return final_candidate_values
    
    def get_relevant_values_for_all_tasks(self) -> List[Dict[Tuple[str, str], List[str]]]:
        """
        Get the relevant values for all tasks.
        
        Returns:
            The relevant values for all tasks.
            A dictionary with tuple of (table name, column name) as key, and a list of relevant values as value.
        """
        if self.save_dir.joinpath("relevant_values_for_all_tasks.pkl").exists():
            with open(self.save_dir.joinpath("relevant_values_for_all_tasks.pkl"), "rb") as f:
                relevant_values_for_all_tasks = pickle.load(f)
        else:
            with ThreadPoolExecutor(max_workers=self.n_parallel_processes) as executor:
                relevant_values_for_all_tasks = list(
                tqdm(executor.map(self.get_relevant_values_for_task, self.tasks), 
                     total=len(self.tasks), 
                     desc="Getting relevant values for all tasks")
                )
            with open(self.save_dir.joinpath("relevant_values_for_all_tasks.pkl"), "wb") as f:
                pickle.dump(relevant_values_for_all_tasks, f)
        return relevant_values_for_all_tasks
    
    def get_gold_relevant_values_for_all_tasks(self) -> List[Dict[Tuple[str, str], List[str]]]:
        """
        Get the gold relevant values for all tasks.
        
        Returns:
            The gold relevant values for all tasks.
            A dictionary with tuple of (table name, column name) as key, and a list of relevant values as value.
        """
        if self.save_dir.joinpath("gold_relevant_values_for_all_tasks.pkl").exists():
            with open(self.save_dir.joinpath("gold_relevant_values_for_all_tasks.pkl"), "rb") as f:
                gold_relevant_values_for_all_tasks = pickle.load(f)
        else:
            with ThreadPoolExecutor(max_workers=self.n_parallel_processes) as executor:
                gold_relevant_values_for_all_tasks = list(
                    tqdm(executor.map(self.get_gold_relevant_values_for_task, self.tasks), 
                     total=len(self.tasks), 
                         desc="Getting gold relevant values for all tasks")
                )
            with open(self.save_dir.joinpath("gold_relevant_values_for_all_tasks.pkl"), "wb") as f:
                pickle.dump(gold_relevant_values_for_all_tasks, f)
        return gold_relevant_values_for_all_tasks
    
    def evaluate_relevant_values_performance_for_one_task(self, predicted_relevant_values: Dict[Tuple[str, str], List[str]], gold_relevant_values: Dict[Tuple[str, str], List[str]]) -> Dict[str, float]:
        """
        Evaluate the performance of the relevant values for one task.
        
        Args:
            predicted_relevant_values (Dict[Tuple[str, str], List[str]]): The predicted relevant values for one task.
            gold_relevant_values (Dict[Tuple[str, str], List[str]]): The gold relevant values for one task.
        
        Returns:
            The performance of the relevant values.
        """
        if len(predicted_relevant_values) == 0 and len(gold_relevant_values) == 0:
            return {
                "precision": 1.0,
                "recall": 1.0,
                "f1_score": 1.0
            }
        if len(predicted_relevant_values) == 0:
            return {
                "precision": 0.0,
                "recall": 0.0,
                "f1_score": 0.0
            }
        if len(gold_relevant_values) == 0:
            return {
                "precision": 0.0,
                "recall": 1.0,
                "f1_score": 0.0
            }
        predicted_relevant_values_set, gold_relevant_values_set = set(), set()  
        for table_name, column_name in predicted_relevant_values.keys():
            for value in predicted_relevant_values[(table_name, column_name)]:
                predicted_relevant_values_set.add((table_name, column_name, value))
        for table_name, column_name in gold_relevant_values.keys():
            for value in gold_relevant_values[(table_name, column_name)]:
                gold_relevant_values_set.add((table_name, column_name, value))
        precision = len(predicted_relevant_values_set & gold_relevant_values_set) / len(predicted_relevant_values_set)
        recall = len(predicted_relevant_values_set & gold_relevant_values_set) / len(gold_relevant_values_set)
        if precision + recall == 0:
            f1_score = 0.0
        else:
            f1_score = 2 * precision * recall / (precision + recall)
        return {
            "precision": precision,
            "recall": recall,
            "f1_score": f1_score
        }
    
    def evaluate_relevant_values_retrieval_performance_for_all_tasks(self, 
                                                                     predicted_relevant_values_for_all_tasks: List[Dict[Tuple[str, str], List[str]]], 
                                                                     gold_relevant_values_for_all_tasks: List[Dict[Tuple[str, str], List[str]]]) -> Dict[str, float]:
        """
        Evaluate the performance of the relevant values retrieval for all tasks.
        
        Args:
            predicted_relevant_values_for_all_tasks (List[Dict[Tuple[str, str], List[str]]]): The predicted relevant values for all tasks.
            gold_relevant_values_for_all_tasks (List[Dict[Tuple[str, str], List[str]]]): The gold relevant values for all tasks.
        
        Returns:
            The performance of the relevant values.
        """
        with ThreadPoolExecutor(max_workers=self.n_parallel_processes) as executor:
            performance_for_all_tasks = list(
                tqdm(executor.map(self.evaluate_relevant_values_performance_for_one_task, predicted_relevant_values_for_all_tasks, gold_relevant_values_for_all_tasks), 
                     total=len(predicted_relevant_values_for_all_tasks), 
                     desc="Evaluating relevant values performance for all tasks")
            )
        save_path = self.save_dir.joinpath("relevant_values_retrieval_performance.json")
        save_data = []
        for task, predicted_relevant_values, gold_relevant_values, performance in zip(self.tasks, predicted_relevant_values_for_all_tasks, gold_relevant_values_for_all_tasks, performance_for_all_tasks):
            save_data.append({
                "question_id": task.question_id,
                "db_id": task.db_id,
                "question": task.question,
                "evidence": task.evidence,
                "sql": task.sql,
                "difficulty": task.difficulty,
                "predicted_relevant_values": [{"table_name": table_name, "column_name": column_name, "values": list(values)} for (table_name, column_name), values in predicted_relevant_values.items()],
                "gold_relevant_values": [{"table_name": table_name, "column_name": column_name, "values": list(values)} for (table_name, column_name), values in gold_relevant_values.items()],
                "performance": performance
            })
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(save_data, f, indent=4)
        
        average_precision = sum([performance["precision"] for performance in performance_for_all_tasks]) / len(performance_for_all_tasks)
        average_recall = sum([performance["recall"] for performance in performance_for_all_tasks]) / len(performance_for_all_tasks)
        average_f1_score = sum([performance["f1_score"] for performance in performance_for_all_tasks]) / len(performance_for_all_tasks)
        performance = {
            "precision": average_precision,
            "recall": average_recall,
            "f1_score": average_f1_score
        }
        logger.info(f"Relevant values retrieval performance: {performance}")
        return performance
    
    def preprocess_schema_context_for_all_tasks(self) -> List[Task]:
        """
        Preprocess the schema context for all tasks.
        
        Returns:
            The preprocessed schema context.
        """
        save_path = self.save_dir.joinpath("tasks.pkl")
        relevant_values_for_all_tasks = pickle.load(open(self.save_dir.joinpath("relevant_values_for_all_tasks.pkl"), "rb"))
        tasks_with_schema_context = []
        for task, relevant_values_one_task in tqdm(zip(self.tasks, relevant_values_for_all_tasks), desc="Preprocessing schema context for all tasks"):
            task = deepcopy(task)
            relevant_values_one_task: Dict[Tuple[str, str], List[str]]
            database_schema = DatabaseManager.get_database_schema(task.db_id, self.database_root_dir)
            table_schema_dict = deepcopy(database_schema.tables)
            # clear the original value examples
            # for table_name in table_schema_dict:
            #     for column_name in table_schema_dict[table_name].columns:
            #         table_schema_dict[table_name].columns[column_name].value_examples = []
            # add the new value examples which are the most similar to the query
            for table_name, column_name in relevant_values_one_task.keys():
                table_schema_dict[table_name].columns[column_name].value_examples = relevant_values_one_task[(table_name, column_name)][:3]
            schema_context = "\n".join([build_table_ddl_statement(
                table_schema_dict[table_name].to_dict(), 
                add_value_description=True,
                add_column_description=True,
                add_value_examples=True,
                add_expanded_column_name=True
            ) for table_name in table_schema_dict])
            task.schema_context = schema_context
            task.table_schema_dict = table_schema_dict
            tasks_with_schema_context.append(task)
        with open(save_path, "wb") as f:
            pickle.dump(tasks_with_schema_context, f)
        return tasks_with_schema_context

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_file_path", type=str, required=True)
    parser.add_argument("--database_root_dir", type=str, required=True)
    parser.add_argument("--save_root_dir", type=str, required=True)
    parser.add_argument("--lsh_threshold", type=float, required=True, default=0.5)
    parser.add_argument("--lsh_signature_size", type=int, required=True, default=128)
    parser.add_argument("--lsh_n_gram", type=int, required=True, default=3)
    parser.add_argument("--lsh_top_k", type=int, required=True, default=20)
    parser.add_argument("--edit_similarity_threshold", type=float, required=True, default=0.3)
    parser.add_argument("--embedding_similarity_threshold", type=float, required=True, default=0.6)
    parser.add_argument("--n_parallel_processes", type=int, required=True, default=8)
    parser.add_argument("--max_dataset_samples", type=int, required=True, default=-1)
    args = parser.parse_args()
    
    preprocessor = Preprocessor(
        data_file_path=args.data_file_path,
        database_root_dir=args.database_root_dir,
        lsh_threshold=args.lsh_threshold,
        lsh_signature_size=args.lsh_signature_size,
        lsh_n_gram=args.lsh_n_gram,
        lsh_top_k=args.lsh_top_k,
        edit_similarity_threshold=args.edit_similarity_threshold,
        embedding_similarity_threshold=args.embedding_similarity_threshold,
        n_parallel_processes=args.n_parallel_processes,
        max_dataset_samples=args.max_dataset_samples,
        save_root_dir=args.save_root_dir
    )
    preprocessor.preprocess_lsh_index()
    # gold_relevant_values_for_all_tasks = preprocessor.get_gold_relevant_values_for_all_tasks()
    predicted_relevant_values_for_all_tasks = preprocessor.get_relevant_values_for_all_tasks()
    # relevant_values_retrieval_performance = preprocessor.evaluate_relevant_values_retrieval_performance_for_all_tasks(predicted_relevant_values_for_all_tasks, gold_relevant_values_for_all_tasks)
    tasks_with_schema_context = preprocessor.preprocess_schema_context_for_all_tasks()
