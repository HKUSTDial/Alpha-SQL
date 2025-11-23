from alphasql.database.sql_execution import cached_execute_sql_with_timeout, is_valid_execution_result
from alphasql.algorithm.selection.utils import measure_sql_execution_time
import pickle
import glob
import json
import argparse
from tqdm import tqdm
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed

EXECUTION_TIME_REPEAT = 20

def select_final_sql_query(results_file_path: str, db_root_dir: str):
    question_id = int(results_file_path.split("/")[-1].split(".")[0])
    with open(results_file_path, "rb") as f:
        results = pickle.load(f)
    if results is None or len(results) == 0:
        return {
            "question_id": question_id,
            "db_id": "financial",
            "sql": "ERROR"
        }
    db_id = results[0][0].db_id
    db_path = f"{db_root_dir}/{db_id}/{db_id}.sqlite"
    result_groups = defaultdict(list)
    result_groups_with_invalid_result = defaultdict(list)
    
    for idx, result in enumerate(results):
        sql_query = result[-1].final_sql_query
        answer = cached_execute_sql_with_timeout(db_path, sql_query)
        if answer.result_type.value == "success":
            if is_valid_execution_result(answer):
                result_groups[frozenset(answer.result)].append(idx)
            result_groups_with_invalid_result[frozenset(answer.result)].append(idx)
    
    if len(result_groups) == 0:
        final_selected_sql_query = "ERROR"
        
        if len(result_groups_with_invalid_result) > 0:
            path_idx_with_sc_score = []
            for answer, path_indices in result_groups_with_invalid_result.items():
                sc_score = len(path_indices) / sum([len(v) for v in result_groups_with_invalid_result.values()])
                execution_time = measure_sql_execution_time(db_path, results[path_indices[0]][-1].final_sql_query, repeat=EXECUTION_TIME_REPEAT)
                # for path_idx in path_indices:
                #     consistency_score[path_idx] = (sc_score, execution_time)
                path_idx_with_sc_score.append((path_indices[0], sc_score, execution_time))
            path_idx_with_sc_score.sort(key=lambda x: (x[1], -x[2]), reverse=True)
            # Group paths by their scores
            # score_to_paths = {}
            # for path_idx, score in consistency_score.items():
            #     if score not in score_to_paths:
            #         score_to_paths[score] = []
            #     score_to_paths[score].append(path_idx)
            
            # Sort scores in descending order
            # sorted_scores = sorted(score_to_paths.keys(), reverse=True)
            path_idx = path_idx_with_sc_score[0][0]
            final_selected_sql_query = results[path_idx][-1].final_sql_query

        return {
            "question_id": question_id,
            "db_id": db_id,
            "sql": final_selected_sql_query
        }

    # consistency_score = {}
    # for answer, path_indices in result_groups.items():
    #     for path_idx in path_indices:
    #         consistency_score[path_idx] = len(path_indices) / sum([len(v) for v in result_groups.values()])
    # Group paths by their scores
    # score_to_paths = {}
    # for path_idx, score in consistency_score.items():
    #     if score not in score_to_paths:
    #         score_to_paths[score] = []
    #     score_to_paths[score].append(path_idx)
    
    # sorted_scores = sorted(score_to_paths.keys(), reverse=True)
    # path_idx = score_to_paths[sorted_scores[0]][0]
    # final_selected_sql_query = results[path_idx][-1].final_sql_query

    path_idx_with_sc_score = []
    for answer, path_indices in result_groups.items():
        sc_score = len(path_indices) / sum([len(v) for v in result_groups.values()])
        execution_time = measure_sql_execution_time(db_path, results[path_indices[0]][-1].final_sql_query, repeat=EXECUTION_TIME_REPEAT)
        path_idx_with_sc_score.append((path_indices[0], sc_score, execution_time))
    path_idx_with_sc_score.sort(key=lambda x: (x[1], -x[2]), reverse=True)
    path_idx = path_idx_with_sc_score[0][0]
    final_selected_sql_query = results[path_idx][-1].final_sql_query

    return {
        "question_id": question_id,
        "db_id": db_id,
        "sql": final_selected_sql_query
    }

def main(args):
    final_pred_sqls = {}
    with ProcessPoolExecutor(max_workers=args.process_num) as executor:
        result_paths = glob.glob(args.results_dir + "/*.pkl")
        future_to_path = {executor.submit(select_final_sql_query, path, args.db_root_dir): path for path in result_paths}
        
        for future in tqdm(as_completed(future_to_path), total=len(future_to_path), desc="Processing results"):
            selected_item = future.result()
            final_pred_sqls[str(selected_item["question_id"])] = selected_item["sql"] + '\t----- bird -----\t' + selected_item["db_id"]

    # Fill missing IDs with default value
    if final_pred_sqls:
        question_ids = [int(qid) for qid in final_pred_sqls.keys()]
        min_id = min(question_ids)
        max_id = max(question_ids)
        
        for qid in range(min_id, max_id + 1):
            if str(qid) not in final_pred_sqls:
                final_pred_sqls[str(qid)] = "ERROR\t----- bird -----\tfinancial"
    
    # Sort by question_id before output
    sorted_pred_sqls = dict(sorted(final_pred_sqls.items(), key=lambda x: int(x[0])))

    with open(args.output_path, "w") as f:
        json.dump(sorted_pred_sqls, f, indent=4)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", type=str, required=True)
    parser.add_argument("--db_root_dir", type=str, required=True)
    parser.add_argument("--process_num", type=int, default=32)
    parser.add_argument("--output_path", type=str, required=True)
    args = parser.parse_args()
    main(args)
