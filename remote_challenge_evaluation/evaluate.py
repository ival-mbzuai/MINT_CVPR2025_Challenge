import random
import json
# import openai
import ast
import time
from collections import defaultdict

from multiprocessing.pool import Pool
# from call_llm_judge import annotate_combined
from llm_judge_parallel import annotate_combined_parallel



def calculate_metrics(result_qa_pairs, data_dict_temp):
    """
    Calculate overall and subset-specific metrics from annotation results.
    
    Args:
        result_qa_pairs (list): List of [evaluation_result_dict, original_prediction_dict] pairs
                                where evaluation_result_dict has 'pred' and 'score' keys
                                and original_prediction_dict has 'subset' key
    
    Returns:
        dict: Dictionary containing overall and subset-specific metrics:
            {
                'overall': {
                    'accuracy': float,
                    'score': float,
                    'count': int
                },
                'subsets': {
                    'subset1': {
                        'accuracy': float,
                        'score': float,
                        'count': int
                    },
                    ...
                }
            }
    """
    # Initialize metrics containers for oe and mcq
    metrics = {}
    metrics["oe"] = {
        'overall': {
            'correct_count': 0,
            'total_score': 0,
            'count': 0
        },
        'languages': defaultdict(lambda: {
            'correct_count': 0,
            'total_score': 0,
            'count': 0
        })
    }

    metrics["mcq"] = {
        'overall': {
            'correct_count': 0,
            'count': 0
        },
        'languages': defaultdict(lambda: {
            'correct_count': 0,
            'count': 0
        })
    }
    
    # Process each result pair for oe
    for result_dict, original_dict in result_qa_pairs["oe"]:
        # Skip any malformed results
        if not isinstance(result_dict, dict):
            continue
            
        # Extract prediction and score, handling potential format variations
        pred = result_dict.get('pred', result_dict.get('prediction', ''))
        score = result_dict.get('score', 0)
        language = original_dict.get('language', 'unknown')
        
        # Normalize prediction to binary correct/incorrect
        is_correct = 1 if pred.lower() == 'correct' else 0
        
        # Update overall metrics
        metrics["oe"]['overall']['correct_count'] += is_correct
        metrics["oe"]['overall']['total_score'] += score
        metrics["oe"]['overall']['count'] += 1
        
        # Create subset dict if it doesn't exist
        if language not in metrics["oe"]['languages']:
            metrics["oe"]['languages'][language] = {
                'correct_count': 0,
                'total_score': 0,
                'count': 0
            }
            
        # Update subset metrics
        metrics["oe"]['languages'][language]['correct_count'] += is_correct
        metrics["oe"]['languages'][language]['total_score'] += score
        metrics["oe"]['languages'][language]['count'] += 1

    # Process each result pair for mcq
    for k, single_dict in data_dict_temp["mcq"].items():
        # Skip any malformed results
        # if not isinstance(result_dict, dict):
        #     continue
            
        # Extract prediction and score, handling potential format variations
        pred = single_dict.get('pred', '')
        gt_answer = single_dict.get('a', 'unknown')
        language = single_dict.get('language', 'unknown')
        
        # Normalize prediction to binary correct/incorrect
        is_correct = 1 if pred.lower() == gt_answer.lower() else 0
        
        # Update overall metrics
        metrics["mcq"]['overall']['correct_count'] += is_correct
        metrics["mcq"]['overall']['count'] += 1
        
        # Create subset dict if it doesn't exist
        if language not in metrics["mcq"]['languages']:
            metrics["mcq"]['languages'][language] = {
                'correct_count': 0,
                'count': 0
            }
            
        # Update subset metrics
        metrics["mcq"]['languages'][language]['correct_count'] += is_correct
        metrics["mcq"]['languages'][language]['count'] += 1
    
    # # Calculate final metrics
    # result_metrics = {
    #     'overall': {
    #         'accuracy': metrics["oe"]['overall']['correct_count'] / max(metrics["oe"]['overall']['count'], 1),
    #         'score': metrics["oe"]['overall']['total_score'] / max(metrics["oe"]['overall']['count'], 1),
    #         'count': metrics["oe"]['overall']['count']
    #     },
    #     'languages': {}
    # }
    
    # # Calculate metrics for each subset
    # for language, language_metrics in metrics["oe"]['languages'].items():
    #     result_metrics['languages'][language] = {
    #         'accuracy': language_metrics['correct_count'] / max(language_metrics['count'], 1),
    #         'score': language_metrics['total_score'] / max(language_metrics['count'], 1),
    #         'count': language_metrics['count']
    #     }
    
    return metrics



def merge_dicts(gt_file, pred_file):
    # json_gt = json.load(open(gt_file))
    # json_pred = json.load(open(pred_file))
    # with open(gt_file, "r") as f:
    #     json_gt = json.load(f)

    # with open(pred_file, "r") as f:
    #     json_pred = json.load(f)


    with open(gt_file, "r", encoding="utf-8") as file:
        json_gt = json.load(file)

    with open(pred_file, "r", encoding="utf-8") as file:
        json_pred = json.load(file)

    json_gt_oe = json_gt["oe"]
    json_pred_oe = json_pred["oe"]

    json_gt_mcq = json_gt["mcq"]
    json_pred_mcq = json_pred["mcq"]

    data_dict = {"oe":{}, "mcq":{}}

    ## OpenEnded QA
    for item in json_gt_oe:
        unique_key = item['unique_id']
        if unique_key not in data_dict["oe"]:
            data_dict["oe"][unique_key] = {'a': item['answer'], 'q': item['question'], 'language': item['language']}

    for item in json_pred_oe:
        unique_key = item['unique_id']
        if unique_key in data_dict["oe"]:
            data_dict["oe"][unique_key]['pred'] = item['A']

    ## MCQ QA
    for item in json_gt_mcq:
        unique_key = item['unique_id']
        if unique_key not in data_dict["mcq"]:
            data_dict["mcq"][unique_key] = {'a': item['answer_choice'], 'q': item['question'], 'language': item['language']}

    for item in json_pred_mcq:
        unique_key = item['unique_id']
        if unique_key in data_dict["mcq"]:
            data_dict["mcq"][unique_key]['pred'] = item['A']

    return data_dict

def evaluate(test_annotation_file, user_submission_file, phase_codename, **kwargs):
    print("Starting Evaluation.....")
    """
    Evaluates the submission for a particular challenge phase and returns score
    Arguments:

        `test_annotations_file`: Path to test_annotation_file on the server
        `user_submission_file`: Path to file submitted by the user
        `phase_codename`: Phase to which submission is made

        `**kwargs`: keyword arguments that contains additional submission
        metadata that challenge hosts can use to send slack notification.
        You can access the submission metadata
        with kwargs['submission_metadata']

        Example: A sample submission metadata can be accessed like this:
        >>> print(kwargs['submission_metadata'])
        {
            'status': u'running',
            'when_made_public': None,
            'participant_team': 5,
            'input_file': 'https://abc.xyz/path/to/submission/file.json',
            'execution_time': u'123',
            'publication_url': u'ABC',
            'challenge_phase': 1,
            'created_by': u'ABC',
            'stdout_file': 'https://abc.xyz/path/to/stdout/file.json',
            'method_name': u'Test',
            'stderr_file': 'https://abc.xyz/path/to/stderr/file.json',
            'participant_team_name': u'Test Team',
            'project_url': u'http://foo.bar',
            'method_description': u'ABC',
            'is_public': False,
            'submission_result_file': 'https://abc.xyz/path/result/file.json',
            'id': 123,
            'submitted_at': u'2017-03-20T19:22:03.880652Z'
        }
    """
    
    output = {}
    # if phase_codename == "dev":
    print(f"Evaluating for {phase_codename} Phase")
    data_dict = merge_dicts(test_annotation_file, user_submission_file)
    print("Data Dicts Merged")

    ## Edit for both oe and mcq
    result_qa_pair = {"oe": [], "mcq": []}

    result_qa_pair["oe"] = annotate_combined_parallel("groq", data_dict["oe"])
    print("LLM Judge Scoring For OE QAs Done !")
    # accuracy, average_score = compute_accuracy(result_qa_pair)
    result_metrics = calculate_metrics(result_qa_pair, data_dict)

    accuracy_mcq = (result_metrics["mcq"]['overall']['correct_count'] / max(result_metrics["mcq"]['overall']['count'], 1))*100
    
    avg_score_oe = result_metrics["oe"]['overall']['total_score'] / max(result_metrics["oe"]['overall']['count'], 1)
    accuracy_oe = (avg_score_oe/5)*100

    # accuracy = result_metrics["overall"]['accuracy']
    accuracy = (accuracy_mcq + accuracy_oe) / 2
    print("Accuracy Computed !")
    output["result"] = [
        {
            "split": phase_codename,
            "show_to_participant": True,
            "accuracies": {"MCQ_Acc": accuracy_mcq, "OE_Acc": accuracy_oe, "AVG_Acc": accuracy},
        },
    ]   
    # output["metrics"] = result_metrics
    print(f'Accuracy: {accuracy}')
    print("Completed evaluation for Dev Phase")
    # elif phase_codename == "test":
    #     # dummy test phase
    #     print("Evaluating for Test Phase")
    #     output["result"] = [
    #         {
    #             "split": "test",
    #             "show_to_participant": True,
    #             "accuracies": {"Acc": 0.0004166666666666667},
    #         },
    #     ]
    #     output["metrics"] = { "overall": {"accuracy": 0.0004166666666666667, "score": 0.001, "count": 1} }
    #     print(f'Accuracy: {0.01}')
    #     print("Completed dummy evaluation for Test Phase. Sleeping for 10 seconds.")
    #     time.sleep(10)

    return output
    # return result_metrics





















##########################################################################################

# import random
# from pycocotools.coco import COCO
# from evaluation_script.cocoeval_mp import COCOevalMP

# def evaluate(test_annotation_file, user_submission_file, phase_codename, **kwargs):
#     print("Starting Evaluation.....")
#     """
#     Evaluates the submission for a particular challenge phase and returns score
#     Arguments:

#         `test_annotations_file`: Path to test_annotation_file on the server
#         `user_submission_file`: Path to file submitted by the user
#         `phase_codename`: Phase to which submission is made

#         `**kwargs`: keyword arguments that contains additional submission
#         metadata that challenge hosts can use to send slack notification.
#         You can access the submission metadata
#         with kwargs['submission_metadata']

#         Example: A sample submission metadata can be accessed like this:
#         >>> print(kwargs['submission_metadata'])
#         {
#             'status': u'running',
#             'when_made_public': None,
#             'participant_team': 5,
#             'input_file': 'https://abc.xyz/path/to/submission/file.json',
#             'execution_time': u'123',
#             'publication_url': u'ABC',
#             'challenge_phase': 1,
#             'created_by': u'ABC',
#             'stdout_file': 'https://abc.xyz/path/to/stdout/file.json',
#             'method_name': u'Test',
#             'stderr_file': 'https://abc.xyz/path/to/stderr/file.json',
#             'participant_team_name': u'Test Team',
#             'project_url': u'http://foo.bar',
#             'method_description': u'ABC',
#             'is_public': False,
#             'submission_result_file': 'https://abc.xyz/path/result/file.json',
#             'id': 123,
#             'submitted_at': u'2017-03-20T19:22:03.880652Z'
#         }
#     """
#     v3det_gt = COCO(test_annotation_file)  # gt annotation file
#     v3det_dt = v3det_gt.loadRes(user_submission_file)  # coco-format det results
#     v3det_eval = COCOevalMP(v3det_gt, v3det_dt, 'bbox', num_proc=8)
#     v3det_eval.params.maxDets = [300]

#     v3det_eval.evaluate()
#     v3det_eval.accumulate()
#     v3det_eval.summarize()
#     # output = {}
#     # if phase_codename == "dev":
#     #     print("Evaluating for Dev Phase")
#     #     output["result"] = [
#     #         {
#     #             "train_split": {
#     #                 "Metric1": random.randint(0, 99),
#     #                 "Metric2": random.randint(0, 99),
#     #                 "Metric3": random.randint(0, 99),
#     #                 "Total": random.randint(0, 99),
#     #             }
#     #         }
#     #     ]
#     #     # To display the results in the result file
#     #     output["submission_result"] = output["result"][0]["train_split"]
#     #     print("Completed evaluation for Dev Phase")
#     # elif phase_codename == "test":
#     #     print("Evaluating for Test Phase")
#     #     output["result"] = [
#     #         {
#     #             "train_split": {
#     #                 "Metric1": random.randint(0, 99),
#     #                 "Metric2": random.randint(0, 99),
#     #                 "Metric3": random.randint(0, 99),
#     #                 "Total": random.randint(0, 99),
#     #             }
#     #         },
#     #         {
#     #             "test_split": {
#     #                 "Metric1": random.randint(0, 99),
#     #                 "Metric2": random.randint(0, 99),
#     #                 "Metric3": random.randint(0, 99),
#     #                 "Total": random.randint(0, 99),
#     #             }
#     #         },
#     #     ]
#     #     # To display the results in the result file
#     #     output["submission_result"] = output["result"][0]
#     #     print("Completed evaluation for Test Phase")
#     # output = dict()
#     # output['result'] = "empty"
#     # return output

#     output = {}
#     if phase_codename == "dev":
#         print("Evaluating for Dev Phase")
#         output["result"] = [
#             {
#                 "OVD": {
#                     "bAP": random.randint(0, 99),
#                     "nAP": random.randint(0, 99),
#                     "AP": random.randint(0, 99),
#                 }
#             }
#         ]
#         print("Completed evaluation for Dev Phase")
#     elif phase_codename == "test":
#         print("Evaluating for Test Phase")
#         output["result"] = [
#             {
#                 "split": "train_split",
#                 "show_to_participant": True,
#                 "accuracies": {"Metric1": 90},
#             },
#             {
#                 "split": "test_split",
#                 "show_to_participant": False,
#                 "accuracies": {"Metric1": 50, "Metric2": 40},
#             },
#         ]
#         print("Completed evaluation for Test Phase")
#     return output
