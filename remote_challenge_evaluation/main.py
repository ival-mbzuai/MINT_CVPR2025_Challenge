import json
import os
import time

import requests

from eval_ai_interface import EvalAI_Interface
from evaluate import evaluate

# Remote Evaluation Meta Data
# See https://evalai.readthedocs.io/en/latest/evaluation_scripts.html#writing-remote-evaluation-script
auth_token = os.environ["eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ0b2tlbl90eXBlIjoicmVmcmVzaCIsImV4cCI6MTc2ODQwOTQwNSwianRpIjoiYzMyNGU2ZmJjNWQ1NDE1OWI0ZGQ5ZDQwZDgxOGEwMGIiLCJ1c2VyX2lkIjo1MDc0Nn0.X9j7cT7ovV3KyMiE3zwjNGhIP46UsR4DG-tnJMmxG80"]
evalai_api_server = os.environ["https://eval.ai"]
queue_name = os.environ["v3det-challenge-2024-vast-vocabulary-visual-dete-2445-production-365662f9-2b88-4"]
challenge_pk = os.environ["2445"]
save_dir = os.environ.get("SAVE_DIR", "./SUBMISSIONS/")


def download(submission, save_dir):
    response = requests.get(submission["input_file"])
    submission_file_path = os.path.join(
        save_dir, submission["input_file"].split("/")[-1]
    )
    with open(submission_file_path, "wb") as f:
        f.write(response.content)
    return submission_file_path


def update_running(evalai, submission_pk):
    status_data = {
        "submission": submission_pk,
        "submission_status": "RUNNING",
    }
    update_status = evalai.update_submission_status(status_data)


def update_failed(
    evalai, phase_pk, submission_pk, submission_error, stdout="", metadata=""
):
    submission_data = {
        "challenge_phase": phase_pk,
        "submission": submission_pk,
        "stdout": stdout,
        "stderr": submission_error,
        "submission_status": "FAILED",
        "metadata": metadata,
    }
    update_data = evalai.update_submission_data(submission_data)


def update_finished(
    evalai,
    phase_pk,
    submission_pk,
    result,
    submission_error="",
    stdout="",
    metadata="",
):
    submission_data = {
        "challenge_phase": phase_pk,
        "submission": submission_pk,
        "stdout": stdout,
        "stderr": submission_error,
        "submission_status": "FINISHED",
        "result": result,
        "metadata": metadata,
    }
    update_data = evalai.update_submission_data(submission_data)

def extract_unique_ids(json_data):
    oe_ids = [entry['unique_id'] for entry in json_data['oe']]
    mcq_ids = [entry['unique_id'] for entry in json_data['mcq']]
    return oe_ids + mcq_ids  # Return combined list of unique IDs

def check_submission_file_ids(val_annotation_file_path, submission_file_path):
    with open(submission_file_path, "r", encoding="utf-8") as file:
        data_sub = json.load(file)

    with open(val_annotation_file_path, "r", encoding="utf-8") as file:
        data_val = json.load(file)

    # Extract unique IDs from both JSONs
    val_json_ids = extract_unique_ids(data_val)
    sub_json_ids = extract_unique_ids(data_sub)

    # Check if all unique_ids from second JSON are in the first JSON
    missing_ids = set(sub_json_ids) - set(val_json_ids)
    if missing_ids:
        # print("Missing unique_ids from second JSON:", missing_ids)
        return False
    else:
        # print("No unique_ids are missing in the second JSON.")
        return True


if __name__ == "__main__":
    evalai = EvalAI_Interface(auth_token, evalai_api_server, queue_name, challenge_pk)

    while True:
        # Get the message from the queue
        message = evalai.get_message_from_sqs_queue()
        message_body = message.get("body")
        if message_body:
            submission_pk = message_body.get("submission_pk")
            challenge_pk = message_body.get("challenge_pk")
            phase_pk = message_body.get("phase_pk")
            # Get submission details -- This will contain the input file URL
            submission = evalai.get_submission_by_pk(submission_pk)
            challenge_phase = evalai.get_challenge_phase_by_pk(phase_pk)
            if (
                submission.get("status") == "finished"
                or submission.get("status") == "failed"
                or submission.get("status") == "cancelled"
            ):
                message_receipt_handle = message.get("receipt_handle")
                evalai.delete_message_from_sqs_queue(message_receipt_handle)

            else:
                if submission.get("status") == "submitted":
                    update_running(evalai, submission_pk)
                submission_file_path = download(submission, save_dir)
                val_annotation_file_path = "./annotations/combined_GT_val_rough_3.json"
                test_annotation_file_path = "./annotations/combined_GT_val_rough_3.json"

                ## Check for ids present in submission file
                if challenge_phase["codename"] == "val":
                    all_ids_present = check_submission_file_ids(val_annotation_file_path, submission_file_path)
                else:
                    all_ids_present = check_submission_file_ids(test_annotation_file_path, submission_file_path)

                if all_ids_present:
                    try:
                        if challenge_phase["codename"] == "val":
                            results = evaluate(
                                val_annotation_file_path, submission_file_path, challenge_phase["codename"]
                            )
                        else:
                            results = evaluate(
                                test_annotation_file_path, submission_file_path, challenge_phase["codename"]
                            )
                        update_finished(
                            evalai, phase_pk, submission_pk, json.dumps(results["result"])
                        )
                    except Exception as e:
                        update_failed(evalai, phase_pk, submission_pk, str(e))
                else:
                    update_failed(evalai, phase_pk, submission_pk, "Missing data in submission file")
        # Poll challenge queue for new submissions
        time.sleep(5)
