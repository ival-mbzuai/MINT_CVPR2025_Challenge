import ast
import time
from tqdm import tqdm
import concurrent.futures
from remote_inference_providers import inference_openai, inference_groq

def process_single_item(item_tuple, provider):
    """
    Process a single question/answer/prediction item.
    
    Args:
        item_tuple (tuple): Tuple containing (key, dict) where dict has q, a, and pred keys
        provider (str): Provider name ("openai" or "groq")
        
    Returns:
        tuple: (response_dict, original_dict) or (error_message, original_dict)
    """
    k, single_dict = item_tuple
    question = single_dict['q']
    answer = single_dict['a']
    pred = single_dict['pred']
    
    try:
        # Call external inference functions
        if provider == "openai":
            response_message = inference_openai(question, answer, pred)
        elif provider == "groq":
            response_message = inference_groq(question, answer, pred)
        else:
            raise ValueError(f"Unsupported provider: {provider}")
            
        # Attempt to parse the response
        response_dict = ast.literal_eval(response_message)
        # print(response_dict)
        return (response_dict, single_dict)
    except Exception as e:
        return (f"Error evaluating item {k}: {e}", single_dict)

def annotate_combined_parallel(provider, prediction_set, max_workers=4):
    """
    Evaluates question and answer pairs using specified inference provider in parallel.
    
    Args:
        provider (str): Provider name ("openai" or "groq").
        prediction_set (dict): Dictionary of prediction data structured as:
            {id1: {'q': question, 'a': answer, 'pred': prediction}, ...}
        max_workers (int): Maximum number of concurrent API calls.
        
    Returns:
        list: [[evaluation_result_dict, original_prediction_dict], ...]
    """
    result_qa_pair = []

    cumulative_length = 0
    
    # Create a ThreadPoolExecutor to handle concurrent API calls
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks to the executor
        future_to_item = {
            executor.submit(process_single_item, (k, single_dict), provider): (k, single_dict)
            for k, single_dict in prediction_set.items()
        }
        
        # Process results as they complete
        for future in tqdm(concurrent.futures.as_completed(future_to_item), 
                           total=len(future_to_item), 
                           desc=f"Processing with {provider}", 
                           smoothing=0.):
            result = future.result()
            
            # Check if result is an error
            if isinstance(result[0], str) and result[0].startswith("Error"):
                print(result[0])  # Print error but continue processing
            else:
                result_qa_pair.append(list(result))
            
            # # Update cumulative length
            # single_dict = future_to_item[future][1]  # Retrieve the corresponding item
            # question = single_dict['q']
            # answer = single_dict['a']
            # pred = single_dict['pred']
            
            # cumulative_length += len(question) + len(answer) + len(pred)
            
            # # Delay logic based on cumulative length
            # if cumulative_length > 5000:
            #     print("Cumulative length exceeded 5000, delaying for 60 seconds...")
            #     time.sleep(60)
            #     cumulative_length = 0  # Reset cumulative length after the delay
            # else:
            #     # Optional: Add a small delay to avoid overwhelming the API
            #     time.sleep(10)
            time.sleep(3)
            
    return result_qa_pair

# Example usage:
# results = annotate_combined_parallel("openai", prediction_set, max_workers=4)