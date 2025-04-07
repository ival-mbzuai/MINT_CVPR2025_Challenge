
import os
import openai
from groq import Groq

groq_client = Groq(api_key="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")
# openai_client = openai.OpenAI()




def inference_openai(question, answer, pred):
    # Compute the correctness score
    completion = openai_client.chat.completions.create(
        model="GPT-3.5-turbo-0125",
        messages=[
            {
                "role": "system",
                "content":
                    "You are an intelligent chatbot designed for evaluating the correctness of AI assistant predictions for question-answer pairs. "
                    "Your task is to compare the predicted answer with the ground-truth answer and determine if the predicted answer is correct or not. Here's how you can accomplish the task:"
                    "------"
                    "##INSTRUCTIONS: "
                    "- Focus on the correctness and accuracy of the predicted answer with the ground-truth.\n"
                    "- Consider predictions with less specific details as correct evaluation, unless such details are explicitly asked in the question.\n"
            },
            {
                "role": "user",
                "content":
                    "Please evaluate the following video-based question-answer pair:\n\n"
                    f"Question: {question}\n"
                    f"Ground truth correct Answer: {answer}\n"
                    f"Predicted Answer: {pred}\n\n"
                    "Provide your evaluation as a correct/incorrect prediction along with the score where the score is an integer value between 0 (fully wrong) and 5 (fully correct). The middle score provides the percentage of correctness."
                    "Please generate the response in the form of a Python dictionary string with keys 'pred', 'score' and 'reason', where value of 'pred' is  a string of 'correct' or 'incorrect', value of 'score' is in INTEGER, not STRING and value of 'reason' should providethe reason behind the decision."
                    "Only provide the Python dictionary string."
                    "For example, your response should look like this: {'pred': 'correct', 'score': 4.8, 'reason': reason}."
            }
        ]
    )
    # Convert response to a Python dictionary.
    response_message = completion.choices[0].message.content

    return response_message

def inference_groq(question, answer, pred):

    completion = groq_client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
                {
                    "role": "system",
                    "content":
                        "You are an intelligent chatbot designed for evaluating the correctness of AI assistant predictions for question-answer pairs. "
                        "Your task is to compare the predicted answer with the ground-truth answer and determine if the predicted answer is correct or not. Here's how you can accomplish the task:"
                        "------"
                        "##INSTRUCTIONS: "
                        "- Focus on the correctness and accuracy of the predicted answer with the ground-truth.\n"
                        "- Consider predictions with less specific details as correct evaluation, unless such details are explicitly asked in the question.\n"
                        "- If predicted answer is semantically correct but not in the same language as ground-truth, then penalize it.\n"
                },
                {
                    "role": "user",
                    "content":
                        "Please evaluate the following video-based question-answer pair:\n\n"
                        f"Question: {question}\n"
                        f"Ground truth correct Answer: {answer}\n"
                        f"Predicted Answer: {pred}\n\n"
                        "Provide your evaluation as a correct/incorrect prediction along with the score where the score is an integer value between 0 (fully wrong) and 5 (fully correct). The middle score provides the percentage of correctness."
                        "Please generate the response in the form of a Python dictionary string with keys 'pred', 'score' and 'reason', where value of 'pred' is  a string of 'correct' or 'incorrect', value of 'score' is in INTEGER, not STRING and value of 'reason' should providethe reason behind the decision."
                        "Only provide the Python dictionary string."
                        "For example, your response should look like this: {'pred': 'correct', 'score': 4.8, 'reason': reason}."
                }
            ],
        temperature=0.,
        max_completion_tokens=1024
    )
    # completion
    response_message = completion.choices[0].message.content
    # print(response_message)
    return response_message






