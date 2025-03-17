import numpy as np
from tqdm import tqdm
import litellm
import openai
from datasets import load_dataset
import os
from dotenv import load_dotenv
from typing import List, Tuple, Dict, Union, Any, Optional


load_dotenv()


def read_data(seed: int, train_size: int, test_size: int) -> Tuple[List[str], List[int], List[str], List[int]]:
    """
    Load and prepare the SST-2 dataset.

    Args:
        seed (int): Random seed for dataset shuffling
        train_size (int): Number of training samples to select
        test_size (int): Number of testing samples to select

    Returns:
        Tuple containing:
            - List[str]: Training sentences
            - List[int]: Training labels (0 for negative, 1 for positive)
            - List[str]: Test sentences
            - List[int]: Test labels (0 for negative, 1 for positive)
    """

    # Load SST-2 dataset
    dataset = load_dataset("stanfordnlp/sst2")

    # TODO: Get training and testing datasets from `train` and `validation` splits
    # Step 1: shuffle with seed
    seeded_dataset = dataset.shuffle(seed) #dataset.train_test_split(test_size=test_size, train_size=train_size, shuffle=True, seed=seed)
    # Step 2: select first train/test size data
    train_data = seeded_dataset["train"][0:train_size]
    test_data = seeded_dataset["validation"][0:test_size]

    # TODO: Extract sentences and labels
    print(test_data)
    train_sentences = train_data["sentence"]
    train_labels = train_data["label"]
    test_sentences = test_data["sentence"]
    test_labels = test_data["label"]

    return train_sentences, train_labels, test_sentences, test_labels


def create_prompt(
    q_prefix: str,
    a_prefix: str,
    few_shot_sentences: List[str], 
    few_shot_labels: List[int], 
    test_sentence: str
) -> str:
    """
    Create a prompt for sentiment analysis using few-shot examples.

    Args:
        few_shot_sentences (List[str]): List of example sentences for few-shot learning
        few_shot_labels (List[int]): List of corresponding labels (0/1)
        test_sentence (str): The sentence to analyze

    Returns:
        str: Formatted prompt string containing examples and test sentence
    """

    prompt = ""

    # Add few-shot samples
    for s, l in zip(few_shot_sentences, few_shot_labels):
        if isinstance(l, int):
            # Convert label to string
            l = "Positive" if l == 1 else "Negative"
        prompt += f"{q_prefix}{s}\n{a_prefix}{l}\n\n"

    # Add test sentence
    prompt += f"{q_prefix}{test_sentence}\n{a_prefix}"
    if a_prefix[-1] == " ":  # GPT-3 does not need a trailing space
        prompt = prompt[:-1]
    
    return prompt


def get_responses(prompts: List[str], echo: bool = False) -> List[Any]:
    """
    Get responses from the language model for given prompts.

    Args:
        prompts (List[str]): List of prompts to send to the model
        echo (bool, optional): If True, echo back the prompt in addition to the completion

    Returns:
        List[Any]: List of model responses, each containing logprobs and other completion data

    Hint: Read the API documentation, or try to play with the API a few times to understand the usage of logprobs.
    """

    # Set OpenAI client
    client = openai.OpenAI(
        api_key=os.environ.get("OPENAI_API_KEY")
    )

    # Get responses from the model
    responses = []
    max_tokens = 0 if echo else 1

    for prompt in tqdm(prompts, desc="Get response"):
        response = client.completions.create(
            model="davinci-002",
            prompt=prompt,
            max_tokens=max_tokens,  # the maximum number of tokens to generate
            stop='\n',  # stop at the first newline
            logprobs=1,  # return log prob of top-1 token
            echo=echo  # whether to also return the input
        )
        responses.append(response)

    return responses


def get_label_probs(
    responses: List[Any], 
    few_shot_sentences: List[str], 
    few_shot_labels: List[int], 
    test_sentences: List[str],
    q_prefix: str,
    a_prefix: str
) -> np.ndarray:
    """
    Calculate label probabilities from model responses.

    Args:
        responses (List[Any]): Model responses containing logprobs
        few_shot_sentences (List[str]): Example sentences used in prompts
        few_shot_labels (List[int]): Labels for example sentences
        test_sentences (List[str]): Test sentences to analyze
        q_prefix (str): Prefix of the review
        a_prefix (str): Prefix of the sentiment

    Returns:
        np.ndarray: Array of label probabilities (not normalized)
    """

    label_dict = {0: "Negative", 1: "Positive"}
    num_labels = len(label_dict)
    all_label_probs = []
    all_missing_positions = []

    # TODO: Initial probabilities from model responses
    for i, response in enumerate(tqdm(responses, desc="Get initial prob")):
        print(response)
        top_logprobs = response.logprobs.top_logprobs  # Hint: Check the structure of the response
        print(top_logprobs)
        #prediction = response.text
        label_probs = [0, 0] #top_logprobs[prediction]
        #print(label_probs)

        for j, label in label_dict.items():
            # add space to match the format
            if a_prefix[-1] == " ":
                label = " " + label

            # Hint: If the label is in the top logprobs, use the probability
            if label in top_logprobs:
                label_probs[j] = np.exp(top_logprobs[label])
            else:
                # add to missing positions
                all_missing_positions.append((i, j))

        all_label_probs.append(label_probs)
    all_label_probs = np.array(all_label_probs)

    # TODO: Fill in those missing positions
    all_additional_prompts = []

    for i, j in all_missing_positions:
        # Hint: Based on the index, use create_prompt to create a new prompt for the missing position
        prompt = create_prompt(q_prefix, a_prefix, few_shot_sentences[i], few_shot_labels, test_sentences)

        # add space to match the format
        # Hint: It's important to understand why we append the label to the input prompt
        if a_prefix[-1] == " ":
            label = label_dict[j]
            prompt += " " + label
        all_additional_prompts.append(prompt)

    # get responses for additional prompts
    # Hint: We set Echo=True. Why?
    additional_responses = get_responses(all_additional_prompts, echo=True)

    for idx, (i, j) in enumerate(all_missing_positions):
        response = additional_responses[idx]
        # TODO: Get the probability from the response
        print(response)
        top_logprobs = response.choices[0].logprobs.top_logprobs

        for j, label in label_dict.items():
            # add space to match the format
            if a_prefix[-1] == " ":
                label = " " + label

            # Hint: If the label is in the top logprobs, use the probability
            if label in top_logprobs:
                all_label_probs[i][j] = np.exp(top_logprobs[label])

        #all_label_probs[i][j] = np.exp(log_prob)

    return all_label_probs  # this is not normalized


def calibrate(
    content_free_input: str, 
    few_shot_sentences: List[str], 
    few_shot_labels: List[int], 
    q_prefix: str,
    a_prefix: str
) -> np.ndarray:
    """
    Calculate calibration vector using content-free input.

    Args:
        content_free_input (str): Content-free input text (e.g., "N/A")
        few_shot_sentences (List[str]): Example sentences used in prompts
        few_shot_labels (List[int]): Labels for example sentences
        q_prefix (str): Prefix of the review
        a_prefix (str): Prefix of the sentiment

    Returns:
        np.ndarray: Calibration vector (normalized probabilities)
    """

    label_dict = {0: "Negative", 1: "Positive"}
    num_labels = len(label_dict)

    # TODO: Create a prompt with content-free input
    prompt = create_prompt(q_prefix, a_prefix, few_shot_sentences, few_shot_labels, content_free_input)
    p_y = [0] * num_labels

    for i, answer in label_dict.items():
        if a_prefix[-1] == " ":  # to match the prompt format
            key = " "+answer
        else:
            key = answer

        response = get_responses(prompts=[prompt+key], echo=True)[0]

        # TODO: Get the probability from the response
        probabilities = get_label_probs(response, few_shot_sentences, few_shot_labels, prompt, q_prefix, a_prefix)
        print(probabilities)
        p_y[i] = probabilities

    # TODO: Normalize the probabilities
    sum_probs = 0
    for p in p_y:
        print(p)
        sum_probs += p

    p_y = [p/sum_probs for p in p_y]

    return p_y


def eval_accuracy(all_label_probs: np.ndarray, test_labels: List[int], p_cf: Optional[np.ndarray] = None) -> float:
    """
    Evaluate classification accuracy with optional calibration.

    Args:
        all_label_probs (np.ndarray): Array of label probabilities for each test sentence
        test_labels (List[int]): True labels for test sentences
        p_cf (Optional[np.ndarray], optional): Calibration vector. Defaults to None.

    Returns:
        float: Classification accuracy (between 0 and 1)

    Note: We use diagonal matrix here as the paper mentions it's better than the identity matrix for classification
    """

    num_labels = len(test_labels)#.shape[1]

    # TODO: Initialize W and b
    if p_cf is None:
        W = np.identity(num_labels)
        #print(W)
        #b = np.zeros(num_labels)
        predictions_raw = np.matmul(W, all_label_probs)# + b
        predictions = np.argmax(predictions_raw, -1)
        accuracy = np.mean(predictions==test_labels)
        return accuracy

    else:
        #print(p_cf)
        W = np.linalg.inv(np.diag(p_cf))
        #np.sum(W, axis=0)
        #print(W)
        #b = np.zeros(num_labels)
        predictions_raw = []
        for prob in all_label_probs:
            #print(prob)
            prediction_raw = np.matmul(W, prob)
            #print(prediction_raw)
            predictions_raw.append(prediction_raw)
        predictions = np.argmax(predictions_raw, -1)
        accuracy = np.mean(predictions==test_labels)
        return accuracy

    # TODO: Calculate the accuracy
    #corrects = 0 #[]
    '''
    index = 0
    for prob, label in zip(all_label_probs, test_labels):
        print(prob)
        print(label)
        #if prob[label] > 0.5:
        #    corrects +=1
        #corrects.append(prob[label])
        prediction_raw = W * prob[index]
        print(prediction_raw)
        index += 1
    predictions_raw = np.matmul(W, all_label_probs)# + b
    predictions = np.argmax(predictions_raw, -1)
    print(predictions)
    #print(corrects)
    #accuracy = corrects / num_labels #np.mean(corrects)
    accuracy = np.mean(predictions==test_labels)
    print(accuracy)

    return accuracy
    '''
