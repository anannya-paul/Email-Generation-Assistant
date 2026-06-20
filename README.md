# Email Generation Assistant
# Project README: Email Generation Assistant

This project implements an AI assistant for generating and evaluating emails using the Llama 3 language model. It features two prompting strategies—'Original' and 'Concise'—and a custom evaluation framework to compare their performance across various metrics.

## Table of Contents
1.  [Project Overview](#project-overview)
2.  [Setup Instructions](#setup-instructions)
3.  [Execution Guide](#execution-guide)
4.  [Key Components](#key-components)
5.  [Results and Analysis](#results-and-analysis)

## 1. Project Overview

The core of this project is to:
*   Generate professional emails based on user-defined intent, key facts, and tone using the Llama 3 model.
*   Evaluate the quality of these generated emails using custom metrics: Fact Recall, Tone Accuracy, and Conciseness & Clarity.
*   Compare the performance of different prompting strategies to identify optimal approaches.

## 2. Setup Instructions

To set up and run this project, follow these steps:

### 2.1 Install Dependencies
First, install all necessary Python libraries. These are typically installed in the first code cell of the notebook.

```python
!pip install transformers accelerate bitsandbytes
!pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
!pip install textstat
```

### 2.2 Hugging Face Login

The Llama 3 model (`meta-llama/Meta-Llama-3-8B-Instruct`) is a gated model on Hugging Face. You must log in to your Hugging Face account with a valid token to access it. If you don't have a token, you can generate one from your [Hugging Face settings page](https://huggingface.co/settings/tokens).

Run the following code cell and enter your token when prompted:

```python
import huggingface_hub
huggingface_hub.login()
```

### 2.3 Load the Llama 3 Model

The project uses 4-bit quantization to efficiently load the Llama 3 model. The tokenizer and model are loaded using `transformers` library:

```python
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
 )

model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    quantization_config=bnb_config,
    device_map="auto",
)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
```

## 3. Execution Guide

### 3.1 Define Email Generation Prompts
The project includes two prompting functions:
*   `generate_email_prompt`: The original, more detailed prompting strategy.
*   `generate_email_prompt_v2`: A more concise and direct prompting strategy.

These functions construct the `messages` list that is fed to the Llama 3 model.

### 3.2 Define Evaluation Functions
Three evaluation functions are defined to assess different aspects of the generated emails:
*   `evaluate_fact_recall`: Checks if all key facts are present and accurate.
*   `evaluate_tone_and_format`: Assesses adherence to the desired tone and basic email formatting.
*   `evaluate_conciseness_and_clarity`: Evaluates the email's brevity and comprehensibility.

An `extract_llm_score` helper function is used to parse scores from LLM-based evaluations.

### 3.3 Create Scenarios
A list of `scenarios` is defined, each containing the `intent`, `key_facts`, `tone`, and a `human_reference_email` for evaluation purposes.

### 3.4 Run the Evaluation
The `run_evaluation_for_strategy` function iterates through each scenario, generates emails using a specified strategy (e.g., `generate_email` or `generate_email_v2`), and performs all three evaluations. It then compiles a `full_evaluation_report` that combines results from all strategies.

To run the evaluation for both strategies, execute the cell containing:

```python
llama3_original_report = run_evaluation_for_strategy(generate_email, "Llama 3 Original", scenarios)
llama3_concise_report = run_evaluation_for_strategy(generate_email_v2, "Llama 3 Concise", scenarios)
full_evaluation_report = llama3_original_report + llama3_concise_report
```

## 4. Key Components

*   **`_llama_generate_response(messages)`**: A core helper function for interacting with the loaded Llama 3 model, handling tokenization, generation parameters, and decoding. It's called by both email generation functions and evaluation functions.
*   **`generate_email(intent, key_facts, tone)`**: Generates an email using the 'Original' prompting strategy.
*   **`generate_email_v2(intent, key_facts, tone)`**: Generates an email using the 'Concise' prompting strategy.
*   **`scenarios`**: A list of dictionaries defining the test cases for email generation and evaluation.
*   **`run_evaluation_for_strategy(...)`**: Orchestrates the email generation and evaluation process for a given strategy across all scenarios.

## 5. Results and Analysis

After running the evaluation, the `df_results` DataFrame will contain all individual scenario results for both strategies. Average scores for each metric and overall average are calculated and displayed.

A bar plot visually compares the performance of the 'Llama 3 Original' and 'Llama 3 Concise' strategies across all metrics, providing a clear overview of their strengths and weaknesses.
