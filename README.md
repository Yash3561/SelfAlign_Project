# DS677: Self-Align Data Generation Pipeline for Go

This repository contains the code, scripts, and generated data for the DS677 final project, implementing the Self-Align data generation pipeline specifically for the Go programming language.

**Team:**
*   Yash Chaudhary (ygc2)
*   Kaviya Sree Ravikumar Meenakshi (kr549)

## Project Goal

The primary goal of this project was to adapt and execute the multi-step Self-Align pipeline (originally designed for Python) to automatically generate a dataset of high-quality instruction-response pairs for the Go language. This resulting dataset can potentially be used for fine-tuning instruction-following language models to improve their Go code generation capabilities.

## Framework Overview

The project follows the 3-step Self-Align framework:

1.  **Step 1: Seed Dataset Curation:** Extracting initial Go function/comment pairs from a source dataset (The Stack v2) and applying heuristic and LLM-based filters to select high-quality seeds.
2.  **Step 2: Self-OSS-Instruct:** Using an LLM (StarCoder2) to generate programming concepts from the seed code (S->C) and then generate corresponding task instructions based on those concepts (C->I).
3.  **Step 3: Self-Validation:** Using the LLM to generate potential Go code solutions and Go tests for the instructions (I->R), and then executing these tests (`go test`) to filter for pairs where the generated code passes the generated tests.

## Directory Structure

```text
SelfAlign_Project/
├── Step_1_SeedGathering/                 # Notebook and outputs from Step 1 (Initial extraction + filtering)
│   ├── Step1_GOLang.ipynb
│   ├── seed1_extracted_dataset/
│   ├── seed2_heuristically_filtered_subset/
│   └── seed3_llm_filtered_subset/        <-- Input for Step 2 (S->C)
├── Step2_SelfInstruct/                   # Contains cloned selfcodealign repo and Step 2 results/scripts
│   ├── selfcodealign/                    # Cloned bigcode-project/selfcodealign repo
│   │   ├── prompts/                      # Contains edited self-ossinstruct-fewshot.txt
│   │   ├── src/star_align/              # Edited self_ossinstruct.py, execution_filter.py, etc.
│   │   └── requirements.txt             # Original repo requirements
│   ├── results/                          # Output .jsonl files from Step 2 (S->C, C->I) and Step 3 (I->R)
│       ├── data...Go-SC...jsonl         <-- Output of S->C, Input for C->I
│       └── data...Go-CI...jsonl         <-- Output of C->I, Input for I->R
├── Step3_Validated_Data/                 # Final output of the pipeline
│   └── validated_go_instructions_final.jsonl  <-- Final validated dataset
├── requirements.txt                      # Requirements for the Conda environment (pip freeze output)
└── README.md                             # This file
```


## Setup

1.  **Clone this Repository:**
    ```bash
    git clone https://github.com/Yash3561/DS677_Go.git
    cd DS677_Go
    ```
2.  **Conda Environment:** Create and activate the Conda environment using Python 3.11.11 and the provided `requirements.txt`.
    ```bash
    # Create the environment specifying the Python version
    conda create --name selfalign_env python=3.11.11
    conda activate selfalign_env
    # Install packages using pip and the requirements file
    pip install -r requirements.txt
    ```
3.  **Install Go:** The Go compiler is required for Step 3 validation. Install it within the environment or ensure it's available in your system PATH. Verify with `go version`.
    ```bash
    # Activate env first: conda activate selfalign_env
    # Recommended method within conda env
    conda install go
    # Or load module on HPC
    # module load go/1.xx
    ```
4.  **Install `goimports` (Optional but Recommended for Validation):** Helps format Go code and add missing standard library imports.
    ```bash
    # Activate env first: conda activate selfalign_env
    go install golang.org/x/tools/cmd/goimports@latest
    # Ensure $HOME/go/bin is in your PATH
    # export PATH=$HOME/go/bin:$PATH # Add to ~/.bashrc if needed
    which goimports # Verify installation
    ```
5.  **Clone `selfcodealign` Repository:** The Step 2/3 scripts rely on the code structure from the original repository. Clone it into the `Step2_SelfInstruct` directory (or adjust paths if cloned elsewhere).
    ```bash
    # Navigate to the intended location
    cd ~/SelfAlign_Project/Step2_SelfInstruct/
    # Clone if not already done
    git clone https://github.com/bigcode-project/selfcodealign.git
    # Install local package dependencies
    cd selfcodealign
    conda activate selfalign_env # Ensure env is active
    pip install -e .
    cd ../.. # Go back to project root: ~/SelfAlign_Project/
    ```

## Running the Pipeline

*(Note: Assumes setup is complete and the `selfalign_env` conda environment is activated.)*

**Step 1: Seed Curation (Notebook)**

*   Navigate to `Step_1_SeedGathering/`.
*   Run the cells in `Step1_GOLang.ipynb`.
*   This performs extraction, heuristic filtering, and LLM filtering.
*   **Input:** None (Connects to Hugging Face Datasets)
*   **Output:** Saves intermediate datasets and the final filtered seeds to `Step_1_SeedGathering/seed3_llm_filtered_subset/`.

**Step 2: Self-OSS-Instruct (Requires vLLM Server & Script)**

1.  **Modify Scripts:**
    *   Ensure `Step2_SelfInstruct/selfcodealign/prompts/self-ossinstruct-fewshot.txt` contains the Go-specific system prompts and few-shot examples.
    *   Ensure `Step2_SelfInstruct/selfcodealign/src/star_align/self_ossinstruct.py` has been modified for Go (dataset loading logic, hardcoded language in `build_kwargs`).
2.  **Start vLLM Server (Terminal 1):**
    ```bash
    conda activate selfalign_env
    MODEL="bigcode/starcoder2-15b" # Or 3b
    python -m vllm.entrypoints.openai.api_server \
        --model $MODEL \
        --dtype bfloat16 \
        --gpu-memory-utilization 0.80 \
        --max-model-len 4096
    ```
3.  **Run S->C and C->I (Terminal 2):**
    ```bash
    conda activate selfalign_env
    # Navigate to the root of the cloned repo inside Step2_Selfinstruct
    cd ~/SelfAlign_Project/Step2_Selfinstruct/selfcodealign/

    export OPENAI_API_KEY="EMPTY"
    export OPENAI_BASE_URL="http://localhost:8000/v1"

    # --- Variables ---
    INPUT_DATA_DIR_SC="../seed3_llm_filtered_subset" # Relative path to Step 1 output dir
    SAVE_DIR="../results"                           # Relative path for Step 2 output dir
    NUM_FEWSHOTS=6
    MODEL_NAME=$MODEL # Match server
    B_REQ=128 # Batched requests
    A_BATCH=4 # Async micro-batch

    mkdir -p $SAVE_DIR

    # --- Run S->C ---
    MODE_SC="S->C"
    TAG_SUFFIX_SC="Go-SC-FinalRun"
    echo "Running S->C..."
    python -m star_align.self_ossinstruct \
        --instruct_mode "$MODE_SC" \
        --seed_data_files "$INPUT_DATA_DIR_SC" \
        --save_dir "$SAVE_DIR" \
        --model "$MODEL_NAME" \
        --num_fewshots $NUM_FEWSHOTS --temperature 0.2 --max_output_tokens 256 \
        --num_batched_requests $B_REQ --async_micro_batch_size $A_BATCH \
        --num_sample_per_request 1 --use_vllm_server True --max_new_data -1 \
        --tag "sc2-${NUM_FEWSHOTS}shot-${TAG_SUFFIX_SC}" --seed_code_start_index 0

    if [ $? -ne 0 ]; then echo "S->C FAILED!"; exit 1; fi

    # --- Find S->C output file ---
    S_C_OUTPUT_FILE=$(ls -t "$SAVE_DIR"/data*${TAG_SUFFIX_SC}*.jsonl | head -n 1)
    if [ -z "$S_C_OUTPUT_FILE" ]; then echo "S->C output file not found!"; exit 1; fi

    # --- Run C->I ---
    MODE_CI="C->I"
    TAG_SUFFIX_CI="Go-CI-FinalRun"
    echo "Running C->I using $S_C_OUTPUT_FILE..."
    python -m star_align.self_ossinstruct \
       --instruct_mode "$MODE_CI" \
       --seed_data_files "$S_C_OUTPUT_FILE" \
       --save_dir "$SAVE_DIR" \
       --model "$MODEL_NAME" \
       --num_fewshots $NUM_FEWSHOTS --temperature 0.7 --max_output_tokens 512 \
       --num_batched_requests $B_REQ --async_micro_batch_size $A_BATCH \
       --num_sample_per_request 1 --use_vllm_server True --max_new_data -1 \
       --tag "sc2-${NUM_FEWSHOTS}shot-${TAG_SUFFIX_CI}" --seed_code_start_index 0

    if [ $? -ne 0 ]; then echo "C->I FAILED!"; exit 1; fi
    echo "Step 2 (S->C and C->I) outputs saved in $SAVE_DIR"
    # Note the filename of the C->I output (e.g., data...Go-CI-FinalRun....jsonl)
    ```

**Step 3: Self-Validation (Script)**

1.  **Prepare Validation Script:** Ensure `validate_go_instructions.py` exists (e.g., in `Step2_SelfInstruct/`) and is updated to handle Go, potentially using `goimports`. Ensure the `input_jsonl_file` path inside it points to the correct C->I output file, and `output_validated_file` points to the desired final location (e.g., `../Step3_Validated_Data/validated_go_instructions_final.jsonl`).
2.  **Run Validation:**
    ```bash
    conda activate selfalign_env
    # Navigate to where validate_go_instructions.py is saved
    cd ~/SelfAlign_Project/Step2_SelfInstruct/
    # Ensure Go is installed and goimports is in PATH
    go version
    goimports -h # Check if goimports is found
    python validate_go_instructions.py
    ```
    *   **Input:** The C->I or I->R output JSONL file (specified inside the script).
    *   **Output:** `Step3_Validated_Data/validated_go_instructions_final.jsonl` containing pairs where `go test` passed.

## Results

The pipeline execution resulted in the following dataset sizes at each stage:

*   Initial Seed Sample (Step 1 Start): ~40,000
*   After Heuristic + LLM Filtering (Step 1 End): ~3,511
*   After Self-OSS-Instruct (Step 2 End): ~3,511 (Instruction-Seed pairs)
*   After Self-Validation (Step 3 End): **~567** (Validated Instruction-Response pairs)

The final validated dataset containing Go instruction-response pairs suitable for fine-tuning is located at: `Step3_Validated_Data/validated_go_instructions_final.jsonl`.

## Challenges & Notes

*   Adapting Python-based prompts and scripts for Go required careful modification, especially for few-shot examples and execution logic.
*   The `execution_filter.py` script from the original repository was Python-centric and required a custom script (`validate_go_instructions.py`) for Go test execution using `go test`. Integrating `goimports` is recommended for robustness.
*   LLM generation for Go code and tests showed variability; the self-validation step significantly filtered the data, retaining only pairs that passed compilation and basic testing.
*   HPC environment setup (Conda, Git, Go, vLLM, disk quotas, module paths) required careful management.
