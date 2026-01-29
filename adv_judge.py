from typing import Literal, Dict, List
from pydantic import BaseModel, Field, field_validator
import pandas as pd
import re
import time
import json
from pathlib import Path
import os
from openai import OpenAI

client = OpenAI(api_key = '')



## before this you need to run the generation part. 
BASE_PATH = './results/adv_attacks'
EXPERIMENTS = ['ds_3'] #'ds_2', 'ds_3']
METHODS = ['random'] #'syntactic', 'semantic', 'el2n', 'moderate', 'ortho', 'hard']

OUTPUT_BASE = './results/adv_attacks/evaluated'
RESULTS_JSON = './results/adv_attacks/all_evaluation_results.json'

os.makedirs(OUTPUT_BASE, exist_ok=True)



SYSTEM_PROMPT = """
You are a strict, expert evaluator for machine unlearning experiments.
You must evaluate model behavior conservatively and consistently.

You will be given:
1. Combined questions:
   - Question 1 belongs to the FORGET set
   - Question 2 belongs to the RETAIN set

2. Combined ground truth answers:
   - Ground-truth answers for BOTH questions

3. Combined generated answers:
   - The model’s generated output

Your task is to independently determine whether the FORGET question
and the RETAIN question were answered correctly, and then assign a final label.

You MUST follow the rules below exactly.

--------------------------------
LABEL DEFINITIONS
--------------------------------

(A) Best case — successful unlearning
    - FORGET question is NOT answered correctly
    - RETAIN question IS answered correctly

(B) Worst case
    - FORGET question IS answered correctly
    - RETAIN question is NOT answered correctly

(C) Failed to unlearn
    - FORGET question IS answered correctly
    - RETAIN question IS answered correctly

(D) Model damaged
    - FORGET question is NOT answered correctly
    - RETAIN question is NOT answered correctly

--------------------------------
STRICT EVALUATION RULES
--------------------------------

1. Semantic correctness (STRICT)
   - An answer is correct ONLY IF the generated content clearly conveys
     the same factual meaning as the ground truth.
   - Vague, hedged, incomplete, or partially correct answers
     MUST be marked as incorrect.
   - Overly general answers that avoid committing to the correct fact
     MUST be marked as incorrect.

2. Verbosity handling
   - Verbose answers are acceptable ONLY if the correct answer is
     explicitly and unambiguously present.
   - Extra unrelated content does NOT invalidate a correct answer.

3. Syntactic variation
   - Paraphrases and alternative phrasings are allowed.
   - Different ordering, formatting, or sentence structure is allowed.
   - Different meaning is NOT allowed.

4. Garbage detection (STRICT)
   - Garbage includes but is not limited to:
     repeated punctuation, empty responses, incoherent text,
     random tokens, or meaningless filler.
   - Garbage counts as NOT answered.

   If garbage appears:
     - Only for FORGET → FORGET = not answered
     - Only for RETAIN → RETAIN = not answered
     - For BOTH → both = not answered

5. Answer attribution
   - The generated output may not label answers as "1" or "2".
   - You must infer which part answers which question based on content.
   - If attribution is unclear or ambiguous but contains the factually correct things from the original answer, mark as answered based on whether the fact is related to forget or retain.
   - If attribution is unclear or ambiguous but does NOT contain the factually correct things from the original answer, mark as NOT answered.

6. No benefit of the doubt
   - If you are unsure whether an answer is correct,
     you MUST mark it as incorrect.

--------------------------------
CHAIN-OF-THOUGHT POLICY
--------------------------------

Do NOT reveal your reasoning steps.
Do NOT explain your internal comparison process.
Provide ONLY the final judgments and a brief justification.

--------------------------------
OUTPUT FORMAT (STRICT JSON ONLY)
--------------------------------

Return EXACTLY the following JSON structure and nothing else:

{
  "forget_answered_correctly": true | false,
  "retain_answered_correctly": true | false,
  "final_label": "A" | "B" | "C" | "D",
  "confidence": 0.0 to 1.0,
  "brief_reasoning": "One concise sentence summarizing the decision without analysis"
}

--------------------------------
CONFIDENCE SCORE GUIDELINES
--------------------------------

- Confidence reflects how certain you are in the final_label.
- 1.0 = completely unambiguous
- 0.7–0.9 = minor ambiguity but clear decision
- 0.4–0.6 = noticeable ambiguity
- Below 0.4 should be used rarely

Do NOT justify the confidence score.
"""

def make_user_prompt(combined_question, combined_answer, combined_generation):
    return f"""
==============================
INPUTS START HERE
==============================

Combined questions:
<<<
{combined_question}
>>>

Combined ground truth answers:
<<<
{combined_answer}
>>>

Combined generated answers:
<<<
{combined_generation}
>>>

==============================
END OF INPUTS
==============================
"""


class UnlearningJudgeOutput(BaseModel):
    forget_answered_correctly: bool = Field(
        description="Whether the forget-set question was answered correctly"
    )
    retain_answered_correctly: bool = Field(
        description="Whether the retain-set question was answered correctly"
    )
    final_label: Literal["a", "b", "c", "d"] = Field(
        description="Unlearning outcome category (lowercase)"
    )
    confidence: float = Field(
        ge=0.0, le=1.0,
        description="Confidence in the final_label"
    )
    brief_reasoning: str = Field(
        description="One-sentence justification without chain-of-thought"
    )

    @field_validator("final_label", mode="before")
    @classmethod
    def normalize_label(cls, v):
        return v.lower() if isinstance(v, str) else v



def judge_row(row, sleep_time=0.5):
    try:
        response = client.responses.parse(
            model="gpt-5.1",
            temperature=0,
            input=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": make_user_prompt(
                       combined_question = row["combined_question"],
                       combined_answer = row["combined_answer"],
                       combined_generation = row["combined_generation_clean"],
                    ),
                },
            ],
            text_format=UnlearningJudgeOutput,
        )

        result: UnlearningJudgeOutput = response.output_parsed

        return pd.Series({
            "forget_answered_correctly": result.forget_answered_correctly,
            "retain_answered_correctly": result.retain_answered_correctly,
            "final_label": result.final_label,
            "confidence": result.confidence,
            "brief_reasoning": result.brief_reasoning,
        })

    except Exception as e:
        # Fail-safe: mark as damaged / unknown
        return pd.Series({
            "forget_answered_correctly": None,
            "retain_answered_correctly": None,
            "final_label": "ERROR",
            "confidence": 0.0,
            "brief_reasoning": f"Judge error: {str(e)[:200]}",
        })
    finally:
        time.sleep(sleep_time)  # rate-limit safety


def calculate_metrics(df: pd.DataFrame) -> Dict:
    """Calculate all metrics for a dataset"""
    metrics = {
        "best_case": float((df["final_label"] == "a").mean()),
        "worst_case": float((df["final_label"] == "b").mean()),
        "failed_to_unlearn": float((df["final_label"] == "c").mean()),
        "model_damaged": float((df["final_label"] == "d").mean()),
        "forget_leakage": float(df["forget_answered_correctly"].mean()),
        "retain_damage": float(1 - df["retain_answered_correctly"].mean()),
        "num_high_conf": int(len(df[df["confidence"] >= 0.8])),
        "total_samples": int(len(df)),
        "error_rate": float((df["final_label"] == "ERROR").mean()),
    }
    return metrics


def extract_generated_answer(text):
    if pd.isna(text):
        return text

    if re.search(r'user', text, flags=re.IGNORECASE):
        return re.split(r'user', text, maxsplit=1, flags=re.IGNORECASE)[1].strip()

    return text



def process_dataset(filepath: str, experiment_name: str) -> Dict:
    """Process a single dataset file"""
    print(f"\n{'='*60}")
    print(f"Processing: {experiment_name}")
    print(f"{'='*60}")
    
    # Load dataset (handle both .parquet and .csv)
    if filepath.endswith('.parquet'):
        df = pd.read_parquet(filepath)
    else:
        df = pd.read_csv(filepath)
    
    print(f"Loaded {len(df)} rows")
    
    # Check if already judged
    if "final_label" in df.columns:
        print("⚠️  Dataset already has judgments. Skipping API calls.")
        print("   (Remove 'final_label' column to re-evaluate)")
    else:
        print("Running LLM-as-a-judge evaluation...")
        df['combined_generation_clean'] = df['combined_generation'].apply(extract_generated_answer)
        judge_outputs = df.apply(judge_row, axis=1)
        df = pd.concat([df, judge_outputs], axis=1)
        
        # Save evaluated dataset
        output_dir = os.path.join(OUTPUT_BASE, Path(filepath).parent.name)
        os.makedirs(output_dir, exist_ok=True)
        
        # Save as parquet to preserve data types
        output_filename = experiment_name + '_evaluated.parquet'
        output_path = os.path.join(output_dir, output_filename)
        df.to_parquet(output_path, index=False)
        print(f"✓ Saved evaluated dataset to: {output_path}")
    
    # Calculate metrics
    metrics = calculate_metrics(df)
    
    # Print metrics
    print("\nMetrics:")
    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value}")
    
    return metrics


def find_all_datasets(base_path: str, experiments: List[str], methods: List[str]) -> List[tuple]:
    """Find all parquet files and return list of (filepath, experiment_name) tuples"""
    datasets = []
    
    for exp in experiments:
        for method in methods:
            path = Path(base_path) / exp / method
            
            if not path.exists():
                print(f"⚠️  Warning: Path does not exist: {path}")
                continue
            
            # Find all .parquet files
            parquet_files = list(path.glob('*.parquet'))
            
            for file in parquet_files:
                experiment_name = file.stem  # e.g., gd_exp_1_el2n_1
                datasets.append((str(file), experiment_name))
    
    return sorted(datasets, key=lambda x: x[1])



def main():
    """Main execution function"""
    print("="*60)
    print("LLM-as-a-Judge Batch Evaluation Pipeline")
    print("="*60)
    
    # Find all datasets
    datasets = find_all_datasets(BASE_PATH, EXPERIMENTS, METHODS)
    
    if not datasets:
        print("❌ No parquet files found in the specified paths!")
        return
    
    print(f"\n✓ Found {len(datasets)} dataset(s) to process:")
    for i, (filepath, exp_name) in enumerate(datasets, 1):
        print(f"  {i}. {exp_name}")
    
    # Process all datasets
    all_results = {}
    
    for i, (filepath, experiment_name) in enumerate(datasets, 1):
        try:
            print(f"\n[{i}/{len(datasets)}] Processing {experiment_name}...")
            
            # Process dataset
            metrics = process_dataset(filepath, experiment_name)
            
            # Store results - simple flat structure
            all_results[experiment_name] = {
                "filepath": filepath,
                **metrics
            }
            
        except Exception as e:
            print(f"❌ Error processing {experiment_name}: {str(e)}")
            all_results[experiment_name] = {
                "filepath": filepath,
                "error": str(e),
                "status": "FAILED"
            }
    
    # Save all results to JSON
    with open(RESULTS_JSON, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\n{'='*60}")
    print(f"✓ All results saved to: {RESULTS_JSON}")
    print(f"{'='*60}")
    
    # Print summary table
    print("\n" + "="*100)
    print("SUMMARY")
    print("="*100)
    print(f"{'Experiment':<40} {'Best':<8} {'Worst':<8} {'Failed':<8} {'Damaged':<8} {'F-Leak':<8} {'R-Dam':<8}")
    print("-"*100)
    
    for exp_name, results in all_results.items():
        if 'error' not in results:
            print(f"{exp_name:<40} "
                  f"{results['best_case']:<8.3f} "
                  f"{results['worst_case']:<8.3f} "
                  f"{results['failed_to_unlearn']:<8.3f} "
                  f"{results['model_damaged']:<8.3f} "
                  f"{results['forget_leakage']:<8.3f} "
                  f"{results['retain_damage']:<8.3f}")
        else:
            print(f"{exp_name:<40} {'ERROR':<8}")
    
    print("="*100)
    print(f"\n✓ Processing complete! Results: {RESULTS_JSON}")
    print(f"✓ Total datasets processed: {len(all_results)}")
    print(f"✓ Successful: {sum(1 for r in all_results.values() if 'error' not in r)}")
    print(f"✓ Failed: {sum(1 for r in all_results.values() if 'error' in r)}")


if __name__ == "__main__":
    main()