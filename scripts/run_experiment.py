"""STAGE 5: run the pipeline over the dataset under one config; compare runs in the UI.
  python -m scripts.run_experiment --chunk-size 300
  python -m scripts.run_experiment --chunk-size 800 --prompt-label staging
"""
import argparse

from app.config import DATASET_NAME, langfuse
from app.evals import correctness, judge_faithfulness
from app.rag import ask

p = argparse.ArgumentParser()
p.add_argument("--chunk-size", type=int, default=500)
p.add_argument("--prompt-label", default="production")
p.add_argument("--run-name")
args = p.parse_args()

run_name = args.run_name or f"chunk{args.chunk_size}-{args.prompt_label}"
dataset = langfuse.get_dataset(DATASET_NAME)
totals = {"correctness": 0.0, "faithfulness": 0.0}

for item in dataset.items:
    q = item.input["question"]
    # Everything inside this block is traced and linked to the dataset item + run
    with item.run(run_name=run_name, run_metadata={"chunk_size": args.chunk_size,
                                                   "prompt_label": args.prompt_label}) as span:
        result = ask(q, user_id="experiment", session_id=run_name,
                     chunk_size=args.chunk_size, prompt_label=args.prompt_label)
        span.update_trace(input=item.input, output=result["answer"])
        c = correctness(result["answer"], item.expected_output)
        f = judge_faithfulness(q, result["answer"], result["contexts"])
        span.score_trace(name="correctness", value=c)
        span.score_trace(name="faithfulness", value=f["score"], comment=f["reason"])
        totals["correctness"] += c
        totals["faithfulness"] += f["score"]

n = len(dataset.items)
langfuse.flush()
print(f"{run_name}: correctness={totals['correctness']/n:.2f} faithfulness={totals['faithfulness']/n:.2f} (n={n})")
