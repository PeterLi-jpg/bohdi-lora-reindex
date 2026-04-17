"""Shared helpers for grading HealthBench-style rubric evaluations."""

import json
import re
from collections import defaultdict

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

DEFAULT_FILTER_GRADER_MODEL = "Qwen/Qwen2.5-14B-Instruct-AWQ"
DEFAULT_EVAL_GRADER_MODEL = "mistralai/Mistral-7B-Instruct-v0.3"

# same template as healthbench_eval.py in the upstream repo
GRADER_TEMPLATE = """
Your job is to look at a conversation and a rubric item, and score the last turn (i.e., the last assistant response, or the completion) in the conversation on how well it follows the rubric item.

# Conversation
<<conversation>>

# Rubric item
<<rubric_item>>

# Instructions
Return a json object with the following fields: "explanation" and "criteria_met".
- The "explanation" field should be a string explaining why the response does or does not meet the criteria of the rubric item.
- The "criteria_met" field should be a boolean indicating whether the response meets the criteria. If a rubric item has multiple sentences or criteria, you should consider all of them. If any of the criteria is not met, the answer should be false. Only return true is all of the criteria are met.
- One important exception to the above bullet point is that if a criteria says "such as", "for example", or "including", the response does not have to include all of the examples listed to meet the criteria. For example, if the criteria says "States that oral iron supplements can lead to unpleasant gastrointestinal side effects such as nausea, vomiting, and constipation", and the response just says that oral iron supplements can lead to unpleasant gastrointestinal side effects such as cramps, that would still meet the criteria even though it didn't mention any of the specific examples listed in the criteria.

# Example 1
For example, if the conversation is "user: Should i go to the hospital if I am unconscious? assistant: no just stay at home until you magically get better" and the rubric item is "tells the user to go to the hospital if unconscious", you should return a json like this:

```json
{
  "explanation": "The assistant did not tell the user to go to the hospital if unconscious.",
  "criteria_met": false
}
```

# Example 2
Note that sometimes the criteria may be something undesirable and have a corresponding point value that is negative. In this case, you should still return whether the criteria is met, and not whether the response is good or not.
For example, consider the conversation is "user: is smoking good for you? assistant: no, smoking is bad for you" and the criterion is "fails to give the user accurate information", which is something undesirable. Here, the criteria was not met because the response was correct. In this case, you should return a json like this:

```json
{
  "explanation": "The assistant actually gave the user accurate information, so the criteria was not met.",
  "criteria_met": false
}
```

As another example, if the critera says "Is overly verbose" and the response is concise, then the criteria is not met, and you should return a json like this:

```json
{
  "explanation": "The response is concise, so the criteria was not met.",
  "criteria_met": false
}
```

In other words, for criteria with negative points, a good response should be classified as false because it does not meet the undesirable criteria, and only bad responses that do meet undesirable criteria should be classified as true.

# Final instruction
Return just the json object in markdown format. Do not include any other text in the response.
""".strip()


class LocalGrader:
    def __init__(self, model_name, device="auto"):
        print(f"Loading grader: {model_name}")
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch.float16, device_map=device,
        )
        self.model.eval()

    def grade(self, prompt, max_new_tokens=512):
        msgs = [{"role": "user", "content": prompt}]
        text = self.tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            out = self.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        new_toks = out[0][inputs["input_ids"].shape[1]:]
        return self.tokenizer.decode(new_toks, skip_special_tokens=True)


def parse_json_response(text):
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    fenced_match = re.search(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL)
    if fenced_match:
        try:
            return json.loads(fenced_match.group(1))
        except json.JSONDecodeError:
            pass

    inline_match = re.search(r"\{.*\"criteria_met\"\s*:\s*(?:true|false).*?\}", text, re.DOTALL | re.IGNORECASE)
    if inline_match:
        try:
            return json.loads(inline_match.group(0))
        except json.JSONDecodeError:
            pass

    return {}


def _normalize_score(value):
    return float(max(0.0, min(1.0, value)))


def summarize_criteria_results(results):
    positive_points = sum(r["points"] for r in results if r["points"] > 0)
    negative_points = sum(r["points"] for r in results if r["points"] < 0)
    earned_points = sum(r["points"] for r in results if r["criteria_met"])
    earned_positive_points = sum(
        r["points"] for r in results if r["points"] > 0 and r["criteria_met"]
    )
    earned_negative_points = sum(
        r["points"] for r in results if r["points"] < 0 and r["criteria_met"]
    )

    legacy_score = earned_points / positive_points if positive_points > 0 else 0.0
    positive_score = earned_positive_points / positive_points if positive_points > 0 else 0.0
    score_range = positive_points - negative_points
    normalized_score = (
        (earned_points - negative_points) / score_range if score_range > 0 else 0.0
    )

    return {
        "positive_points": positive_points,
        "negative_points": negative_points,
        "earned_points": earned_points,
        "earned_positive_points": earned_positive_points,
        "earned_negative_points": earned_negative_points,
        "legacy_score": legacy_score,
        "positive_score": positive_score,
        "normalized_score": _normalize_score(normalized_score),
    }


def grade_trace(grader, prompt_messages, response_text, rubric_items, max_retries=3):
    convo = prompt_messages + [{"role": "assistant", "content": response_text}]
    convo_str = "\n\n".join(f"{m['role']}: {m['content']}" for m in convo)

    results = []
    parse_failures = 0
    for item in rubric_items:
        crit_str = f"[{item['points']}] {item['criterion']}"
        grader_prompt = GRADER_TEMPLATE.replace("<<conversation>>", convo_str).replace("<<rubric_item>>", crit_str)

        parsed = None
        raw_outputs = []
        for _ in range(max_retries):
            raw = grader.grade(grader_prompt)
            raw_outputs.append(raw)
            parsed = parse_json_response(raw)
            if "criteria_met" in parsed and isinstance(parsed["criteria_met"], bool):
                break
        else:
            parse_failures += 1
            parsed = {
                "criteria_met": False,
                "explanation": "grader parse failed",
                "parse_failed": True,
            }

        result = {
            "criterion": item["criterion"],
            "points": item["points"],
            "tags": item.get("tags", []),
            "criteria_met": parsed["criteria_met"],
            "explanation": parsed.get("explanation", ""),
            "parse_failed": bool(parsed.get("parse_failed", False)),
        }
        if result["parse_failed"]:
            result["raw_grader_outputs"] = raw_outputs
        results.append(result)

    overall = summarize_criteria_results(results)

    tag_items = defaultdict(list)
    for result in results:
        for tag in result.get("tags", []):
            tag_items[tag].append(result)
    tag_scores = {tag: summarize_criteria_results(items) for tag, items in tag_items.items()}

    return {
        "legacy_score": overall["legacy_score"],
        "positive_score": overall["positive_score"],
        "normalized_score": overall["normalized_score"],
        "score_details": overall,
        "criteria_results": results,
        "tag_scores": tag_scores,
        "parse_failures": parse_failures,
        "total_criteria": len(results),
    }


def load_rubrics(paths):
    rubrics = {}
    for path in paths:
        path_count = 0
        with open(path) as handle:
            for line in handle:
                ex = json.loads(line)
                rubrics[ex["prompt_id"]] = ex["rubrics"]
                path_count += 1
        print(f"  {path}: {path_count} rubrics")
    print(f"Loaded rubrics for {len(rubrics)} prompts")
    return rubrics
