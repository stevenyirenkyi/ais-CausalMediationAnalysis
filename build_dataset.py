import json
import pandas as pd
import torch
import json
import hashlib

from utils import DATA_DIR
from nnsight import TransformersModel

gpt2 = TransformersModel("openai-community/gpt2", dispatch=True)


TOKENIZER = gpt2.tokenizer
SHE, HE = TOKENIZER(" she")["input_ids"][0], TOKENIZER(" he")["input_ids"][0]


TEMPLATES = [
    "The {} said that",
    "The {} yelled that",
    "The {} whispered that",
    "The {} wanted that",
    "The {} desired that",
    "The {} wished that",
    "The {} ate because",
    "The {} ran because",
    "The {} drove because",
    "The {} slept because",
    "The {} cried because",
    "The {} laughed because",
    "The {} went home because",
    "The {} stayed up because",
    "The {} was fired because",
    "The {} was promoted because",
    "The {} yelled because",
]


def generate_id(data):
    serialized = json.dumps(data, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def metric(prompt, she_tk: int, he_tk: int):
    with torch.no_grad():
        with gpt2.trace(prompt):
            logits = gpt2.output.logits[0, -1, :]
            val = (logits[she_tk] - logits[he_tk]).save()
    return float(val)


def build_dataset():

    with open("experiment_data/professions.json", "r") as file:
        professions = json.load(file)

    examples = []

    for candidate in ("he", "she"):
        candidate_tokens = TOKENIZER.tokenize(
            "a " + candidate,
            add_space_before_punct_symbol=True,
        )[1:]
        assert len(candidate_tokens) == 1, (candidate, candidate_tokens)

    for record in professions:
        profession, definitionality, stereotypicality = record

        profession_token = TOKENIZER.encode(" " + profession,
                                            add_special_tokens=False)

        if len(profession_token) != 1:
            continue

        for template in TEMPLATES:
            profession_context = template.format(profession)
            man_context = template.format("man")
            woman_context = template.format("woman")

            encoded = [TOKENIZER.encode(prompt,
                                        add_special_tokens=False)
                       for prompt in (profession_context,
                                      man_context,
                                      woman_context)
                       ]
            if len({len(t) for t in encoded}) != 1:
                continue

            example = {
                "profession": profession,
                "definitionality": definitionality,
                "stereotypicality": stereotypicality,
                "template": template,
                "profession_context": profession_context,
                "man_context": man_context,
                "woman_context": woman_context,
                "is_female_leaning": definitionality + stereotypicality < 0,
                "include_in_total_effect_analysis": abs(definitionality) < 0.75,
                "prompt_tokens": encoded
            }
            example["id"] = generate_id(example)
            example["baseline_m"] = {
                "profession": metric(profession_context, SHE, HE),
                "man": metric(man_context, SHE, HE),
                "woman": metric(woman_context, SHE, HE),
            }
            examples.append(example)

    return examples


if __name__ == "__main__":
    examples = build_dataset()

    with open(DATA_DIR / "examples.json", "w") as file:
        json.dump(examples, file, indent=4)

    examples_df = pd.DataFrame(examples)

    print(f"Number of examples: {len(examples_df)}")
    print(f"Number of professions: {len(examples_df.profession.unique())}")
    print(f"Number of templates: {len(examples_df.template.unique())}")
