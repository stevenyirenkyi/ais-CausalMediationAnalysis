import json
import pandas as pd

from utils import generate_id
from nnsight import TransformersModel

gpt2 = TransformersModel("openai-community/gpt2", dispatch=True)


TOKENIZER = gpt2.tokenizer

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
            examples.append(example)

    return examples


if __name__ == "__main__":
    examples = build_dataset()

    with open("./data/examples.json", "w") as file:
        json.dump(examples, file, indent=4)

    examples_df = pd.DataFrame(examples)

    print(f"Number of examples: {len(examples_df)}")
    print(f"Number of professions: {len(examples_df.profession.unique())}")
    print(f"Number of templates: {len(examples_df.template.unique())}")
