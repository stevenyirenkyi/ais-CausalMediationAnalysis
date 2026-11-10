import torch
import numpy as np
import os
import nnsight
import json
import importlib.util
from nnsight import TransformersModel
from pathlib import Path
from dataclasses import dataclass
from typing import Callable

ROOT_DIR = Path.cwd().parent
spec = importlib.util.spec_from_file_location(
    "root_utils", ROOT_DIR / "utils.py"
)
root_utils = importlib.util.module_from_spec(spec)  # type:ignore
spec.loader.exec_module(root_utils)  # type:ignore

ATTENTION_INTERVENTIONS_DIR = root_utils.ATTENTION_INTERVENTIONS_DIR
NEURON_INTERVENTIONS_DIR = root_utils.NEURON_INTERVENTIONS_DIR
EXAMPLES_PATH = root_utils.DATA_DIR / "examples.json"

CONDITIONS = [
    {
        "name": "man_direct",
        "base_key": "man_context",
        "source_key": "profession_context",
    },
    {
        "name": "man_indirect",
        "base_key": "profession_context",
        "source_key": "man_context",
    },
    {
        "name": "woman_direct",
        "base_key": "woman_context",
        "source_key": "profession_context",
    },
    {
        "name": "woman_indirect",
        "base_key": "profession_context",
        "source_key": "woman_context",
    },
]


class Config:
    def __init__(self, model: TransformersModel) -> None:
        self.model = model
        self.model.eval()
        self.device = torch.device("cuda" if torch.cuda.is_available()
                                   else "cpu")
        print(f"Device: {self.device}")

        self.n_layers = model.config.n_layer
        self.n_heads = model.config.n_head
        self.n_embed = model.config.n_embd
        self.he_tk = model.tokenizer(" he")["input_ids"][0]
        self.she_tk = model.tokenizer(" she")["input_ids"][0]

        # ------ FOR ONLY NEURON INTERVENTIONS ------
        self.batch_size = 768
        wte = [self.model.transformer.wte]
        mlps = [self.model.transformer.h[i].mlp
                for i in range(self.n_layers)]
        self.modules = wte + mlps  # used only for neuron interventions


class ProcessedIDs:
    def __init__(self,
                 condition: str,
                 examples: list[dict],
                 output_dir: Path,
                 effect_shape: tuple):
        self.id_path = output_dir / f"{condition}.txt"
        self.effect_path = output_dir / f"{condition}.npy"

        self.row_indices = {
            str(example["id"]): i
            for i, example in enumerate(examples)
        }

        self.effects_shape = (len(examples), *effect_shape)

        self.load_processed_ids()
        self.load_effects()

    def load_effects(self):
        if self.effect_path.exists():
            self.effects = np.load(self.effect_path, mmap_mode="r+")

            if self.effects.shape != self.effects_shape:
                raise ValueError(
                    f"Existing array has shape {self.effects.shape}; "
                    f"expected {self.effects_shape}."
                )
        else:
            self.effects = np.lib.format.open_memmap(self.effect_path,
                                                     mode="w+",
                                                     dtype=np.float32,
                                                     shape=self.effects_shape)

    def load_processed_ids(self):
        if self.id_path.exists():
            self.processed_ids = set(
                self.id_path.read_text().splitlines()
            )
            print(f"Loaded {len(self.processed_ids)} processed IDs.")
        else:
            self.processed_ids = set()

    def save(self, example_id, effects):
        example_id = str(example_id)

        row = self.row_indices[example_id]

        # Write effects directly to disk
        self.effects[row] = np.asarray(effects, dtype=np.float32)
        self.effects.flush()

        # Record the ID only after the results are written
        with open(self.id_path, "a") as f:
            f.write(example_id + "\n")
            f.flush()
            os.fsync(f.fileno())

        self.processed_ids.add(example_id)

    def contains(self, example_id):
        return str(example_id) in self.processed_ids

    def __len__(self):
        return len(self.processed_ids)


class SourceActivations:
    def __init__(self) -> None:
        self.cache = {}

    def __call__(
        self,
        cfg: Config,
        source_prompt: str,
        modules: list,
        position: int = 1,
    ):
        cache_key = (source_prompt, position)

        if cache_key in self.cache:
            return self.cache[cache_key]

        with torch.no_grad(), cfg.model.trace(source_prompt):
            saved_acts = nnsight.save([
                module.output[0, position, :].save()
                for module in modules
            ])

        activations = [
            activation.detach().cpu().clone()
            for activation in saved_acts
        ]

        self.cache[cache_key] = activations
        return activations


def run_one_condition(cfg: Config,
                      examples: list[dict],
                      condition: dict,
                      patch_fn: Callable,
                      source_activations: SourceActivations,
                      output_dir: Path,
                      effect_shape: tuple):

    processed_ids = ProcessedIDs(condition["name"],
                                 examples,
                                 output_dir,
                                 effect_shape)

    print(f"{condition['name']}: {len(processed_ids)} already saved; "
          f"{len(examples) - len(processed_ids)} remaining")

    for i, example in enumerate(examples):
        if processed_ids.contains(example["id"]):
            continue

        base = example[condition["base_key"]]
        source = example[condition["source_key"]]

        effects = patch_fn(cfg=cfg,
                           base=base,
                           source=source,
                           source_activations=source_activations)

        processed_ids.save(example["id"], effects)

        if (i + 1) % 30 == 0:
            print(
                f"{condition['name']}: saved examples "
                f"{i + 1}/{len(examples)}"
            )


@dataclass(frozen=True)
class PatchTarget:
    patch_fn: Callable
    output_dir: Path
    effect_shape: tuple


def run_experiment(examples_filepath: Path, pt: PatchTarget):
    gpt2 = TransformersModel("openai-community/gpt2",
                             dispatch=True,
                             attn_implementation="eager")
    source_activations = SourceActivations()
    cfg = Config(gpt2)

    with open(examples_filepath, "r") as f:
        examples = json.load(f)

    for condition in CONDITIONS:
        run_one_condition(cfg=cfg,
                          examples=examples,
                          condition=condition,
                          patch_fn=pt.patch_fn,
                          source_activations=source_activations,
                          output_dir=pt.output_dir,
                          effect_shape=pt.effect_shape)


def test_fn(patch_fn: Callable):
    gpt2 = TransformersModel("openai-community/gpt2",
                             dispatch=True,
                             attn_implementation="eager",)
    cfg = Config(gpt2)

    base = "The nurse said that"
    source = "The man said that"

    source_activations = SourceActivations()

    results = patch_fn(cfg, base, source, source_activations)

    print(results.shape)
    print(results)
