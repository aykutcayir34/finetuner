import ast

import pytest

from finetuner.core.codegen import generate_script
from finetuner.core.registry import TASKS
from finetuner.core.training import RunConfig, build_trainer_args


@pytest.mark.parametrize("task_id", list(TASKS))
def test_generated_script_is_valid_python(task_id):
    cfg = RunConfig(task=task_id, model_name=TASKS[task_id].default_model)
    script = generate_script(cfg, dataset_source="yahma/alpaca-cleaned")
    ast.parse(script)  # must be syntactically valid
    assert TASKS[task_id].trainer in script
    assert TASKS[task_id].loader in script


def test_local_dataset_block():
    cfg = RunConfig(task="sft", model_name="m")
    script = generate_script(cfg, dataset_source="/tmp/data.jsonl", dataset_is_local=True)
    assert "json.loads" in script
    ast.parse(script)


def test_trainer_args_epochs_vs_steps():
    cfg = RunConfig(num_train_epochs=3)
    args = build_trainer_args(cfg)
    assert args["num_train_epochs"] == 3 and "max_steps" not in args
    cfg2 = RunConfig(max_steps=50)
    assert build_trainer_args(cfg2)["max_steps"] == 50


def test_dpo_beta_default():
    args = build_trainer_args(RunConfig(task="dpo"))
    assert args["beta"] == 0.1
