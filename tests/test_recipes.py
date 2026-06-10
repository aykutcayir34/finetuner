from finetuner.core import recipes
from finetuner.core.training import RunConfig


def test_recipe_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setattr(recipes, "RECIPE_DIR", tmp_path)
    cfg = RunConfig(task="dpo", model_name="some/model", learning_rate=1e-5,
                    extra={"beta": 0.2})
    path = recipes.save_recipe(cfg, "my-run", "org/dataset", dataset_is_local=False)
    loaded, source, is_local = recipes.load_recipe(str(path))
    assert loaded.task == "dpo"
    assert loaded.learning_rate == 1e-5
    assert loaded.extra["beta"] == 0.2
    assert source == "org/dataset" and is_local is False
    assert str(path) in recipes.list_recipes()
