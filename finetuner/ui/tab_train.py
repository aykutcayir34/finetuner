"""Train tab: hyperparameters, recipes, the code generator, and the launch button."""

from __future__ import annotations

import gradio as gr

from ..core import recipes
from ..core.codegen import generate_script
from ..core.detector import normalize
from ..core.jobs import MANAGER
from ..core.registry import get_task, mlx_available
from ..core.state import STATE
from ..core.training import run_training


def _collect(cfg_fields: dict) -> None:
    cfg = STATE.config
    cfg.output_dir = cfg_fields["output_dir"]
    cfg.batch_size = int(cfg_fields["batch_size"])
    cfg.gradient_accumulation_steps = int(cfg_fields["grad_accum"])
    cfg.learning_rate = float(cfg_fields["lr"])
    cfg.max_steps = int(cfg_fields["max_steps"])
    cfg.num_train_epochs = float(cfg_fields["epochs"]) or None
    cfg.warmup_steps = int(cfg_fields["warmup"])
    cfg.gradient_checkpointing = bool(cfg_fields["grad_ckpt"])
    cfg.seed = int(cfg_fields["seed"])
    extra = {}
    if cfg.task in ("dpo", "orpo") and cfg_fields["beta"]:
        extra["beta"] = float(cfg_fields["beta"])
    cfg.extra = extra


def build(app):
    with gr.Tab("🚀 Train", id="train"):
        gr.Markdown("### Hyperparameters")
        with gr.Row():
            output_dir = gr.Textbox(value="outputs", label="Output directory")
            batch_size = gr.Slider(1, 32, value=2, step=1, label="Batch size")
            grad_accum = gr.Slider(1, 64, value=1, step=1, label="Gradient accumulation")
        with gr.Row():
            lr = gr.Number(value=2e-4, label="Learning rate")
            max_steps = gr.Slider(10, 10000, value=100, step=10, label="Max steps")
            epochs = gr.Number(value=0, label="Epochs (0 → use max steps)")
        with gr.Row():
            warmup = gr.Slider(0, 500, value=5, step=5, label="Warmup steps")
            seed = gr.Number(value=42, precision=0, label="Seed")
            grad_ckpt = gr.Checkbox(value=False, label="Gradient checkpointing (saves memory)")
            beta = gr.Number(value=0.1, label="β (DPO/ORPO only)")

        with gr.Row():
            start_btn = gr.Button("🏁 Start training", variant="primary", scale=2)
            gen_btn = gr.Button("🧾 Generate Python script", scale=1)
        status = gr.Markdown()

        with gr.Accordion("Generated script (standalone mlx-tune code)", open=False):
            script_out = gr.Code(language="python", label="train.py")
            gr.Markdown("Copy this script anywhere — it reproduces this run without the GUI.")

        with gr.Accordion("Recipes (save / load runs as YAML)", open=False):
            with gr.Row():
                recipe_name = gr.Textbox(label="Recipe name", placeholder="my-sft-run")
                save_recipe_btn = gr.Button("💾 Save recipe")
            with gr.Row():
                recipe_pick = gr.Dropdown(label="Saved recipes", choices=recipes.list_recipes(),
                                          allow_custom_value=True)
                load_recipe_btn = gr.Button("📂 Load recipe")
            recipe_status = gr.Markdown()

        hp_inputs = [output_dir, batch_size, grad_accum, lr, max_steps, epochs,
                     warmup, seed, grad_ckpt, beta]

        def _fields(*vals) -> dict:
            keys = ["output_dir", "batch_size", "grad_accum", "lr", "max_steps",
                    "epochs", "warmup", "seed", "grad_ckpt", "beta"]
            return dict(zip(keys, vals))

        # ----- start training -------------------------------------------------
        def on_start(*vals):
            _collect(_fields(*vals))
            cfg = STATE.config
            ok, reason = mlx_available()
            if not ok:
                return f"❌ {reason}"
            if STATE.model is None:
                return "❌ Load a model first (🧠 Model tab)."
            if not STATE.raw_rows:
                return "❌ Load a dataset first (📚 Dataset tab)."
            if STATE.model_loaded_for_task != cfg.task:
                cfg.task = STATE.model_loaded_for_task
            try:
                dataset = normalize(STATE.raw_rows, STATE.detection, cfg.task, STATE.tokenizer)
            except Exception as exc:
                return (f"❌ Dataset incompatible with **{get_task(cfg.task).label}**: {exc}\n\n"
                        f"Detected format: {STATE.detection.label if STATE.detection else '—'}")
            job = MANAGER.submit(f"{cfg.task} · {cfg.model_name}", run_training,
                                 cfg, STATE.model, STATE.tokenizer, dataset)
            return (f"🏃 **Job #{job.id}** started ({len(dataset)} samples). "
                    "Follow it in the **📈 Monitor** tab.")

        start_btn.click(on_start, hp_inputs, status)

        # ----- codegen --------------------------------------------------------
        def on_generate(*vals):
            _collect(_fields(*vals))
            cfg = STATE.config
            if STATE.model_loaded_for_task:
                cfg.task = STATE.model_loaded_for_task
            if not cfg.model_name:
                cfg.model_name = get_task(cfg.task).default_model
            return generate_script(cfg, STATE.dataset_source, STATE.dataset_is_local)

        gen_btn.click(on_generate, hp_inputs, script_out)

        # ----- recipes ---------------------------------------------------------
        def on_save_recipe(name, *vals):
            _collect(_fields(*vals))
            path = recipes.save_recipe(STATE.config, name, STATE.dataset_source,
                                       STATE.dataset_is_local)
            return f"💾 Saved `{path}`", gr.update(choices=recipes.list_recipes())

        save_recipe_btn.click(on_save_recipe, [recipe_name, *hp_inputs],
                              [recipe_status, recipe_pick])

        def on_load_recipe(path):
            if not path:
                return ["❌ Pick a recipe."] + [gr.update()] * len(hp_inputs)
            try:
                cfg, src, is_local = recipes.load_recipe(path)
            except Exception as exc:
                return [f"❌ {exc}"] + [gr.update()] * len(hp_inputs)
            STATE.config = cfg
            STATE.dataset_source, STATE.dataset_is_local = src, is_local
            return [f"📂 Loaded `{path}` — task **{cfg.task}**, model `{cfg.model_name}`. "
                    "Reload model/dataset to run it.",
                    cfg.output_dir, cfg.batch_size, cfg.gradient_accumulation_steps,
                    cfg.learning_rate, cfg.max_steps, cfg.num_train_epochs or 0,
                    cfg.warmup_steps, cfg.seed, cfg.gradient_checkpointing,
                    cfg.extra.get("beta", 0.1)]

        load_recipe_btn.click(on_load_recipe, recipe_pick, [recipe_status, *hp_inputs])
