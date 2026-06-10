# 🧑‍🤝‍🧑 The Finetuner Studio Team

Finetuner Studio was planned and built by a virtual software team. Each role
owns one planning document and the corresponding slice of the codebase.

| Role | Owner doc | Owns in code |
|---|---|---|
| 🧭 **Product Manager** | [PRD.md](PRD.md) | Scope, personas, success metrics |
| 🏗 **Software Architect** | [ARCHITECTURE.md](ARCHITECTURE.md) | `finetuner/core/` module boundaries, mlx-tune integration contract |
| 🎨 **UX Designer** | [UX_DESIGN.md](UX_DESIGN.md) | `finetuner/ui/` tab flow, progressive disclosure, error surfaces |
| 🔬 **ML/Data Engineer** | [DATASET_DETECTION.md](DATASET_DETECTION.md) | `core/detector.py`, format heuristics, normalization rules |
| 🧪 **QA Engineer** | [TEST_PLAN.md](TEST_PLAN.md) | `tests/`, CI gates |
| 🗺 **Delivery Lead** | [ROADMAP.md](ROADMAP.md) | Milestones, release plan |

## Working agreements

1. **mlx-tune is the only training backend.** No training code is written in
   this repo; every trainer, config and export path is delegated to
   [mlx-tune](https://github.com/ARahim3/mlx-tune). The registry
   (`core/registry.py`) is the single contract between us and the library.
2. **The GUI must never be a dead end.** Anything configured visually can be
   exported as a standalone Python script (`core/codegen.py`) or a YAML recipe
   (`core/recipes.py`).
3. **Degrade gracefully.** On a machine without MLX (CI, Linux), the Studio
   still launches for dataset inspection, recipe authoring and code generation.
4. **No silent magic.** Auto-detection always shows its confidence, the
   inferred mapping, and which trainers are compatible — the user can override.
