# Changelog

## [0.3.0] - 2026-03-24

### Added
- **Training pipeline**: QLoRA fine-tuning with anticipation conditioning (HuggingFace + MLX)
- **Thunder-Train integration**: Distributed training across Mac4+Mac5 over Thunderbolt 5
- **Inscription conditioner**: 10-category behavioral motif vocabulary for training data
- **4 data formats**: ChatML messages, text, prompt/completion, annotated messages
- **5-run experiment framework**: Control + treatment + 3 ablation conditions
- **Bootstrap CI evaluation**: NLL, commitment accuracy, scalar MSE, ECE
- **Trainable anticipation modules**: InscriptionEmbedding, ScalarProjection, CommitmentGate, AnticipationHead
- **LSE reward tracker**: Exponential weight updates from holdout evaluation
- **DATA_FORMATS.md**: Complete data format specification with migration paths
- **GOTCHAS.md**: 15+ hard-won debugging lessons across MLX, HuggingFace, Vast.ai

### Changed
- Restructured Python package: `python/anticipation_geometry/` → `anticipation_geometry/`
- Added `pyproject.toml` with optional dependency groups (`[train]`, `[mlx]`, `[embeddings]`)
- Bumped version to 0.3.0

## [0.2.0] - 2026-03-22

### Added
- Knowledge graph reward evaluation (81.0% pairwise accuracy, Cohen's d = 2.23)
- High-dimensional embedding support (e5-large, 1024D)
- Evaluation harness with seeded reproducibility
- Rust crate: real-time scalar computation (<2ms at 50Hz)

## [0.1.0] - 2026-03-21

### Added
- Initial release: 7-scalar computation from vector trajectories
- Conversation convergence prediction (71.8% accuracy)
- Python library with numpy/scipy backend
- Rust workspace with 4 crates (anticipation, core-rs, protocol, types)
