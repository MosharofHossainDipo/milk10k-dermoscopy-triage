# Who Needs Dermoscopy? Missing-modality-robust skin lesion triage on MILK10k

Code for the paper *"Who Needs Dermoscopy? Missing-Modality-Robust Skin Lesion Triage with Skin-Tone-Aware Conformal Guarantees and Validated Explanations"* (Mosharof Hossain Dipo et al., American International University-Bangladesh, manuscript in preparation).

The pipeline trains one model that accepts any combination of a clinical close-up photograph, a dermoscopic image and patient metadata, decides which patients need dermoscopy, and bounds the share of treatable lesions that are wrongly reassured within each skin tone group. Every explanation method is tested before it is used.

## Data

The notebooks use the MILK10k training release (5,240 lesions, 10,480 images) and the MILK10k benchmark (479 lesions, labels hidden), both from the ISIC Archive: https://challenge.isic-archive.com/data/#milk10k. The data are licensed CC BY-NC by their owners and are **not** included here. Download them from ISIC and upload them to Kaggle as a dataset (the notebooks call it `MILK10_DATASET`).

`results/milk10k_lesion_folds.csv` lists the fold of every training lesion (lesion and image IDs only, no labels), so the exact five lesion-level folds can be reproduced. MILK10k has no patient identifier, so splits are made by lesion; both images of a lesion are always in the same fold.

## Notebooks

All notebooks were run on Kaggle with one NVIDIA T4 GPU. Each one finds its inputs by walking `/kaggle/input`, so attach the listed datasets and earlier notebook outputs with **Add Input**, and run with **Save & Run All (Commit)** so the outputs are kept.

| Notebook | What it does | Inputs | Paper |
|---|---|---|---|
| `01_milk10k_audit_and_folds` | Lesion table, data audit, five folds stratified by diagnosis and skin tone | MILK10k | Sec. 3.2–3.3 |
| `02_unimodal_baselines_and_embeddings` | Development encoders (v1): one ConvNeXtV2-Tiny per image type and fold; saves embeddings | MILK10k, 01 | Sec. 3.4 (v1) |
| `03_fusion_missing_modality` | Late fusion with modality dropout vs dedicated models and cross-attention | 02 | Sec. 3.5 |
| `04_dermoscopy_acquisition_policy`, `04b_calibrated_policies` | Selective dermoscopy policies, calibrated | 03 | Sec. 3.7 |
| `05_conformal_triage`, `05b_melanoma_tone_safe_triage` | Two-stage binary conformal triage; group, per-tone and per-diagnosis calibration | 03 | Sec. 3.6 |
| `06a_concept_xai`, `06b_gradcam_faithfulness` | Concept tests, shortcut audit, Grad-CAM faithfulness (development run) | 02, 03 | Sec. 3.8 |
| `07_training_v2` | Final encoders (v2): 384 px dermoscopy, Shades of Gray, layer-wise LR decay, EMA, TTA. Set `TRAIN_SEED` to 42, 43, 44 and commit once per seed | MILK10k | Sec. 3.4 |
| `08_seed_pipeline` | Fusion, calibration, policies and triage for every seed; mean ± SD tables | 07 (all seeds) | Sec. 4 |
| `09_strong_xai` | Exact Shapley over inputs, Grad-CAM / HiResCAM / IG / SmoothGrad with deletion, insertion and randomisation tests, TCAV, case retrieval | 07 (seed 42), 08, MILK10k | Sec. 3.8 |
| `10_deployment_benchmark` | MONET ablation, ONNX FP32/INT8 export, fidelity, latency, time per patient | 07 (seed 42), 08, MILK10k | Sec. 3.9 |
| `11_milk10k_benchmark_submission` | Scores the 479 benchmark lesions with all 15 v2 models; per-class thresholds by nested CV; writes the ISIC submission | 07 (all seeds), MILK10k, benchmark | Sec. 3.10 |
| `12_external_baselines` | PanDerm linear probe, PanDerm + our fusion, concatenation MLP, gated fusion, on the same folds; paired bootstrap | 07 (seed 42), MILK10k | Sec. 3.10 |

Notebook 12 downloads the PanDerm-Large weights from the authors' Google Drive link (https://github.com/SiyuanYan1/PanDerm). If the download is refused, upload `panderm_ll_data6_checkpoint-499.pth` as a Kaggle dataset and attach it.

## Results included here

`results/external_baselines/` holds the tables of notebook 12 (out-of-fold, seed 42). `docs/MILK10k_approach_description.pdf` is the approach description submitted with the hidden benchmark entry (overall macro F1 / Dice 0.464, leaderboard rank 130 of 200, no external data).

## Requirements

See `requirements.txt`. Kaggle's default GPU image has everything except `onnx`, `onnxruntime`, `gdown` and `open_clip_torch`, which notebooks 10 and 12 install themselves.

## Citation

If you use this code, please cite the paper (citation will be added on publication) and the MILK10k dataset:

Tschandl P, Akay BN, Rosendahl C, et al. MILK10k: a hierarchical multimodal imaging-learning toolkit for diagnosing pigmented and nonpigmented skin cancer and its simulators. J Invest Dermatol. 2026;146(2):357.

## License

Code: MIT (see `LICENSE`). The MILK10k images and metadata keep their own CC BY-NC license and are not redistributed.
