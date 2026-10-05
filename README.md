# Color-invariant saree design retrieval

Zero-shot motif retrieval with a frozen [DINOv2](https://github.com/facebookresearch/dinov2) ViT-B/14 backbone (`timm` name `vit_base_patch14_dinov2`, the `dinov2_vitb14` checkpoint). Nothing is trained. Weave folder names are not class ids, and ArcFace is not used.

Each image is embedded three times: RGB, grayscale, and Canny edges. The three L2-normalized 768-d vectors are averaged and re-normalized. Gallery search uses cosine similarity and [k-reciprocal re-ranking](https://openaccess.thecvf.com/content_cvpr_2017/html/Zhong_Re-Ranking_Person_Re-Identification_CVPR_2017_paper.html).

Positives are the Roboflow `.rf.` copies of one source photo. The notebook reports Rank-1, mAP, a held-out same-photo verification threshold, and a hue-shift probe that recolors the query before matching.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/dhruviervu/deeplure-saree-design-recognition/blob/main/saree_design_recognition.ipynb)

## Run on Colab

1. Runtime, Change runtime type, GPU.
2. Open `saree_design_recognition.ipynb`.
3. Put the Kaggle export where the notebook can see a `train` folder, and optionally the unlabeled handloom images. Either set `ROBOFLOW_ROOT` and `HANDLOOM_ROOT`, or place the folders at `/content/drive/MyDrive/DeepLure/archive` and `/content/drive/MyDrive/DeepLure/handloom_sarees` and uncomment the Drive mount cell.

Do not reinstall PyTorch on Colab. The GPU runtime already includes a CUDA build. The setup cell stops if CUDA is missing.

Images are not in this repository. The handloom files are local-only and must not be redistributed.

## Layout

- `saree_design_recognition.ipynb` — end-to-end extraction, evaluation, plots, and `match_topk`
- `deeplure_core.py` — grouping, Rank-1, mAP, and k-reciprocal re-ranking
- `test_retrieval.py` — checks for those metrics
