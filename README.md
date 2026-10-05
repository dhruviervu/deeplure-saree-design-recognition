# Color-invariant saree design retrieval

Open `saree_design_recognition.ipynb`, set the runtime to a GPU, and run it from the top. The notebook and both image sets are in this repository:

- `data/archive` — Kaggle / Roboflow export (`train`, `valid`, `test`)
- `data/handloom` — unlabeled handloom images

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/dhruviervu/deeplure-saree-design-recognition/blob/main/saree_design_recognition.ipynb)

On Colab the notebook file arrives without the images. The path cell clones this repository when `data/archive/train` is not already in the working directory, then reads those folders. No Drive paths are required.
