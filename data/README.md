# Data

No data is stored in this repository. Download the dataset yourself and check its license and terms before use.

## Dataset

The notebook read the Kaggle dataset **Blood Cell Images for Cancer Detection** (slug `blood-cell-images-for-cancer-detection`): 5,000 images, 1,000 per class.

Expected layout (the folder names are the ones used by the notebook; label order follows the list):

```
blood-cell-images-for-cancer-detection/
├── basophil/        (label 0)
├── erythroblast/    (label 1)
├── monocyte/        (label 2)
├── myeloblast/      (label 3)
└── seg_neutrophil/  (label 4)
```

## Preprocessing (as in the notebook)

1. Read with OpenCV (BGR channel order) and resize to 128 x 128. This is the **raw** view and is not rescaled.
2. **Optical density:** `OD = -log((RGB + 1) / 256)` on the RGB version of the image.
3. **Anisotropic diffusion:** Perona-Malik diffusion of the OD image, per channel (10 iterations, kappa 20, gamma 0.2).
4. Stratified 80% / 20% split with seed 42 (4,000 train / 1,000 test). The three views are split with the same indices.

## Source

The notebook records only the Kaggle dataset name. Please confirm the exact Kaggle page, the original source of the images, and the license for your copy, and cite the dataset's original publication when you publish results.
