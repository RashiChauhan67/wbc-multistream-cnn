# Recorded results

Everything on this page is copied from, or derived from, the saved outputs of the original notebook, [`../notebooks/cvpr-final.ipynb`](../notebooks/cvpr-final.ipynb). **None of it was re-run** for this repository.

Protocol of the recorded run: stratified 80% / 20% split (seed 42), 4,000 training and 1,000 test images (200 per class). The test split was passed as `validation_data` during training; there were no callbacks, so the final-epoch weights were evaluated. 50 epochs, batch size 32, Adam 1e-4.

## Training and test

| Quantity | Value |
|---|---|
| Parameters | 451,333 (451,077 trainable, 256 non-trainable) |
| Epochs | 50 |
| Best validation accuracy | 0.9830 |
| Final-epoch validation accuracy | 0.9790 |
| Test accuracy (final weights) | 0.9790 |
| Test loss | 0.0660 |
| Estimated cost | 1,157,532,254 FLOPs (hand-written estimator; approximate) |

## Confusion matrix (counts)

Rows are the true class, columns the predicted class.

| | basophil | erythroblast | monocyte | myeloblast | seg_neutrophil |
|---|---|---|---|---|---|
| **basophil** | 194 | 5 | 1 | 0 | 0 |
| **erythroblast** | 1 | 197 | 2 | 0 | 0 |
| **monocyte** | 6 | 4 | 190 | 0 | 0 |
| **myeloblast** | 0 | 0 | 0 | 199 | 1 |
| **seg_neutrophil** | 0 | 0 | 1 | 0 | 199 |

979 of 1,000 correct.

## Per-class metrics (derived from the confusion matrix)

The notebook printed no classification report, so these are computed from the counts above.

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| basophil | 0.9652 | 0.9700 | 0.9676 | 200 |
| erythroblast | 0.9563 | 0.9850 | 0.9704 | 200 |
| monocyte | 0.9794 | 0.9500 | 0.9645 | 200 |
| myeloblast | 1.0000 | 0.9950 | 0.9975 | 200 |
| seg_neutrophil | 0.9950 | 0.9950 | 0.9950 | 200 |
| macro avg | 0.9792 | 0.9790 | 0.9790 | 1000 |

The ROC plot legend shows an AUC of 1.00 (two decimals) for every class.

## Input ablation (inputs zeroed at inference, no retraining)

| Inputs | Accuracy |
|---|---|
| all three | 0.9790 |
| without optical density | 0.5640 |
| without diffused | 0.9790 |
| raw only | 0.5630 |
