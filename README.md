# Untargeted FGSM Attack on CIFAR-10

This repository implements an **untargeted FGSM adversarial attack** on a ResNet18 model trained with the **CIFAR-10 dataset**.

## What it does

- Trains a ResNet18 model from scratch on CIFAR-10
- Applies **FGSM untargeted attack** to test images
- Compares **clean accuracy** and **adversarial accuracy**

## FGSM Attack Details

- Attack type: Untargeted
- Epsilon (`eps`) = 0.03
- Perturbs input in direction of gradient to cause **misclassification**

## How to Run

```bash
pip install -r requirements.txt
python test.py
```

## Example Output
```bash
Epoch 1, Loss: ...
...
[Clean Accuracy] 84.25%
[FGSM Untargeted Accuracy] eps=0.03 → 41.00%
```

## Notes
ResNet18 is trained from scratch (no pretraining)

You can adjust eps to control attack strength
