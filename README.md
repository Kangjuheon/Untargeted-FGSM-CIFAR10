## CIFAR-10 FGSM Attack

This project fine-tunes a pretrained **ConvNeXt-Tiny** model on the **CIFAR-10** dataset and evaluates its robustness using **FGSM (Fast Gradient Sign Method)** adversarial attacks.

## Assignment Instructions

This implementation was done as part of **"Trustworthy AI" Assignment**.  
We implemented the following tasks:

- Fine-tuning a pretrained ConvNeXt-Tiny model on CIFAR-10
- Implementing an **untargeted FGSM** adversarial attack
- Measuring clean and adversarial accuracy
- Comparing performance before and after the attack

## File Descriptions

- `test.py` — Main script that fine-tunes the model and evaluates FGSM attacks
- `requirements.txt` — List of required Python packages

## How to Run

```bash
# (1) Set up virtual environment (recommended)
python -m venv venv
source venv/bin/activate        # or venv\Scripts\activate on Windows

# (2) Install dependencies
pip install -r requirements.txt

# (3) Run the main script
python test.py
```

## Model Details
- Architecture: ConvNeXt-Tiny (Pretrained on ImageNet)
- Modifications: Final classifier layer replaced for CIFAR-10 (10 classes)
- Input Size: Resized to 224×224
- Training: Only final layer trained (feature extractor frozen)

## Example Output
```bash
Epoch 1: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 782/782 [00:33<00:00, 23.63it/s, loss=0.59]
[Epoch 1] Average Loss: 0.5741
Epoch 2: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 782/782 [00:32<00:00, 24.01it/s, loss=0.231] 
[Epoch 2] Average Loss: 0.3801
Epoch 3: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 782/782 [00:32<00:00, 24.00it/s, loss=0.0177] 
[Epoch 3] Average Loss: 0.3469
Epoch 4: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 782/782 [00:32<00:00, 24.02it/s, loss=0.357] 
[Epoch 4] Average Loss: 0.3290
Epoch 5: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 782/782 [00:32<00:00, 24.08it/s, loss=0.416] 
[Epoch 5] Average Loss: 0.3200
Clean Evaluation: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 40/40 [00:10<00:00,  3.87it/s] 

[Clean Accuracy] 89.95%
FGSM Attack Evaluation: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 40/40 [00:29<00:00,  1.34it/s] 

[FGSM Untargeted Accuracy] eps=0.03 → 11.33%
```
