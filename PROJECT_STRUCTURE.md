# Project structure

## Source code

- models/: model definitions. U-Net files stay here; SegFormer will be added here later.
- utils/: shared dataset loading and binary Dice/IoU metrics.
- losses/: loss implementations used by training.
- scripts/: canonical command-line entry points for training, evaluation, and visualization.
- configs/: experiment settings for each model.
- notebooks/: interactive notebooks.

## Data and experiments

- data/raw/Kvasir-SEG/: original Kvasir-SEG images and masks.
- data/samples/: optional synthetic/sample data.
- experiments/unet_scratch/: Scratch model checkpoints and history.
- experiments/unet_resnet34/: pretrained ResNet34 model checkpoints and history.
- experiments/comparison/results/: comparison tables and figures.

Large data, model weights, generated results, virtual environments, and W&B logs are ignored by Git.

## Commands

Run commands from the project root:

python scripts/train.py --model scratch --data_dir data/raw/Kvasir-SEG
python scripts/train.py --model pretrained --backbone resnet34 --data_dir data/raw/Kvasir-SEG
python scripts/evaluate.py
python scripts/visualize.py --num_samples 8

The root-level train.py, evaluate.py, and visualize.py files are compatibility wrappers. New code should use the scripts in scripts/.
