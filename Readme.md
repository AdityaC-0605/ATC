# Cattle ATC — PyTorch Animal Type Classification & ATC Scoring

A small PyTorch-based system for cattle breed classification and automated ATC (Animal Type Classification) scoring. Key components:

- Classification model & inference: [main.py](main.py), [`CattleClassifier`](models.py)
- ATC scoring logic: [`ATCScorer`](main.py)
- System wrapper & report generation: [`CattleATCSystem`](classifier.py), [`generate_report`](classifier.py)
- Training utilities: [trainer.py](trainer.py)
- Configuration: [config.py](config.py)
- Example outputs: [outputs/test_results.json](outputs/test_results.json)

Requirements
- Python 3.8+
- PyTorch (CPU or GPU build depending on your machine)
- Common packages: numpy, opencv-python, matplotlib, scikit-learn (install as needed)

Quick start
1. Create a virtual environment and activate it:
   ```sh
   python -m venv .venv
   source .venv/bin/activate    # macOS / Linux
   .venv\Scripts\activate       # Windows
   ```
2. Install PyTorch and other dependencies (example):
   ```sh
   pip install torch torchvision numpy opencv-python matplotlib scikit-learn
   ```
3. Run the main script (inference / demo):
   ```sh
   python main.py
   ```
   - See [main.py](main.py) for available command-line options and entry points (training, evaluation, prediction).
   - The ATC scoring module is implemented in [`ATCScorer`](main.py) — it extracts body parameters and produces a scored result.

Model files
- Pretrained / checkpoint files included in repo root and outputs/: `best_atc_model.pth`, `complete_atc_model.pth`, `cattle_resnet50.pth`, plus `outputs/best_atc_model.pth`.

Outputs
- Predictions and scoring examples are stored in [outputs/test_results.json](outputs/test_results.json).
- Confusion matrix: `outputs/confusion_matrix.png`
- Grad-CAM visualizations: `outputs/gradcam/`

How ATC scoring works (high level)
- Body measurements are estimated from images inside [`ATCScorer`](main.py).
- Each parameter (body_length, height_at_withers, chest_width, rump_angle, udder_attachment, leg_structure, overall_conformation) is normalized and weighted per [config.py](config.py) scoring criteria.
- The final report can be generated with [`CattleATCSystem.generate_report`](classifier.py).

Development notes
- Freeze backbone utility is available: [`freeze_backbone`](main.py) for head-only training.
- Use [trainer.py](trainer.py) to run training loops and schedulers.
- See model definitions in [models.py](models.py).

If you want a more detailed README (usage examples, CLI flags, dependency list, or contribution guidelines) I can expand this file to