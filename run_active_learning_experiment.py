"""
Active learning validation experiment (design doc Sec. 4): does MC-Dropout /
BALD uncertainty selection actually beat random selection at an equal
labelling budget?

This directly simulates the design doc's "~200 vehicle labelling budget"
scenario, scaled to CarDD's ~2800 train images: start from a small labelled
SEED set, then compare two ways of spending an equal additional labelling
budget from the unlabelled POOL:

  A) random selection  (today's default if nobody thinks about it)
  B) active selection   (rank the pool by MCDropoutYOLO uncertainty, take
                          the most-uncertain images - Sec. 4's "Select
                          high-uncertainty vehicles (MC Dropout)")

Both arms train from the same yolo11n.pt starting point, same hyperparameters,
same seed+budget size - the ONLY difference is which additional images were
picked. Both are evaluated on the untouched official CarDD test split, which
neither arm's selection process ever sees. If active beats random by a
meaningful margin, the extra complexity of running MC-Dropout in production is
justified; if not, that's an honest, useful negative result and says to just
label randomly.

Usage:
    source .venv/bin/activate
    python run_active_learning_experiment.py --seed-frac 0.15 --budget-frac 0.15 --epochs 40
"""

import argparse
import random
from pathlib import Path

from ultralytics import YOLO

DATA_ROOT = Path("data")
SPLITS_DIR = Path("al_splits")
DATA_YAML_TEMPLATE = """\
path: {root}
train: {train_list}
val: val/images
test: test/images

nc: 6
names: ["dent", "scratch", "crack", "glass_shatter", "lamp_broken", "tire_flat"]
"""


def write_image_list(paths: list, name: str) -> Path:
    SPLITS_DIR.mkdir(exist_ok=True)
    out = SPLITS_DIR / name
    out.write_text("\n".join(str(p) for p in paths) + "\n")
    return out


def write_data_yaml(train_list_path: Path, name: str) -> Path:
    SPLITS_DIR.mkdir(exist_ok=True)
    out = SPLITS_DIR / name
    out.write_text(DATA_YAML_TEMPLATE.format(root=DATA_ROOT, train_list=train_list_path.resolve()))
    return out


def train_and_eval(data_yaml: Path, run_name: str, epochs: int, device: str = "0"):
    model = YOLO("yolo11n.pt")
    model.train(
        data=str(data_yaml),
        epochs=epochs,
        imgsz=640,
        batch=16,
        device=device,
        name=run_name,
        patience=max(10, epochs // 3),
        seed=0,
        verbose=False,
    )
    # Ultralytics resolves a relative `project=` as RUNS_DIR/task/project, so passing
    # "runs/detect" as project doubles it to runs/detect/runs/detect/<name> - leave
    # project unset (default "") to get the plain runs/detect/<name>, and read the
    # actual save_dir back from the trainer instead of re-deriving the path by hand.
    save_dir = model.trainer.save_dir
    metrics = model.val(data=str(data_yaml), split="test", device=device, name=f"{run_name}_test", verbose=False)
    best_weights = str(save_dir / "weights" / "best.pt")
    return {
        "run_name": run_name,
        "weights": best_weights,
        "mAP50": float(metrics.box.map50),
        "mAP50_95": float(metrics.box.map),
        "per_class_ap50": {name: float(metrics.box.ap50[i]) for i, name in model.names.items()},
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed-frac", type=float, default=0.15, help="fraction of train images in the initial labelled seed set")
    parser.add_argument("--budget-frac", type=float, default=0.15, help="fraction of train images added per round (active vs random)")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--mc-passes", type=int, default=8)
    parser.add_argument("--device", type=str, default="0")
    parser.add_argument("--rng-seed", type=int, default=0)
    args = parser.parse_args()

    all_train_images = sorted((DATA_ROOT / "train" / "images").glob("*.jpg"))
    rng = random.Random(args.rng_seed)
    shuffled = all_train_images[:]
    rng.shuffle(shuffled)

    n_total = len(shuffled)
    n_seed = int(n_total * args.seed_frac)
    n_budget = int(n_total * args.budget_frac)

    seed_images = shuffled[:n_seed]
    pool_images = shuffled[n_seed:]

    print(f"Total train images: {n_total} | seed: {n_seed} | pool: {len(pool_images)} | budget/round: {n_budget}")

    # --- Stage 1: train on seed only ---
    seed_list = write_image_list(seed_images, "seed.txt")
    seed_yaml = write_data_yaml(seed_list, "seed.yaml")
    print("\n=== Training on SEED only ===")
    seed_result = train_and_eval(seed_yaml, "al_seed", epochs=args.epochs, device=args.device)
    print(seed_result)

    # --- Stage 2: score the pool with the seed model's MC-Dropout uncertainty ---
    print("\n=== Scoring pool with MC-Dropout/BALD (using seed model) ===")
    from active_learning.mc_dropout import MCDropoutYOLO

    scorer = MCDropoutYOLO(seed_result["weights"], n_passes=args.mc_passes, device=f"cuda:{args.device}" if args.device.isdigit() else args.device)
    ranked = scorer.rank_pool([str(p) for p in pool_images], top_k=20)
    active_selected = [Path(r["image"]) for r in ranked[:n_budget]]
    del scorer  # free GPU memory before the next training run

    # --- Stage 3: random selection of equal size from the SAME pool ---
    random_selected = rng.sample(pool_images, n_budget)

    # --- Stage 4: train seed+active and seed+random, same epochs/hparams ---
    active_list = write_image_list(seed_images + active_selected, "seed_plus_active.txt")
    active_yaml = write_data_yaml(active_list, "seed_plus_active.yaml")
    print("\n=== Training on SEED + ACTIVE-selected ===")
    active_result = train_and_eval(active_yaml, "al_seed_plus_active", epochs=args.epochs, device=args.device)
    print(active_result)

    random_list = write_image_list(seed_images + random_selected, "seed_plus_random.txt")
    random_yaml = write_data_yaml(random_list, "seed_plus_random.yaml")
    print("\n=== Training on SEED + RANDOM-selected ===")
    random_result = train_and_eval(random_yaml, "al_seed_plus_random", epochs=args.epochs, device=args.device)
    print(random_result)

    # --- Report ---
    print("\n" + "=" * 60)
    print("ACTIVE LEARNING VALIDATION — TEST SET RESULTS")
    print("=" * 60)
    print(f"{'Arm':<20}{'n_train':<10}{'mAP50':<10}{'mAP50-95':<10}")
    print(f"{'seed only':<20}{n_seed:<10}{seed_result['mAP50']:<10.4f}{seed_result['mAP50_95']:<10.4f}")
    print(f"{'seed+random':<20}{n_seed + n_budget:<10}{random_result['mAP50']:<10.4f}{random_result['mAP50_95']:<10.4f}")
    print(f"{'seed+active':<20}{n_seed + n_budget:<10}{active_result['mAP50']:<10.4f}{active_result['mAP50_95']:<10.4f}")
    delta = active_result["mAP50"] - random_result["mAP50"]
    print(f"\nActive vs random mAP50 delta: {delta:+.4f}")


if __name__ == "__main__":
    main()
