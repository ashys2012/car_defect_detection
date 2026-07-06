"""
Honest evaluation report for a trained checkpoint: test-set mAP, per-class AP
sorted worst-to-best (not alphabetically - the point is to surface weak
classes, not hide them), and pointers to the confusion matrix / PR curves
Ultralytics already saves.

Usage:
    source .venv/bin/activate
    python evaluate.py --weights runs/detect/cardd_baseline/weights/best.pt
"""

import argparse

from ultralytics import YOLO

DATA_YAML = "data/cardd.yaml"
WEAK_AP50_THRESHOLD = 0.3


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", required=True)
    parser.add_argument("--data", default=DATA_YAML)
    parser.add_argument("--device", default="0")
    args = parser.parse_args()

    model = YOLO(args.weights)
    metrics = model.val(data=args.data, split="test", device=args.device, verbose=False)

    per_class = [(name, float(metrics.box.ap50[i]), float(metrics.box.ap[i]))
                 for i, name in model.names.items()]
    per_class.sort(key=lambda x: x[1])  # worst AP50 first

    print(f"\nTest set: mAP50={metrics.box.map50:.4f}  mAP50-95={metrics.box.map:.4f}")
    print(f"{'class':<16}{'AP50':<10}{'AP50-95':<10}")
    for name, ap50, ap in per_class:
        flag = "  <-- weak" if ap50 < WEAK_AP50_THRESHOLD else ""
        print(f"{name:<16}{ap50:<10.4f}{ap:<10.4f}{flag}")

    weak = [name for name, ap50, _ in per_class if ap50 < WEAK_AP50_THRESHOLD]
    if weak:
        print(f"\nClasses below AP50={WEAK_AP50_THRESHOLD}: {weak}")
        print("These are the classes to prioritize in the next active-learning round.")

    print(f"\nConfusion matrix / PR curves saved under: {metrics.save_dir}")


if __name__ == "__main__":
    main()
