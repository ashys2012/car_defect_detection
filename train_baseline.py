"""
Baseline YOLOv11 fine-tune on CarDD.

Starting point is the stock COCO-pretrained yolo11n.pt (transfer learning, not
training from scratch - matches the design doc's "Rejected Approaches": training
from scratch was ruled out given limited data). This is a rough baseline, not a
tuned model - the exercise brief explicitly says accuracy/mAP isn't being scored.

Usage:
    source .venv/bin/activate
    python train_baseline.py --epochs 50
"""

import argparse

from ultralytics import YOLO

DATA_YAML = "data/cardd.yaml"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--weights", type=str, default="yolo11n.pt")
    parser.add_argument("--name", type=str, default="cardd_baseline")
    parser.add_argument("--device", type=str, default="0")
    args = parser.parse_args()

    model = YOLO(args.weights)
    model.train(
        data=DATA_YAML,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        name=args.name,
        patience=15,
        seed=0,
    )

    # Test-set metrics, reported separately from val (used during training for
    # model selection / early stopping) so the headline numbers aren't the same
    # split the checkpoint was tuned against.
    # (project= intentionally omitted: a relative project string gets doubled onto
    # RUNS_DIR/task/project by Ultralytics, e.g. "runs/detect" -> runs/detect/runs/detect/...)
    metrics = model.val(
        data=DATA_YAML, split="test", device=args.device,
        name=f"{args.name}_test",
    )
    print("\n=== TEST SET METRICS ===")
    print(f"mAP50:    {metrics.box.map50:.4f}")
    print(f"mAP50-95: {metrics.box.map:.4f}")
    for i, name in model.names.items():
        print(f"  {name}: AP50={metrics.box.ap50[i]:.4f}")


if __name__ == "__main__":
    main()
