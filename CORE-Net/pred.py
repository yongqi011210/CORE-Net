import os
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO


def to_px(poly_norm: np.ndarray, w: int, h: int) -> np.ndarray:
    """(4,2) normalized -> pixel int"""
    pts = poly_norm.copy()
    pts[:, 0] *= w
    pts[:, 1] *= h
    return pts.astype(int)


def draw_pred_only_red(image_path: str, pred_path: str, save_path: str, thickness: int = 2):
    img = cv2.imread(image_path)
    if img is None:
        print(f"[WARN] failed to read image: {image_path}")
        return
    h, w = img.shape[:2]

    if not os.path.exists(pred_path):
        print(f"[INFO] no PRED label: {pred_path}")
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        cv2.imwrite(save_path, img)
        return

    pred_polys = []
    with open(pred_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            # 期望格式：cls x1 y1 x2 y2 x3 y3 x4 y4（共 9 项及以上）
            if len(parts) < 9:
                continue
            coords = list(map(float, parts[1:9]))
            poly = np.array(coords, dtype=float).reshape(4, 2)
            pred_polys.append(poly)

    # 只画 Pred：红色 (B,G,R) = (0,0,255)
    for p in pred_polys:
        cv2.polylines(img, [to_px(p, w, h)], True, (0, 0, 255), thickness)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    cv2.imwrite(save_path, img)
    print(f"[SAVE][PRED-RED] {save_path}")


def main():
    # ====== 1) 预测 ======
    model = YOLO(
        "E:/mul/CRU-YOLO/runs/obb/yolo11-obb/weights/best.pt",
        task="obb",
    )

    results = model.predict(
        source="E:/mul/data/RSDD/test_inshore/images/",
        save=True,        # 保存带框图片（原版颜色/样式）
        save_txt=True,    # 保存预测标签到 save_dir/labels
        save_conf=False,
        show_labels=False,
        show_conf=False,
        imgsz=512,
        device="cuda:0",
        conf=0.1,  # rcnn.score_thr
        iou=0.8,  # rcnn.nms.iou_thr
        max_det=2000,  # rcnn.max_per_img
    )

    # ====== 2) 用 txt 重新画一份“红色pred”可视化 ======
    save_dir = Path(results[0].save_dir)       # runs/obb/predict*
    pred_dir = save_dir / "labels"
    img_out_dir = save_dir                     # 预测图片也在这里

    for r in results:
        src_path = Path(r.path)
        name = src_path.stem

        # 找到 Ultralytics 保存的预测图片
        img_path = img_out_dir / src_path.name
        if not img_path.exists():
            for ext in [".jpg", ".png", ".jpeg", ".bmp"]:
                cand = img_out_dir / f"{name}{ext}"
                if cand.exists():
                    img_path = cand
                    break

        if not img_path.exists():
            print(f"[WARN] no predicted image found for {src_path.name}")
            continue

        pred_txt = pred_dir / f"{name}.txt"
        save_path = img_out_dir / f"{name}_pred_red.jpg"

        draw_pred_only_red(
            image_path=str(img_path),
            pred_path=str(pred_txt),
            save_path=str(save_path),
            thickness=2,
        )

    print("[DONE] Pred-only (red) visualization finished!")


if __name__ == "__main__":
    main()
