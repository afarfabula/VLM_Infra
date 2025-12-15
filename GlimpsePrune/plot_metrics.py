import argparse
import json
import os
import sys
from collections import defaultdict

def load_log_history(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    hist = data.get("log_history", [])
    return hist

def collect_keys(hist, prefix=None, exact=None):
    keys = set()
    for item in hist:
        for k in item.keys():
            if exact and k in exact:
                keys.add(k)
            elif prefix and k.startswith(prefix):
                keys.add(k)
    return sorted(keys)

def build_series(hist, keys):
    series = {}
    for k in keys:
        points = {}
        for item in hist:
            if k in item and "step" in item:
                points[item["step"]] = item[k]
        if len(points) > 0:
            xs = sorted(points.keys())
            ys = [points[x] for x in xs]
            series[k] = (xs, ys)
    return series

def ensure_matplotlib():
    try:
        import matplotlib
        import matplotlib.pyplot as plt
        return matplotlib, plt
    except Exception:
        print("matplotlib 未安装，无法绘图。请先安装 matplotlib")
        sys.exit(1)

def get_cjk_fontprop(matplotlib):
    try:
        import matplotlib.font_manager as fm
        # Candidate font family names commonly available on Linux servers
        candidates = [
            "Noto Sans CJK SC",
            "Noto Sans CJK",
            "Source Han Sans SC",
            "Source Han Sans",
            "WenQuanYi Zen Hei",
            "SimHei",
            "AR PL UMing CN",
            "MS Gothic",
            "IPAGothic",
            "Arial Unicode MS",
        ]
        for name in candidates:
            try:
                path = fm.findfont(name, fallback_to_default=False)
                if path and os.path.exists(path):
                    return fm.FontProperties(fname=path)
            except Exception:
                continue
        # Fallback: scan system fonts for CJK-related names
        for fp in fm.findSystemFonts(fontext="ttf"):
            lower = os.path.basename(fp).lower()
            if any(k in lower for k in ["noto", "sourcehan", "wenquanyi", "simhei", "arialuni", "ipagothic", "msgothic"]):
                return fm.FontProperties(fname=fp)
    except Exception:
        pass
    return None

def plot_groups(output_dir, hist):
    matplotlib, plt = ensure_matplotlib()
    # Enable unicode minus and attempt to use a CJK-capable font
    matplotlib.rcParams["axes.unicode_minus"] = False
    fp = get_cjk_fontprop(matplotlib)
    # Titles and labels: use Chinese if CJK font found, else fallback to English
    ylabel_cn = "值" if fp is not None else "Value"
    titles = {
        "loss": ("损失相关指标" if fp is not None else "Loss Metrics"),
        "box": ("Box相关指标" if fp is not None else "Box Metrics"),
        "mask": ("Mask相关指标" if fp is not None else "Mask Metrics"),
    }
    title_kwargs = {"fontproperties": fp} if fp is not None else {}
    ylabel_kwargs = {"fontproperties": fp} if fp is not None else {}
    loss_keys = set()
    loss_keys.update(collect_keys(hist, exact={"loss", "le_loss"}))
    loss_keys.update(collect_keys(hist, prefix="loc_loss_"))
    loss_series = build_series(hist, sorted(loss_keys))
    if len(loss_series) > 0:
        plt.figure(figsize=(9, 6))
        for k, (xs, ys) in loss_series.items():
            plt.plot(xs, ys, label=k)
        plt.xlabel("Step")
        plt.ylabel(ylabel_cn, **ylabel_kwargs)
        plt.title(titles["loss"], **title_kwargs)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "metrics_losses.png"))
        plt.close()
    box_keys = collect_keys(hist, prefix="box/")
    box_series = build_series(hist, box_keys)
    if len(box_series) > 0:
        plt.figure(figsize=(9, 6))
        for k, (xs, ys) in box_series.items():
            plt.plot(xs, ys, label=k)
        plt.xlabel("Step")
        plt.ylabel(ylabel_cn, **ylabel_kwargs)
        plt.title(titles["box"], **title_kwargs)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "metrics_box.png"))
        plt.close()
    mask_keys = collect_keys(hist, exact={"pred_mask_ratio", "ref_mask_ratio"})
    mask_series = build_series(hist, mask_keys)
    if len(mask_series) > 0:
        plt.figure(figsize=(9, 6))
        for k, (xs, ys) in mask_series.items():
            plt.plot(xs, ys, label=k)
        plt.xlabel("Step")
        plt.ylabel(ylabel_cn, **ylabel_kwargs)
        plt.title(titles["mask"], **title_kwargs)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "metrics_masks.png"))
        plt.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trainer_state", type=str, default="/data/model/Inference_VLM/VLM_Infra/GlimpsePrune/output/llava1_5_7b_gp_0801/trainer_state.json")
    parser.add_argument("--out_dir", type=str, default=None)
    args = parser.parse_args()
    hist = load_log_history(args.trainer_state)
    out_dir = args.out_dir
    if out_dir is None:
        out_dir = os.path.dirname(args.trainer_state)
    os.makedirs(out_dir, exist_ok=True)
    plot_groups(out_dir, hist)
    print("完成绘图。输出目录:", out_dir)

if __name__ == "__main__":
    main()
