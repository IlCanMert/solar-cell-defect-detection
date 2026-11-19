"""Interactive Flask web interface for SENG435 Solar Cell Defect Classification."""

from __future__ import annotations

import argparse
import base64
import csv
import io
import json
import sys
import threading
import webbrowser
from pathlib import Path
from typing import Any

import cv2
import matplotlib

matplotlib.use("Agg")
import numpy as np
import pandas as pd
from flask import Flask, abort, jsonify, render_template, request, send_file

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.demo import _build_meta_from_row, _lookup_split_metadata  # noqa: E402
from src.predict import (  # noqa: E402
    load_image_from_path,
    load_inference_bundle,
    predict_single_image,
    render_prediction_card,
)
from src.preprocessing import validate_grayscale_image  # noqa: E402

PRESET_CASES: list[dict[str, Any]] = [
    {
        "subset_index": 1180,
        "tag": "Sağlıklı",
        "title": "Hücre #1180",
        "sub": "Monokristal",
        "desc": "İki model de sağlıklı etiketiyle eşleşiyor.",
    },
    {
        "subset_index": 470,
        "tag": "Kusurlu",
        "title": "Hücre #470",
        "sub": "Monokristal",
        "desc": "İki model de kusurlu etiketiyle eşleşiyor.",
    },
    {
        "subset_index": 1576,
        "tag": "Modeller farklı karar verdi",
        "title": "Hücre #1576",
        "sub": "Polikristal",
        "desc": "Tüm özellikli model yanlış alarm verdi. GA, sağlıklı etiketiyle eşleşiyor.",
    },
    {
        "subset_index": 850,
        "tag": "Kusur gözden kaçtı",
        "title": "Hücre #850",
        "sub": "Monokristal",
        "desc": "İki model de kusurlu etiketli hücreyi sağlıklı olarak sınıflandırdı.",
    },
]

GALLERY_FIGURES: list[dict[str, str]] = [
    {
        "rel_path": "figures/dataset_samples.png",
        "title": "ELPV Veri Seti Örnekleri",
        "caption": "Sağlıklı ve kusurlu etiketlerine sahip iki tip güneş hücresi.",
    },
    {
        "rel_path": "figures/feature_visualization.png",
        "title": "Kenar ve doku ölçümleri",
        "caption": "HOG çizgilerin yönünü, LBP piksel çevresindeki dokuyu özetler.",
    },
    {
        "rel_path": "figures/ga_convergence.png",
        "title": "Genetik algoritmanın arama süreci",
        "caption": "Arama ilerledikçe seçim puanı ve seçilen özellik sayısı değişiyor.",
    },
    {
        "rel_path": "figures/selected_features_breakdown.png",
        "title": "Kullanılan görüntü özellikleri",
        "caption": "Ana GA koşusu 324 kenar ölçümünden 198’ini ve 10 doku ölçümünün tamamını seçti.",
    },
    {
        "rel_path": "figures/confusion_matrices.png",
        "title": "Doğru ve yanlış tahmin sayıları",
        "caption": "Satırlar bilinen etiketi, sütunlar modelin tahminini gösterir.",
    },
    {
        "rel_path": "figures/error_analysis_grid.png",
        "title": "Tahmin örnekleri",
        "caption": "Doğru tahminler, yanlış alarmlar ve modelin kaçırdığı kusurlar.",
    },
    {
        "rel_path": "sample_predictions/e1_vs_e2_disagreements.png",
        "title": "İki modelin farklı kararları",
        "caption": "GA’nın düzelttiği ve yeni hata yaptığı hücrelerden örnekler.",
    },
    {
        "rel_path": "sample_predictions/demo_showcase.png",
        "title": "Örnek inceleme kartları",
        "caption": "Bir hücrenin görüntüsü, kenar ve doku ölçümleri ve iki modelin kararı.",
    },
]


def _encode_png_b64(img_arr: np.ndarray) -> str:
    """Encode a 2D grayscale or 3-channel BGR numpy image into a PNG base64 string."""
    ok, buf = cv2.imencode(".png", img_arr)
    if not ok:
        raise ValueError("Failed to encode image buffer to PNG")
    return base64.b64encode(buf.tobytes()).decode("ascii")


def _compute_outcome(y_true: int | None, pred_label: int) -> str | None:
    """Return confusion outcome tag (TP, TN, FP, FN) when ground truth is known."""
    if y_true is None:
        return None
    if y_true == 1 and pred_label == 1:
        return "TP"
    if y_true == 0 and pred_label == 0:
        return "TN"
    if y_true == 0 and pred_label == 1:
        return "FP"
    return "FN"


def _load_static_context(root: Path) -> dict[str, Any]:
    """Load dataset summary, metrics, GA history, error summary, and test split rows."""
    with (root / "results" / "dataset_summary.json").open("r", encoding="utf-8") as f:
        dataset_summary = json.load(f)
    with (root / "results" / "ga_history.json").open("r", encoding="utf-8") as f:
        ga_history = json.load(f)
    with (root / "results" / "ga_multiseed_summary.json").open("r", encoding="utf-8") as f:
        ga_multiseed = json.load(f)
    with (root / "results" / "sample_predictions" / "error_analysis_summary.json").open(
        "r", encoding="utf-8"
    ) as f:
        error_summary = json.load(f)

    with (root / "results" / "metrics.csv").open("r", encoding="utf-8") as f:
        metrics_list = list(csv.DictReader(f))
    metrics_map = {row["experiment_id"]: row for row in metrics_list}

    splits_path = root / "data" / "splits.csv"
    df_splits = pd.read_csv(splits_path) if splits_path.exists() else pd.DataFrame()

    fixed_indices = set(error_summary["e1_vs_e2_disagreements"]["fixed_subset_indices"])
    regressed_indices = set(error_summary["e1_vs_e2_disagreements"]["regressed_subset_indices"])

    test_samples: list[dict[str, Any]] = []
    if not df_splits.empty:
        df_test = df_splits[df_splits["split"] == "test"].copy()
        for _, row in df_test.iterrows():
            s_idx = int(row["subset_index"])
            test_samples.append(
                {
                    "subset_index": s_idx,
                    "original_index": int(row["original_index"]),
                    "image_path": str(row["image_path"]),
                    "filename": Path(str(row["image_path"])).name,
                    "cell_type": str(row["cell_type"]),
                    "label": int(row["label"]),
                    "label_name": str(row["label_name"]),
                    "is_fixed_by_ga": s_idx in fixed_indices,
                    "is_regressed_by_ga": s_idx in regressed_indices,
                }
            )

    return {
        "dataset": dataset_summary,
        "ga_history": {
            "search_time_s": ga_history["search_time_s"],
            "n_unique_evaluations": ga_history["n_unique_evaluations"],
            "n_cache_hits": ga_history["n_cache_hits"],
            "best_eval": ga_history["best_eval"],
            "ga_params": ga_history["ga_params"],
        },
        "ga_multiseed": ga_multiseed,
        "error_summary": error_summary,
        "metrics_list": metrics_list,
        "metrics": metrics_map,
        "df_splits": df_splits,
        "test_samples": test_samples,
    }


def create_app(project_root: str | Path | None = None) -> Flask:
    """Create and configure the Flask inspection dashboard application."""
    root = Path(project_root) if project_root is not None else PROJECT_ROOT
    bundle = load_inference_bundle(root)
    static_ctx = _load_static_context(root)

    app = Flask(__name__)
    app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16 MB upload cap
    card_lock = threading.Lock()

    allowed_fig_roots = [
        (root / "results" / "figures").resolve(),
        (root / "results" / "sample_predictions").resolve(),
    ]

    @app.get("/")
    def index() -> str:
        return render_template("dashboard.html")

    @app.get("/favicon.ico")
    def favicon() -> tuple[str, int]:
        return ("", 204)

    @app.get("/api/overview")
    def api_overview() -> Any:
        return jsonify(
            {
                "dataset": static_ctx["dataset"],
                "ga_history": static_ctx["ga_history"],
                "ga_multiseed": static_ctx["ga_multiseed"],
                "error_summary": static_ctx["error_summary"],
                "metrics_list": static_ctx["metrics_list"],
                "metrics": static_ctx["metrics"],
                "presets": PRESET_CASES,
                "gallery_figures": GALLERY_FIGURES,
            }
        )

    @app.get("/api/samples")
    def api_samples() -> Any:
        fixed_set = set(
            static_ctx["error_summary"]["e1_vs_e2_disagreements"]["fixed_subset_indices"]
        )
        return jsonify(
            {
                "total": len(static_ctx["test_samples"]),
                "fixed_count": len(fixed_set),
                "samples": static_ctx["test_samples"],
            }
        )

    @app.post("/api/predict")
    def api_predict() -> Any:
        df_splits = static_ctx["df_splits"]
        meta: dict[str, Any]

        try:
            if "file" in request.files and request.files["file"].filename:
                up = request.files["file"]
                raw_bytes = up.read()
                if not raw_bytes:
                    return jsonify({"error": "Yüklenen dosya boş."}), 400
                arr = np.frombuffer(raw_bytes, dtype=np.uint8)
                decoded = cv2.imdecode(arr, cv2.IMREAD_UNCHANGED)
                if decoded is None:
                    return jsonify({"error": "Görüntü dosyası çözümlenemedi."}), 400
                img = validate_grayscale_image(decoded)
                meta = _build_meta_from_row(None, fallback_name=Path(up.filename).name)
            else:
                payload = request.get_json(silent=True) or {}
                subset_index = payload.get("subset_index")
                image_path = payload.get("image_path")
                if subset_index is not None:
                    row = _lookup_split_metadata(df_splits, subset_index=int(subset_index))
                    if row is None:
                        return jsonify({"error": f"subset_index={subset_index} bulunamadı."}), 404
                    img, resolved = load_image_from_path(row["image_path"], root)
                    meta = _build_meta_from_row(row, fallback_name=resolved.name)
                elif image_path:
                    img, resolved = load_image_from_path(str(image_path), root)
                    row = _lookup_split_metadata(df_splits, image_path=str(image_path))
                    meta = _build_meta_from_row(row, fallback_name=resolved.name)
                else:
                    return jsonify({"error": "subset_index, image_path veya file gönderin."}), 400
        except (FileNotFoundError, KeyError, ValueError) as exc:
            return jsonify({"error": str(exc)}), 400

        pred = predict_single_image(img, bundle, include_visuals=True)
        pred["meta"] = meta

        # Attach this request's card so later predictions cannot overwrite a pending download.
        with io.BytesIO() as card, card_lock:
            render_prediction_card([pred], card, suptitle=f"Cell Inspection: {meta['sample_tag']}")
            card_b64 = base64.b64encode(card.getvalue()).decode("ascii")

        vis = pred.pop("visuals")
        hog_u8 = np.clip(vis["hog_vis_rescaled"] * 255.0, 0.0, 255.0).astype(np.uint8)
        lbp_u8 = np.clip(np.round(vis["lbp_map"] * (255.0 / 9.0)), 0.0, 255.0).astype(
            np.uint8
        )
        lbp_bgr = cv2.applyColorMap(lbp_u8, cv2.COLORMAP_VIRIDIS)

        y_true = meta.get("true_label")
        pred["outcome_e1"] = _compute_outcome(y_true, pred["baseline_label"])
        pred["outcome_e2"] = _compute_outcome(y_true, pred["ga_label"])
        pred["images"] = {
            "raw_b64": _encode_png_b64(vis["img_gray"]),
            "hog_b64": _encode_png_b64(hog_u8),
            "lbp_b64": _encode_png_b64(lbp_bgr),
        }
        pred["lbp_hist"] = [round(float(x), 6) for x in vis["lbp_hist"]]
        pred["card_url"] = f"data:image/png;base64,{card_b64}"
        return jsonify(pred)

    @app.get("/api/figure/<path:rel_path>")
    def api_figure(rel_path: str) -> Any:
        candidate = (root / "results" / rel_path).resolve()
        if (
            not candidate.is_file()
            or candidate.suffix.lower() != ".png"
            or not any(candidate.is_relative_to(ar) for ar in allowed_fig_roots)
        ):
            abort(404)
        as_att = request.args.get("download") in {"1", "true", "yes"}
        return send_file(candidate, mimetype="image/png", as_attachment=as_att)

    return app


def run_smoke_test(app: Flask) -> None:
    """Verify every Flask UI route and inference endpoint using test_client()."""
    client = app.test_client()

    # 1. GET /
    r1 = client.get("/")
    assert r1.status_code == 200, f"GET / failed: {r1.status_code}"
    html = r1.get_data(as_text=True)
    assert "Güneş hücresinde" in html and "Örnek hücreyi incele" in html
    print("[PASS 1/7] GET / returned 200 OK with Turkish dashboard HTML.")

    # 2. GET /api/overview
    r2 = client.get("/api/overview")
    assert r2.status_code == 200, f"GET /api/overview failed: {r2.status_code}"
    ov = r2.get_json()
    assert abs(float(ov["metrics"]["E1"]["test_macro_f1"]) - 0.7258) < 1e-4
    assert abs(float(ov["metrics"]["E2_seed42"]["test_macro_f1"]) - 0.7325) < 1e-4
    print("[PASS 2/7] GET /api/overview verified E1 (0.7258) and E2_seed42 (0.7325) metrics.")

    # 3. GET /api/samples
    r3 = client.get("/api/samples")
    assert r3.status_code == 200, f"GET /api/samples failed: {r3.status_code}"
    samp = r3.get_json()
    assert samp["total"] == 334, f"Expected 334 test samples, got {samp['total']}"
    assert samp["fixed_count"] == 11, f"Expected 11 GA-fixed samples, got {samp['fixed_count']}"
    print("[PASS 3/7] GET /api/samples returned 334 locked test cells and 11 GA-fixed samples.")

    # 4. POST /api/predict JSON subset_index=1576
    r4 = client.post("/api/predict", json={"subset_index": 1576})
    assert r4.status_code == 200, f"POST #1576 failed: {r4.status_code}"
    p1576 = r4.get_json()
    assert p1576["baseline_name"] == "DEFECTIVE"
    assert p1576["ga_name"] == "HEALTHY"
    assert p1576["models_agree"] is False
    assert all(len(p1576["images"][k]) > 100 for k in ("raw_b64", "hog_b64", "lbp_b64"))
    print("[PASS 4/7] POST /api/predict #1576 confirmed E1 FP -> E2 GA-SVM fix + PNG payloads.")

    # 5. POST /api/predict JSON subset_index=470
    r5 = client.post("/api/predict", json={"subset_index": 470})
    assert r5.status_code == 200, f"POST #470 failed: {r5.status_code}"
    p470 = r5.get_json()
    assert p470["ga_name"] == "DEFECTIVE"
    print("[PASS 5/7] POST /api/predict #470 verified DEFECTIVE detection.")

    # 6. POST /api/predict multipart file upload
    synth = np.full((300, 300), 140, dtype=np.uint8)
    ok, buf = cv2.imencode(".png", synth)
    assert ok
    r6 = client.post(
        "/api/predict",
        data={"file": (io.BytesIO(buf.tobytes()), "synthetic_cell.png")},
        content_type="multipart/form-data",
    )
    assert r6.status_code == 200, f"POST upload failed: {r6.status_code}"
    pup = r6.get_json()
    assert pup["image_shape"] == [300, 300]
    assert len(pup["lbp_hist"]) == 10
    print("[PASS 6/7] POST /api/predict multipart file upload verified cleanly.")

    # 7. GET /api/figure/figures/ga_convergence.png
    r7 = client.get("/api/figure/figures/ga_convergence.png")
    assert r7.status_code == 200 and r7.mimetype == "image/png"
    assert len(r7.data) > 10_000
    print("[PASS 7/7] GET /api/figure/figures/ga_convergence.png served valid PNG.")
    print("\nAll 7 Flask UI smoke checks passed!")


def parse_args() -> argparse.Namespace:
    """Parse CLI options for launching or smoke-testing the Flask dashboard."""
    parser = argparse.ArgumentParser(
        description="Launch the interactive web interface for SENG435 Solar Cell Defect Classification."
    )
    parser.add_argument("--host", default="127.0.0.1", help="Bind host (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=5000, help="Bind port (default: 5000)")
    parser.add_argument(
        "--no-browser",
        action="store_true",
        help="Do not automatically open the browser tab on startup.",
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Run automated test_client() checks across all routes and exit.",
    )
    return parser.parse_args()


def main() -> None:
    """Entry point for starting the Flask web server or running smoke verification."""
    args = parse_args()
    app = create_app(PROJECT_ROOT)

    if args.smoke_test:
        run_smoke_test(app)
        return

    url = f"http://{args.host}:{args.port}"
    print(f"Starting SENG435 Solar Cell Defect Inspection Web UI at {url}")
    if not args.no_browser:
        threading.Timer(1.0, lambda: webbrowser.open(url)).start()

    app.run(host=args.host, port=args.port, debug=False, use_reloader=False)


if __name__ == "__main__":
    main()
