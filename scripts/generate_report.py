"""Generate formatted academic PDF and DOCX final reports from verified experiment results."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor
from PIL import Image as PILImage
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    HRFlowable,
    Image as RLImage,
    KeepTogether,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _load_verified_artifacts(root: Path) -> dict[str, Any]:
    """Load all verified dataset, validation, evaluation, GA, and error-analysis files."""
    with (root / "results" / "dataset_summary.json").open("r", encoding="utf-8") as f:
        dataset_summary = json.load(f)
    with (root / "results" / "ga_history.json").open("r", encoding="utf-8") as f:
        ga_history = json.load(f)
    with (root / "results" / "ga_multiseed_summary.json").open("r", encoding="utf-8") as f:
        ga_multiseed = json.load(f)
    with (root / "results" / "evaluation_summary.json").open("r", encoding="utf-8") as f:
        eval_summary = json.load(f)
    with (root / "results" / "sample_predictions" / "error_analysis_summary.json").open(
        "r", encoding="utf-8"
    ) as f:
        error_summary = json.load(f)
    with (root / "results" / "feature_extraction_summary.json").open("r", encoding="utf-8") as f:
        feat_summary = json.load(f)

    with (root / "results" / "baseline_validation_grid.csv").open("r", encoding="utf-8") as f:
        val_grid = list(csv.DictReader(f))
    with (root / "results" / "metrics.csv").open("r", encoding="utf-8") as f:
        metrics_rows = list(csv.DictReader(f))

    return {
        "dataset": dataset_summary,
        "features": feat_summary,
        "val_grid": val_grid,
        "ga_history": ga_history,
        "ga_multiseed": ga_multiseed,
        "eval_summary": eval_summary,
        "error_summary": error_summary,
        "metrics": {r["experiment_id"]: r for r in metrics_rows},
    }


def _register_fonts() -> tuple[str, str, str, str, str]:
    """Register Unicode TrueType fonts on Windows when present, else return standard Type1."""
    win_fonts = Path(r"C:\Windows\Fonts")
    times = win_fonts / "times.ttf"
    times_bd = win_fonts / "timesbd.ttf"
    times_it = win_fonts / "timesi.ttf"
    arial = win_fonts / "arial.ttf"
    arial_bd = win_fonts / "arialbd.ttf"

    if all(p.exists() for p in (times, times_bd, times_it, arial, arial_bd)):
        pdfmetrics.registerFont(TTFont("ReportSerif", str(times)))
        pdfmetrics.registerFont(TTFont("ReportSerifBold", str(times_bd)))
        pdfmetrics.registerFont(TTFont("ReportSerifItalic", str(times_it)))
        pdfmetrics.registerFont(TTFont("ReportSans", str(arial)))
        pdfmetrics.registerFont(TTFont("ReportSansBold", str(arial_bd)))
        pdfmetrics.registerFontFamily(
            "ReportSerif",
            normal="ReportSerif",
            bold="ReportSerifBold",
            italic="ReportSerifItalic",
            boldItalic="ReportSerifBold",
        )
        pdfmetrics.registerFontFamily(
            "ReportSans",
            normal="ReportSans",
            bold="ReportSansBold",
            italic="ReportSans",
            boldItalic="ReportSansBold",
        )
        return "ReportSerif", "ReportSerifBold", "ReportSerifItalic", "ReportSans", "ReportSansBold"

    return "Times-Roman", "Times-Bold", "Times-Italic", "Helvetica", "Helvetica-Bold"


def _scaled_rl_image(img_path: Path, max_w_cm: float, max_h_cm: float) -> RLImage:
    """Create a ReportLab Image scaled proportionally within max width/height bounds."""
    with PILImage.open(img_path) as im:
        w_px, h_px = im.size
    max_w = max_w_cm * cm
    max_h = max_h_cm * cm
    scale = min(max_w / float(w_px), max_h / float(h_px))
    return RLImage(str(img_path), width=w_px * scale, height=h_px * scale)


def _header_footer(canvas: Any, doc: Any) -> None:
    """Draw running academic header and centered page number footer on PDF pages."""
    canvas.saveState()
    page_num = canvas.getPageNumber()
    if page_num > 1:
        canvas.setFont("ReportSans", 8)
        canvas.setFillColor(colors.HexColor("#475569"))
        canvas.drawString(
            1.8 * cm,
            A4[1] - 1.25 * cm,
            "SENG435 Course Project Report | Solar Cell Defect Classification via GA Feature Selection",
        )
        canvas.setStrokeColor(colors.HexColor("#cbd5e1"))
        canvas.setLineWidth(0.5)
        canvas.line(1.8 * cm, A4[1] - 1.40 * cm, A4[0] - 1.8 * cm, A4[1] - 1.40 * cm)

    canvas.setFont("ReportSans", 8.5)
    canvas.setFillColor(colors.HexColor("#475569"))
    canvas.drawCentredString(A4[0] / 2.0, 1.15 * cm, f"Page {page_num}")
    canvas.restoreState()


def _styled_table(
    data: list[list[Any]],
    col_widths: list[float],
    highlight_row: int | None = None,
    font_name: str = "ReportSans",
    bold_font: str = "ReportSansBold",
    font_size: float = 8.0,
) -> Table:
    """Build a crisp shaded academic table."""
    tbl = Table(data, colWidths=col_widths, repeatRows=1)
    cmds: list[Any] = [
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1e3a8a")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), bold_font),
        ("FONTSIZE", (0, 0), (-1, 0), font_size),
        ("FONTNAME", (0, 1), (-1, -1), font_name),
        ("FONTSIZE", (0, 1), (-1, -1), font_size),
        ("ALIGN", (1, 0), (-1, -1), "CENTER"),
        ("ALIGN", (0, 0), (0, -1), "LEFT"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f8fafc")]),
        ("GRID", (0, 0), (-1, -1), 0.45, colors.HexColor("#cbd5e1")),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("LEFTPADDING", (0, 0), (-1, -1), 5),
        ("RIGHTPADDING", (0, 0), (-1, -1), 5),
    ]
    if highlight_row is not None:
        cmds.extend(
            [
                ("BACKGROUND", (0, highlight_row), (-1, highlight_row), colors.HexColor("#eff6ff")),
                ("FONTNAME", (0, highlight_row), (-1, highlight_row), bold_font),
            ]
        )
    tbl.setStyle(TableStyle(cmds))
    return tbl


def build_pdf_report(
    root: Path,
    art: dict[str, Any],
    author: str,
    student_id: str,
    out_pdf: Path,
) -> Path:
    """Compile the formatted academic PDF report using ReportLab."""
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    serif, serif_b, serif_i, sans, sans_b = _register_fonts()
    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        "AcadTitle",
        parent=styles["Title"],
        fontName=sans_b,
        fontSize=15.5,
        leading=19.5,
        alignment=TA_CENTER,
        textColor=colors.HexColor("#0f172a"),
        spaceAfter=6,
    )
    meta_style = ParagraphStyle(
        "AcadMeta",
        fontName=sans,
        fontSize=9.5,
        leading=13.5,
        alignment=TA_CENTER,
        textColor=colors.HexColor("#334155"),
        spaceAfter=10,
    )
    h1_style = ParagraphStyle(
        "AcadH1",
        fontName=sans_b,
        fontSize=12.0,
        leading=15.0,
        textColor=colors.HexColor("#1e3a8a"),
        spaceBefore=12,
        spaceAfter=5,
        keepWithNext=True,
    )
    h2_style = ParagraphStyle(
        "AcadH2",
        fontName=sans_b,
        fontSize=10.3,
        leading=13.5,
        textColor=colors.HexColor("#0f172a"),
        spaceBefore=8,
        spaceAfter=4,
        keepWithNext=True,
    )
    body_style = ParagraphStyle(
        "AcadBody",
        fontName=serif,
        fontSize=10.2,
        leading=14.5,
        alignment=TA_JUSTIFY,
        textColor=colors.HexColor("#111827"),
        spaceAfter=6,
    )
    abstract_style = ParagraphStyle(
        "AcadAbstract",
        fontName=serif,
        fontSize=9.6,
        leading=13.5,
        alignment=TA_JUSTIFY,
        leftIndent=8,
        rightIndent=8,
        textColor=colors.HexColor("#1e293b"),
        spaceAfter=6,
    )
    bullet_style = ParagraphStyle(
        "AcadBullet",
        parent=body_style,
        leftIndent=14,
        firstLineIndent=-8,
        spaceAfter=4,
    )
    formula_style = ParagraphStyle(
        "AcadFormula",
        fontName=serif_i,
        fontSize=10.4,
        leading=14.0,
        alignment=TA_CENTER,
        textColor=colors.HexColor("#0f172a"),
        spaceBefore=4,
        spaceAfter=7,
    )
    caption_style = ParagraphStyle(
        "AcadCaption",
        fontName=sans,
        fontSize=8.6,
        leading=11.8,
        alignment=TA_JUSTIFY,
        textColor=colors.HexColor("#334155"),
        spaceBefore=4,
        spaceAfter=8,
    )
    ref_style = ParagraphStyle(
        "AcadRef",
        fontName=serif,
        fontSize=9.2,
        leading=12.8,
        alignment=TA_LEFT,
        leftIndent=16,
        firstLineIndent=-16,
        textColor=colors.HexColor("#111827"),
        spaceAfter=4,
    )

    m = art["metrics"]
    e1 = m["E1"]
    e2 = m["E2_seed42"]
    e2m = m["E2_3seed_mean"]
    e3 = m["E3"]
    e4 = m["E4"]
    e5s = m["E5_seed42"]
    e5m = m["E5_5seed_mean"]
    ds = art["dataset"]
    fs = art["features"]
    ga_ms = art["ga_multiseed"]["mean_std"]
    err = art["error_summary"]
    ct = err["cell_type_breakdown"]
    dis = err["e1_vs_e2_disagreements"]

    doc = SimpleDocTemplate(
        str(out_pdf),
        pagesize=A4,
        leftMargin=1.8 * cm,
        rightMargin=1.8 * cm,
        topMargin=1.9 * cm,
        bottomMargin=1.8 * cm,
        title="Intelligent Solar Cell Defect Classification Using Genetic Algorithm-Based Feature Selection",
        author=f"{author} ({student_id})",
    )

    story: list[Any] = []

    # Header / Title block
    story.append(
        Paragraph(
            "Intelligent Solar Cell Defect Classification Using Genetic Algorithm-Based Feature Selection",
            title_style,
        )
    )
    author_line = f"<b>Author:</b> {author} &nbsp;|&nbsp; <b>Student ID:</b> {student_id}"
    story.append(
        Paragraph(
            f"<b>Course:</b> SENG435 — Introduction to Digital Image Processing &nbsp;|&nbsp; "
            f"<b>Instructor:</b> Assoc. Prof. Tuba ÇAĞLIKANTAR<br/>{author_line} &nbsp;|&nbsp; "
            f"<b>Date:</b> December 2026",
            meta_style,
        )
    )
    story.append(HRFlowable(width="100%", thickness=1.0, color=colors.HexColor("#1e3a8a")))
    story.append(Spacer(1, 8))

    # Abstract & Keywords
    story.append(Paragraph("Abstract", h1_style))
    story.append(
        Paragraph(
            f"Automated visual inspection of photovoltaic (PV) solar cells using electroluminescence (EL) "
            f"imaging enables early detection of micro-cracks, finger interruptions, and electrically inactive "
            f"regions invisible under visible-light cameras. However, dense local descriptors introduce redundant "
            f"dimensions and degrade generalization across heterogeneous monocrystalline and polycrystalline grain "
            f"backgrounds. This study evaluates a classification and evolutionary feature selection pipeline "
            f"on a held-out cell-level split of ELPV. From the {ds['raw_total_images']:,} raw "
            f"grayscale EL images (300×300), we extract the {ds['binary_subset_total']:,} definite binary-labeled "
            f"samples ({ds['binary_healthy']:,} HEALTHY, {ds['binary_defective']} DEFECTIVE) and partition them via "
            f"joint stratification across class label and crystal type (1,556 train, 333 validation, 334 locked test). "
            f"Each cell is resized to 64×64 pixels and represented by a 334-dimensional vector combining 324 Histogram of "
            f"Oriented Gradients (HOG) structural features and 10 normalized Uniform Local Binary Pattern (LBP) texture "
            f"bins. A baseline StandardScaler + LinearSVC (C=0.001, class_weight='balanced') using all 334 features "
            f"obtains a locked test Macro F1 of {float(e1['test_macro_f1']):.4f}, Defective Recall of "
            f"{float(e1['test_defective_recall']):.4f}, and Accuracy of {float(e1['test_accuracy']):.4f}. Our binary-mask "
            f"Genetic Algorithm (GA) wrapper, optimized strictly via 3-fold stratified cross-validation on the training split, "
            f"selects {int(float(e2['selected_features']))} of 334 features ({int(float(e2['hog_features']))}/324 HOG and "
            f"{int(float(e2['lbp_features']))}/10 LBP bins; {float(e2['feature_reduction_pct']):.2f}% reduction) on seed=42 "
            f"while raising locked test Macro F1 to {float(e2['test_macro_f1']):.4f}, Defective Recall to "
            f"{float(e2['test_defective_recall']):.4f}, and Accuracy to {float(e2['test_accuracy']):.4f}. Across three "
            f"independent seeds (42, 123, 999), GA reduces dimensionality by {ga_ms['feature_reduction_pct']['mean']:.2f}% ± "
            f"{ga_ms['feature_reduction_pct']['std']:.2f}% ({ga_ms['selected_features']['mean']:.2f} ± "
            f"{ga_ms['selected_features']['std']:.2f} features) with mean test Macro F1 of {ga_ms['test_macro_f1']['mean']:.4f} "
            f"± {ga_ms['test_macro_f1']['std']:.4f}, close to the all-feature baseline on these cells. "
            f"The released labels lack module identifiers, so cells from the same module may be present in different splits.",
            abstract_style,
        )
    )
    story.append(
        Paragraph(
            "<b>Keywords:</b> Solar Cell Defect Classification, Electroluminescence Imaging, Feature Selection, "
            "Genetic Algorithm, Histogram of Oriented Gradients (HOG), Local Binary Patterns (LBP), Support Vector Machine.",
            abstract_style,
        )
    )

    # Section 1
    story.append(Paragraph("1. Introduction & Problem Definition", h1_style))
    story.append(
        Paragraph(
            "Photovoltaic modules experience thermal and mechanical stress throughout manufacturing, shipping, and field "
            "deployment, inducing micro-cracks, broken interconnects, and electrically isolated cell regions. Because silicon "
            "solar cells emit near-infrared electroluminescence (EL) emission under forward bias roughly proportional to local "
            "electrical connectivity and carrier lifetime, defects appear as sharp dark lines or inactive patches in EL imagery "
            "even when undetected under visible RGB inspection [1, 2]. Manual inspection of thousands of cell images is slow, "
            "fatiguing, and subjective, motivating automated computer vision pipelines.",
            body_style,
        )
    )
    story.append(
        Paragraph(
            "Although deep convolutional networks achieve high accuracy on large curated datasets, classical pipelines pairing "
            "transparent edge and texture descriptors with linear maximum-margin classifiers remain attractive for CPU-only quality "
            "control (~1.3 ms per cell) and interpretable error diagnosis. However, dense gradient representations often carry "
            "redundant or noisy bins caused by busbar geometry and polycrystalline grain boundaries. This project investigates a "
            "concrete research question: <i>Can a Genetic Algorithm (GA) feature-selection wrapper over combined HOG and Uniform LBP "
            "descriptors reduce feature dimensionality relative to an all-feature LinearSVC baseline while maintaining or improving "
            "locked test-set defect classification performance?</i>",
            body_style,
        )
    )

    # Section 2
    story.append(Paragraph("2. Related Work", h1_style))
    story.append(
        Paragraph(
            "Buerhop-Lutz et al. [1] established the public ELPV benchmark of 2,624 EL cell images harvested from 44 commercial "
            "monocrystalline and polycrystalline modules with expert defect likelihood annotations. Deitsch et al. [2] benchmarked "
            "handcrafted local descriptors with SVMs against fine-tuned VGG-19 convolutional networks on ELPV, confirming that "
            "classical pipelines offer competitive performance at negligible computational overhead, while Deitsch et al. [3] "
            "subsequently automated cell extraction from uncalibrated module images.",
            body_style,
        )
    )
    story.append(
        Paragraph(
            "For feature extraction, Histogram of Oriented Gradients (HOG) introduced by Dalal and Triggs [4] summarizes local "
            "contrast-normalized edge orientations capable of capturing directional cracks and fracture discontinuities, whereas "
            "Uniform Local Binary Patterns (LBP) by Ojala et al. [5] capture rotation-invariant micro-texture balance across "
            "uniform silicon, spots, and grain clutter. Within maximum-margin classification via Support Vector Machines [6], "
            "wrapper feature selection via Genetic Algorithms [7] evolves binary inclusion masks directly against cross-validation "
            "performance rather than relying purely on univariate filter scores.",
            body_style,
        )
    )

    # Section 3
    story.append(Paragraph("3. Dataset & Preprocessing", h1_style))
    story.append(
        Paragraph(
            f"The ELPV dataset is loaded programmatically via the official Python package elpv-dataset [1] (images under "
            f"CC BY-NC-SA 4.0, loader code under Apache 2.0). The raw corpus contains {ds['raw_total_images']:,} 8-bit grayscale "
            f"images (300×300 pixels, uint8) across {ds['raw_cell_type_distribution']['mono']:,} monocrystalline ('mono') and "
            f"{ds['raw_cell_type_distribution']['poly']:,} polycrystalline ('poly') cells, with expert probabilities of "
            f"0.0 ({ds['raw_probability_distribution']['0.0']:,}), 0.3333 ({ds['raw_probability_distribution']['0.3333']}), "
            f"0.6667 ({ds['raw_probability_distribution']['0.6667']}), and 1.0 ({ds['raw_probability_distribution']['1.0']}). "
            f"To isolate unambiguous ground truth, the {ds['excluded_intermediate_count']} intermediate-probability samples are "
            f"excluded, yielding {ds['binary_subset_total']:,} binary images ({ds['binary_healthy']:,} HEALTHY [67.84%] and "
            f"{ds['binary_defective']} DEFECTIVE [32.16%]).",
            body_style,
        )
    )
    story.append(
        Paragraph(
            "Inspection of the bundled labels.csv metadata confirmed that only image path, defect probability, and crystal type "
            "are present (has_module_id = false). Because module-level identifiers are absent, module-grouped splitting could not be "
            "enforced; instead, we froze a 70% / 15% / 15% partition (data/splits.csv, seed=42) using joint stratification across "
            "(label, cell_type) so defect rate (~32.2%) and mono/poly ratio remain identical across splits (Table 1), and we "
            "note that cells from the same module may appear in different splits. During preprocessing (src/preprocessing.py), "
            "each 300×300 image is validated and resized via area averaging (cv2.INTER_AREA) to 64×64 pixels, suppressing sensor "
            "noise and finger-grid aliasing while accelerating HOG computation.",
            body_style,
        )
    )

    sp = ds["splits"]
    t1_data = [
        ["Split", "Total", "Share", "Healthy (0)", "Defective (1)", "Def. Rate", "Mono", "Poly"],
        [
            "Train",
            str(sp["train"]["total"]),
            f"{sp['train']['percentage']:.2f}%",
            str(sp["train"]["healthy"]),
            str(sp["train"]["defective"]),
            f"{sp['train']['defective_rate_pct']:.2f}%",
            str(sp["train"]["mono"]),
            str(sp["train"]["poly"]),
        ],
        [
            "Validation",
            str(sp["val"]["total"]),
            f"{sp['val']['percentage']:.2f}%",
            str(sp["val"]["healthy"]),
            str(sp["val"]["defective"]),
            f"{sp['val']['defective_rate_pct']:.2f}%",
            str(sp["val"]["mono"]),
            str(sp["val"]["poly"]),
        ],
        [
            "Locked Test",
            str(sp["test"]["total"]),
            f"{sp['test']['percentage']:.2f}%",
            str(sp["test"]["healthy"]),
            str(sp["test"]["defective"]),
            f"{sp['test']['defective_rate_pct']:.2f}%",
            str(sp["test"]["mono"]),
            str(sp["test"]["poly"]),
        ],
        [
            "Binary Total",
            str(ds["binary_subset_total"]),
            "100.00%",
            str(ds["binary_healthy"]),
            str(ds["binary_defective"]),
            "32.16%",
            str(sp["train"]["mono"] + sp["val"]["mono"] + sp["test"]["mono"]),
            str(sp["train"]["poly"] + sp["val"]["poly"] + sp["test"]["poly"]),
        ],
    ]
    story.append(
        KeepTogether(
            [
                Paragraph(
                    "<b>Table 1.</b> Frozen stratified ELPV binary subset distribution across Train, Validation, and Locked Test partitions (data/splits.csv, seed=42).",
                    caption_style,
                ),
                _styled_table(
                    t1_data,
                    [2.8 * cm, 1.8 * cm, 2.0 * cm, 2.2 * cm, 2.3 * cm, 2.1 * cm, 1.9 * cm, 1.9 * cm],
                    highlight_row=4,
                    font_name=sans,
                    bold_font=sans_b,
                ),
                Spacer(1, 8),
            ]
        )
    )

    fig1 = _scaled_rl_image(root / "results" / "figures" / "dataset_samples.png", 16.5, 11.5)
    story.append(
        KeepTogether(
            [
                fig1,
                Paragraph(
                    "<b>Figure 1.</b> Representative 300×300 grayscale ELPV solar cell images spanning HEALTHY vs. DEFECTIVE classes and monocrystalline ('mono') vs. polycrystalline ('poly') wafer types.",
                    caption_style,
                ),
            ]
        )
    )

    # Section 4
    story.append(Paragraph("4. Methodology", h1_style))
    story.append(Paragraph("4.1 HOG + Uniform LBP Feature Representation (334 Features)", h2_style))
    story.append(
        Paragraph(
            f"From each standardized 64×64 image, two complementary handcrafted feature blocks are concatenated (src/features.py): "
            f"(1) <b>HOG</b> (indices 0–323, {fs['hog_dim']} features) via skimage.feature.hog [8] using orientations=9, "
            f"pixels_per_cell=(16, 16), cells_per_block=(2, 2), and block_norm='L2-Hys', producing 3×3 overlapping blocks of "
            f"2×2×9 = 36 gradient bins each; and (2) <b>Uniform LBP</b> (indices 324–333, {fs['lbp_dim']} features) with circular "
            f"neighbors P=8, radius R=1, and method='uniform', normalized into an L1 probability histogram summing to 1.0. "
            f"Extracting features across all 2,223 images required {fs['extraction_time_s']:.4f} s ({fs['ms_per_image']:.4f} ms/image) "
            f"with verified finite values (has_nan={fs['has_nan']}, has_inf={fs['has_inf']}).",
            body_style,
        )
    )
    fig2 = _scaled_rl_image(root / "results" / "figures" / "feature_visualization.png", 16.6, 10.5)
    story.append(
        KeepTogether(
            [
                fig2,
                Paragraph(
                    "<b>Figure 2.</b> Side-by-side comparison of raw EL cell, 64×64 HOG gradient magnitudes (324 features), 64×64 Uniform LBP code map, and normalized 10-bin LBP histogram for healthy cell0004 (top) and defective cracked cell0001 (bottom).",
                    caption_style,
                ),
            ]
        )
    )

    story.append(Paragraph("4.2 LinearSVC Baseline (E1) and Regularization Selection", h2_style))
    story.append(
        Paragraph(
            "Classification is implemented via sklearn.pipeline.Pipeline(StandardScaler(), LinearSVC(max_iter=5000, random_state=42)) [9]. "
            "Placing StandardScaler inside the Pipeline ensures mean/variance normalization is learned strictly on training/CV folds. "
            "Using 3-fold stratified CV on train and validation performance on val only (Table 2), strong regularization C=0.001 paired with "
            "class_weight='balanced' achieved the top Validation Macro F1 (0.7635) and Defective Recall (0.7103), whereas weaker "
            "regularization (C=1.0, C=10.0) overfit high-dimensional HOG variance (0.6848 and 0.6652 Val Macro F1). We therefore "
            "froze C=0.001 and class_weight='balanced' identically across E1–E5.",
            body_style,
        )
    )

    t2_data = [["class_weight", "C", "Train CV F1 (±Std)", "Val Macro F1", "Val Def. F1", "Val Def. Prec", "Val Def. Rec", "Val Acc"]]
    for row in art["val_grid"][:6]:
        t2_data.append(
            [
                row["class_weight"],
                row["C"],
                f"{float(row['cv_macro_f1_mean']):.4f} ± {float(row['cv_macro_f1_std']):.4f}",
                f"{float(row['val_macro_f1']):.4f}",
                f"{float(row['val_defective_f1']):.4f}",
                f"{float(row['val_defective_precision']):.4f}",
                f"{float(row['val_defective_recall']):.4f}",
                f"{float(row['val_accuracy']):.4f}",
            ]
        )
    story.append(
        KeepTogether(
            [
                Paragraph(
                    "<b>Table 2.</b> Top 6 validation grid configurations on Train CV and Validation split (full 10-row grid stored in results/baseline_validation_grid.csv; locked test split excluded).",
                    caption_style,
                ),
                _styled_table(
                    t2_data,
                    [2.1 * cm, 1.4 * cm, 3.3 * cm, 2.1 * cm, 2.0 * cm, 2.2 * cm, 2.0 * cm, 1.9 * cm],
                    highlight_row=1,
                    font_name=sans,
                    bold_font=sans_b,
                ),
                Spacer(1, 8),
            ]
        )
    )

    story.append(Paragraph("4.3 Binary-Mask Wrapper Genetic Algorithm (E2)", h2_style))
    story.append(
        Paragraph(
            "Each GA candidate chromosome is a 334-bit binary inclusion mask <i>m</i> with entries in {0, 1}<sup>334</sup> "
            "(src/ga_selection.py; empty masks are repaired by enabling one random feature bit). Candidate masks are scored "
            "exclusively inside the 1,556-sample train partition using 3-fold stratified CV with "
            "Pipeline(StandardScaler, LinearSVC(C=0.001, class_weight='balanced')) via the penalized objective:",
            body_style,
        )
    )
    story.append(
        Paragraph(
            "<b>Fitness(<i>m</i>) = mean_macro_F1<sub>CV</sub>(<i>m</i>) &minus; &lambda; &middot; (||<i>m</i>||<sub>1</sub> / 334)</b>, "
            "&nbsp;&nbsp; where &lambda; = 0.05",
            formula_style,
        )
    )
    story.append(
        Paragraph(
            "Search uses population size 12, 8 generations, tournament selection (k=3), two-point crossover (p_c=0.7), bit-flip mutation "
            "(p_m=0.002994 ≈ 1/334), elitism retaining the two highest-ranked candidates per generation, and a byte-key fitness cache preventing "
            "redundant SVM CV evaluations (26 cache hits out of 108 candidate evaluations on seed=42, finishing in 8.18 s).",
            body_style,
        )
    )
    fig3 = _scaled_rl_image(root / "results" / "figures" / "ga_convergence.png", 16.5, 9.5)
    story.append(
        KeepTogether(
            [
                fig3,
                Paragraph(
                    "<b>Figure 3.</b> Genetic Algorithm convergence trajectory (seed=42): best/mean penalized fitness and 3-fold train CV Macro F1 (left) alongside active feature count reduction from 240/334 down to 208/334 features (right).",
                    caption_style,
                ),
            ]
        )
    )

    # Section 5
    story.append(Paragraph("5. Experimental Results & Discussion", h1_style))
    story.append(Paragraph("5.1 Locked Test Set Evaluation Across E1–E5", h2_style))
    story.append(
        Paragraph(
            "After locking all splits, hyperparameters, and GA masks, experiments E1–E5 were executed once on the held-out test split "
            "(n = 334: 226 Healthy, 108 Defective). Table 3 summarizes the complete test benchmark from results/metrics.csv.",
            body_style,
        )
    )

    t3_data = [
        [
            "Exp ID",
            "Model / Feature Subset",
            "Feats (HOG/LBP)",
            "Red.",
            "Val F1",
            "Test Macro F1",
            "Def. Rec",
            "Test Acc",
            "Time (s)",
        ],
        [
            "E1",
            "Baseline SVM (All)",
            "334 (324/10)",
            "0.00%",
            f"{float(e1['val_macro_f1']):.4f}",
            f"{float(e1['test_macro_f1']):.4f}",
            f"{float(e1['test_defective_recall']):.4f}",
            f"{float(e1['test_accuracy']):.4f}",
            f"{float(e1['train_or_search_time_s']):.4f}",
        ],
        [
            "E2 (s=42)",
            "GA-SVM (seed=42)",
            "208 (198/10)",
            "37.72%",
            f"{float(e2['val_macro_f1']):.4f}",
            f"{float(e2['test_macro_f1']):.4f}",
            f"{float(e2['test_defective_recall']):.4f}",
            f"{float(e2['test_accuracy']):.4f}",
            f"{float(e2['train_or_search_time_s']):.2f} (GA)",
        ],
        [
            "E2 (3-sd)",
            "GA-SVM (3-Seed Mean)",
            "192.7 (184.7/8.0)",
            "42.31%",
            f"{float(e2m['val_macro_f1']):.4f}",
            f"{float(e2m['test_macro_f1']):.4f}",
            f"{float(e2m['test_defective_recall']):.4f}",
            f"{float(e2m['test_accuracy']):.4f}",
            f"{float(e2m['train_or_search_time_s']):.2f} (GA)",
        ],
        [
            "E3",
            "HOG-Only + SVM",
            "324 (324/0)",
            "2.99%",
            f"{float(e3['val_macro_f1']):.4f}",
            f"{float(e3['test_macro_f1']):.4f}",
            f"{float(e3['test_defective_recall']):.4f}",
            f"{float(e3['test_accuracy']):.4f}",
            f"{float(e3['train_or_search_time_s']):.4f}",
        ],
        [
            "E4",
            "LBP-Only + SVM",
            "10 (0/10)",
            "97.01%",
            f"{float(e4['val_macro_f1']):.4f}",
            f"{float(e4['test_macro_f1']):.4f}",
            f"{float(e4['test_defective_recall']):.4f}",
            f"{float(e4['test_accuracy']):.4f}",
            f"{float(e4['train_or_search_time_s']):.4f}",
        ],
        [
            "E5 (s=42)",
            "Random-208 (seed=42)",
            "208 (201/7)",
            "37.72%",
            f"{float(e5s['val_macro_f1']):.4f}",
            f"{float(e5s['test_macro_f1']):.4f}",
            f"{float(e5s['test_defective_recall']):.4f}",
            f"{float(e5s['test_accuracy']):.4f}",
            f"{float(e5s['train_or_search_time_s']):.4f}",
        ],
        [
            "E5 (5-sd)",
            "Random-208 (5-Seed)",
            "208 (202/6)",
            "37.72%",
            f"{float(e5m['val_macro_f1']):.4f}",
            f"{float(e5m['test_macro_f1']):.4f}",
            f"{float(e5m['test_defective_recall']):.4f}",
            f"{float(e5m['test_accuracy']):.4f}",
            f"{float(e5m['train_or_search_time_s']):.4f}",
        ],
    ]
    story.append(
        KeepTogether(
            [
                Paragraph(
                    "<b>Table 3.</b> Complete locked test comparison across E1–E5 (n=334). E2 (seed=42) achieves both the highest Test Macro F1 (0.7325) and 37.72% feature reduction.",
                    caption_style,
                ),
                _styled_table(
                    t3_data,
                    [1.65 * cm, 3.35 * cm, 2.50 * cm, 1.25 * cm, 1.40 * cm, 1.85 * cm, 1.45 * cm, 1.50 * cm, 2.25 * cm],
                    highlight_row=2,
                    font_name=sans,
                    bold_font=sans_b,
                    font_size=7.5,
                ),
                Spacer(1, 8),
            ]
        )
    )

    story.append(
        Paragraph(
            f"The primary GA-SVM run (E2, seed=42) retained {int(float(e2['selected_features']))} of 334 features "
            f"({float(e2['feature_reduction_pct']):.2f}% fewer) and showed small increases on this test split in Macro F1 "
            f"({float(e1['test_macro_f1']):.4f} → {float(e2['test_macro_f1']):.4f}), Defective F1 ({float(e1['test_defective_f1']):.4f} → "
            f"{float(e2['test_defective_f1']):.4f}), Defective Recall ({float(e1['test_defective_recall']):.4f} → "
            f"{float(e2['test_defective_recall']):.4f}), and Accuracy ({float(e1['test_accuracy']):.4f} → {float(e2['test_accuracy']):.4f}). "
            f"Using HOG alone yielded {float(e3['test_macro_f1']):.4f} Macro F1. LBP alone yielded "
            f"{float(e4['test_macro_f1']):.4f} with defective precision of {float(e4['test_defective_precision']):.4f}. "
            f"At the same 208-feature budget, the five random masks averaged {float(e5m['test_macro_f1']):.4f} Macro F1. "
            f"GA seed 42 retained all 10 LBP and 198 HOG dimensions (Figure 4). These comparisons measure entire feature subsets, "
            f"so the contribution of any particular bin remains uncertain.",
            body_style,
        )
    )

    fig4 = _scaled_rl_image(root / "results" / "figures" / "selected_features_breakdown.png", 16.6, 9.2)
    story.append(
        KeepTogether(
            [
                fig4,
                Paragraph(
                    "<b>Figure 4.</b> GA feature retention breakdown across HOG and LBP modalities (left) and Test Macro F1 / Defective Recall comparison across E1–E5 (right).",
                    caption_style,
                ),
            ]
        )
    )

    story.append(Paragraph("5.2 Multi-Seed Stability & Cost-Benefit Trade-Off", h2_style))
    t4_data = [["Seed", "Selected\n(HOG / LBP)", "Red.", "Best\nFitness", "Train\nCV F1", "Val\nF1", "Test\nMacro F1", "Def.\nRec", "Test\nAcc", "Search\nTime (s)"]]
    for run in art["ga_multiseed"]["runs"]:
        t4_data.append(
            [
                str(run["seed"]),
                f"{run['selected_features']} ({run['hog_features']}/{run['lbp_features']})",
                f"{run['feature_reduction_pct']:.2f}%",
                f"{run['best_fitness']:.4f}",
                f"{run['train_cv_macro_f1']:.4f}",
                f"{run['val_macro_f1']:.4f}",
                f"{run['test_macro_f1']:.4f}",
                f"{run['test_defective_recall']:.4f}",
                f"{run['test_accuracy']:.4f}",
                f"{run['search_time_s']:.2f}",
            ]
        )
    t4_data.append(
        [
            "Mean±Std",
            f"{ga_ms['selected_features']['mean']:.1f}±{ga_ms['selected_features']['std']:.1f}",
            f"{ga_ms['feature_reduction_pct']['mean']:.2f}%",
            f"{ga_ms['best_fitness']['mean']:.4f}",
            f"{ga_ms['train_cv_macro_f1']['mean']:.4f}",
            f"{ga_ms['val_macro_f1']['mean']:.4f}",
            f"{ga_ms['test_macro_f1']['mean']:.4f}±{ga_ms['test_macro_f1']['std']:.4f}",
            f"{ga_ms['test_defective_recall']['mean']:.4f}",
            f"{ga_ms['test_accuracy']['mean']:.4f}",
            f"{ga_ms['search_time_s']['mean']:.2f}±{ga_ms['search_time_s']['std']:.2f}",
        ]
    )
    story.append(
        KeepTogether(
            [
                Paragraph(
                    "<b>Table 4.</b> Multi-seed GA stability across random seeds [42, 123, 999] (results/ga_multiseed_runs.csv).",
                    caption_style,
                ),
                _styled_table(
                    t4_data,
                    [1.70 * cm, 2.45 * cm, 1.40 * cm, 1.55 * cm, 1.60 * cm, 1.35 * cm, 2.30 * cm, 1.50 * cm, 1.45 * cm, 1.90 * cm],
                    highlight_row=4,
                    font_name=sans,
                    bold_font=sans_b,
                    font_size=7.3,
                ),
                Spacer(1, 8),
            ]
        )
    )
    story.append(
        Paragraph(
            "The three GA runs (Table 4) averaged 0.7232 ± 0.0081 test Macro F1 and removed 42.31% ± 9.82% of features. "
            "Seed 999 achieved the highest training CV F1 (0.7518) with 155 features and reached 0.7191 on the held-out cells. "
            "Offline GA search took 4.95–8.18 s on the measured CPU runs. The measured batched classifier times were "
            "0.00137 and 0.00090 ms per cell for E1 and E2, respectively. The full feature vector is computed for both models, "
            "so feature extraction dominates the combined measurement (~1.28 ms per cell in batch).",
            body_style,
        )
    )

    story.append(Paragraph("5.3 Confusion Matrices, Subgroup Breakdown & Error Analysis", h2_style))
    story.append(
        Paragraph(
            f"Figure 5 shows that E2 improves both error types by one sample relative to E1 (TP: 72→73 [67.6%], TN: 180→181 [80.1%], "
            f"FP: 46→45 [19.9%], FN: 36→35 [32.4%]). Disaggregating test predictions by crystal type (Table 5) highlights that "
            f"polycrystalline cells drive nearly twice the error rate of monocrystalline cells ({ct['poly']['e2_error_rate']*100:.2f}% vs. "
            f"{ct['mono']['e2_error_rate']*100:.2f}%), and GA pruning lowers poly errors from {ct['poly']['e1_errors']} "
            f"({ct['poly']['e1_error_rate']*100:.2f}%) down to {ct['poly']['e2_errors']} ({ct['poly']['e2_error_rate']*100:.2f}%). "
            f"Across the {dis['total_disagreements']} test disagreements between E1 and E2, GA fixes {dis['e2_fixed_e1_error']} Baseline "
            f"errors (predominantly noisy poly grain-boundary false alarms such as #1576) while regressing on {dis['e2_regressed_vs_e1']}.",
            body_style,
        )
    )
    fig5 = _scaled_rl_image(root / "results" / "figures" / "confusion_matrices.png", 16.5, 8.8)
    story.append(
        KeepTogether(
            [
                fig5,
                Paragraph(
                    "<b>Figure 5.</b> Locked test confusion matrices (n=334) comparing E1 Baseline (334 features) and E2 GA-SVM (seed=42, 208 features).",
                    caption_style,
                ),
            ]
        )
    )

    t5_data = [
        ["Crystal Subgroup", "Test\nCells", "True Healthy /\nDefective", "E1 Errors\n(Rate)", "E2 Errors\n(Rate)", "E2\nTP / TN / FP / FN"],
        [
            "Monocrystalline (mono)",
            str(ct["mono"]["total"]),
            f"{ct['mono']['true_healthy']} / {ct['mono']['true_defective']}",
            f"{ct['mono']['e1_errors']} ({ct['mono']['e1_error_rate']*100:.2f}%)",
            f"{ct['mono']['e2_errors']} ({ct['mono']['e2_error_rate']*100:.2f}%)",
            f"{ct['mono']['e2_TP']} / {ct['mono']['e2_TN']} / {ct['mono']['e2_FP']} / {ct['mono']['e2_FN']}",
        ],
        [
            "Polycrystalline (poly)",
            str(ct["poly"]["total"]),
            f"{ct['poly']['true_healthy']} / {ct['poly']['true_defective']}",
            f"{ct['poly']['e1_errors']} ({ct['poly']['e1_error_rate']*100:.2f}%)",
            f"{ct['poly']['e2_errors']} ({ct['poly']['e2_error_rate']*100:.2f}%)",
            f"{ct['poly']['e2_TP']} / {ct['poly']['e2_TN']} / {ct['poly']['e2_FP']} / {ct['poly']['e2_FN']}",
        ],
        [
            "Combined Test Set",
            str(err["test_total"]),
            f"{ds['splits']['test']['healthy']} / {ds['splits']['test']['defective']}",
            f"{err['e1_total_errors']} (24.55%)",
            f"{err['e2_total_errors']} (23.95%)",
            f"{err['e2_outcome_counts']['TP']} / {err['e2_outcome_counts']['TN']} / {err['e2_outcome_counts']['FP']} / {err['e2_outcome_counts']['FN']}",
        ],
    ]
    story.append(
        KeepTogether(
            [
                Paragraph(
                    "<b>Table 5.</b> Subgroup error breakdown across monocrystalline and polycrystalline cells in the locked test split (results/sample_predictions/error_analysis_summary.json).",
                    caption_style,
                ),
                _styled_table(
                    t5_data,
                    [3.5 * cm, 2.2 * cm, 3.0 * cm, 2.6 * cm, 2.6 * cm, 3.1 * cm],
                    highlight_row=3,
                    font_name=sans,
                    bold_font=sans_b,
                    font_size=7.8,
                ),
                Spacer(1, 8),
            ]
        )
    )

    fig6 = _scaled_rl_image(root / "results" / "figures" / "error_analysis_grid.png", 16.0, 15.8)
    story.append(
        KeepTogether(
            [
                fig6,
                Paragraph(
                    "<b>Figure 6.</b> Qualitative E2 (GA-SVM, seed=42) error analysis showcasing representative TP (pronounced cracks #470, #467; dead cells #1614, #1979), TN (#1180, #832, #1873, #1880), FP (mono vignetting #8, #937; dark poly grains #123, #1709), and FN (fine border micro-crack #2070; low-contrast defects #850, #1550, #2155).",
                    caption_style,
                ),
            ]
        )
    )

    fig7 = _scaled_rl_image(root / "results" / "sample_predictions" / "demo_showcase.png", 16.5, 13.8)
    story.append(
        KeepTogether(
            [
                fig7,
                Paragraph(
                    "<b>Figure 7.</b> CLI prediction cards for defective (#470), healthy (#1180), and GA-corrected (#1576) cells. "
                    "The Flask demo presents the same model decisions beside the dataset label.",
                    caption_style,
                ),
            ]
        )
    )

    # Section 6
    story.append(Paragraph("6. Conclusion, Limitations & Future Work", h1_style))
    story.append(
        Paragraph(
            f"Across three runs, GA used {ga_ms['feature_reduction_pct']['mean']:.2f}% fewer features on average and "
            f"achieved test Macro F1 of {ga_ms['test_macro_f1']['mean']:.4f} ± {ga_ms['test_macro_f1']['std']:.4f}, "
            f"close to the all-feature baseline of {float(e1['test_macro_f1']):.4f} on the same {sp['test']['total']} cells. "
            f"The seed 42 model kept {int(float(e2['selected_features']))} features and made two fewer errors than the baseline. "
            f"The other two GA seeds scored below the baseline on this split. Evaluating performance on previously unseen modules "
            f"requires a module-grouped split or separate external data.",
            body_style,
        )
    )
    story.append(
        Paragraph(
            "<b>Methodological limitations.</b> The absence of module identifiers allows cells from a module to appear in different "
            "splits. The binary study also excludes 401 samples with intermediate defect labels. LinearSVC scores above zero indicate "
            "a defective prediction. Probability estimates would require separate calibration. Resizing to 64×64 can make fine cracks "
            "harder to resolve (Figure 6, FN #2070).",
            body_style,
        )
    )
    story.append(
        Paragraph(
            "<b>Future Work:</b> Extending descriptor computation to block-sparse HOG skipping pruned blocks at runtime, applying vignette "
            "normalization or multi-scale 8×8/16×16 HOG grids inside GA selection, calibrating probabilistic confidence scores on validation data, "
            "and evaluating module-disjoint external datasets.",
            body_style,
        )
    )

    # Section 7 References
    refs = [
        "[1] Buerhop-Lutz, C., Deitsch, S., Maier, A., Gallwitz, F., Berger, S., Doll, B., Hauch, J., Camus, C., & Brabec, C. J. (2018). A Benchmark for Visual Identification of Defective Solar Cells in Electroluminescence Imagery. 35th EU PVSEC. DOI: 10.4229/35thEUPVSEC20182018-5CV.3.15 (https://github.com/zae-bayern/elpv-dataset).",
        "[2] Deitsch, S., Christlein, V., Berger, S., Buerhop-Lutz, C., Maier, A., Gallwitz, F., & Riess, C. (2019). Automatic classification of defective photovoltaic module cells in electroluminescence images. Solar Energy, 185, 455–468. DOI: 10.1016/j.solener.2019.02.067.",
        "[3] Deitsch, S., Buerhop-Lutz, C., Sovetkin, E., Steland, A., Maier, A., Gallwitz, F., & Riess, C. (2021). Segmentation of photovoltaic module cells in uncalibrated electroluminescence images. Machine Vision and Applications, 32(4). DOI: 10.1007/s00138-021-01191-9.",
        "[4] Dalal, N., & Triggs, B. (2005). Histograms of Oriented Gradients for Human Detection. Proceedings of the IEEE CVPR, Vol. 1, pp. 886–893.",
        "[5] Ojala, T., Pietikäinen, M., & Mäenpää, T. (2002). Multiresolution gray-scale and rotation invariant texture classification with local binary patterns. IEEE TPAMI, 24(7), 971–987.",
        "[6] Cortes, C., & Vapnik, V. (1995). Support-vector networks. Machine Learning, 20(3), 273–297.",
        "[7] Siedlecki, W., & Sklansky, J. (1989). A note on genetic algorithms for large-scale feature selection. Pattern Recognition Letters, 10(5), 335–347.",
        "[8] van der Walt, S., Schönberger, J. L., Nunez-Iglesias, J., Boulogne, F., Warner, J. D., Yager, N., Gouillart, E., Yu, T., & scikit-image contributors (2014). scikit-image: image processing in Python. PeerJ, 2:e453. DOI: 10.7717/peerj.453.",
        "[9] Pedregosa, F., Varoquaux, G., Gramfort, A., Michel, V., Thirion, B., Grisel, O., et al. (2011). Scikit-learn: Machine Learning in Python. Journal of Machine Learning Research, 12(85), 2825–2830.",
    ]
    story.append(KeepTogether([
        Paragraph("7. References", h1_style),
        *(Paragraph(reference, ref_style) for reference in refs),
    ]))

    # Section 8 & 9
    story.append(Paragraph("8. Contribution Statement", h1_style))
    story.append(
        Paragraph(
            f"{author} ({student_id}) carried out the individual course project. The project work comprises the HOG and LBP "
            f"pipeline, genetic algorithm implementation, comparative evaluation, and working demonstration. AI assistance used "
            f"during development and revision is disclosed below.",
            body_style,
        )
    )

    story.append(Paragraph("9. AI Tool Use Statement", h1_style))
    story.append(
        Paragraph(
            "Antigravity / Gemini assisted with Python code scaffolding, refactoring evaluation plots, drafting documentation, "
            "and formatting the report. OpenAI Codex assisted with implementation review, reproduction of the reported experiments, "
            "revision of the report and slides, and development and testing of the Flask interface. The reported results "
            "come from the project scripts and recorded evaluation outputs.",
            body_style,
        )
    )

    doc.build(story, onFirstPage=_header_footer, onLaterPages=_header_footer)
    return out_pdf


def build_docx_report(
    root: Path,
    art: dict[str, Any],
    author: str,
    student_id: str,
    out_docx: Path,
) -> Path:
    """Generate an editable Word (.docx) companion report from the Markdown draft and figures."""
    out_docx.parent.mkdir(parents=True, exist_ok=True)
    doc = Document()

    for section in doc.sections:
        section.top_margin = Inches(0.8)
        section.bottom_margin = Inches(0.8)
        section.left_margin = Inches(0.85)
        section.right_margin = Inches(0.85)

    normal = doc.styles["Normal"]
    normal.font.name = "Calibri"
    normal.font.size = Pt(10.5)
    normal.paragraph_format.line_spacing = 1.04
    doc.core_properties.author = author

    p_title = doc.add_paragraph()
    p_title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r_title = p_title.add_run(
        "Intelligent Solar Cell Defect Classification Using Genetic Algorithm-Based Feature Selection"
    )
    r_title.bold = True
    r_title.font.size = Pt(15)
    r_title.font.color.rgb = RGBColor(0x0F, 0x17, 0x2A)

    p_meta = doc.add_paragraph()
    p_meta.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r_meta = p_meta.add_run(
        f"Course: SENG435 — Introduction to Digital Image Processing | Instructor: Assoc. Prof. Tuba ÇAĞLIKANTAR\n"
        f"Author: {author} | Student ID: {student_id} | Date: December 2026"
    )
    r_meta.font.size = Pt(9.5)
    r_meta.font.color.rgb = RGBColor(0x47, 0x55, 0x69)

    draft_path = root / "report" / "final_report_draft.md"
    lines = draft_path.read_text(encoding="utf-8").splitlines()

    def _add_md_table(tbl_lines: list[str]) -> None:
        parsed = []
        for raw in tbl_lines:
            cells = [c.strip().replace("`", "").replace("**", "") for c in raw.strip().strip("|").split("|")]
            parsed.append(cells)
        if len(parsed) < 2:
            return
        rows = [parsed[0]] + parsed[2:]
        table = doc.add_table(rows=len(rows), cols=len(rows[0]))
        table.style = "Table Grid"
        table.rows[0]._tr.get_or_add_trPr().append(OxmlElement("w:tblHeader"))
        margins = OxmlElement("w:tblCellMar")
        for side in ("left", "right"):
            margin = OxmlElement(f"w:{side}")
            margin.set(qn("w:w"), "48")
            margin.set(qn("w:type"), "dxa")
            margins.append(margin)
        table._tbl.tblPr.insert_element_before(margins, "w:tblLook")
        for r_i, row_vals in enumerate(rows):
            table.rows[r_i]._tr.get_or_add_trPr().insert_element_before(
                OxmlElement("w:cantSplit"), "w:tblHeader"
            )
            for c_i, val in enumerate(row_vals):
                cell = table.cell(r_i, c_i)
                cell.text = val
                for p in cell.paragraphs:
                    for run in p.runs:
                        run.font.size = Pt(8.5)
                        if r_i == 0:
                            run.bold = True

    idx = 0
    while idx < len(lines):
        line = lines[idx].strip()
        if idx < 8 or line == "---":
            idx += 1
            continue
        if line.startswith("## "):
            doc.add_heading(line[3:], level=1)
        elif line.startswith("### "):
            doc.add_heading(line[4:], level=2)
        elif line.startswith("!["):
            # Extract relative image path after ](
            start_p = line.find("](")
            end_p = line.find(")", start_p)
            if start_p != -1 and end_p != -1:
                rel = line[start_p + 2 : end_p].replace("../", "")
                img_file = root / rel
                if img_file.exists():
                    p_img = doc.add_paragraph()
                    p_img.alignment = WD_ALIGN_PARAGRAPH.CENTER
                    p_img.paragraph_format.keep_with_next = True
                    p_img.add_run().add_picture(str(img_file), width=Inches(6.2))
        elif line.startswith("|"):
            tbl_buf: list[str] = []
            while idx < len(lines) and lines[idx].strip().startswith("|"):
                tbl_buf.append(lines[idx])
                idx += 1
            _add_md_table(tbl_buf)
            doc.add_paragraph()
            continue
        elif line.startswith("$$") and line.endswith("$$"):
            p_eq = doc.add_paragraph()
            p_eq.alignment = WD_ALIGN_PARAGRAPH.CENTER
            run_eq = p_eq.add_run(line.strip("$ "))
            run_eq.italic = True
        elif line:
            clean = line.replace("`", "")
            p = doc.add_paragraph()
            p.paragraph_format.space_after = Pt(4)
            # Replace placeholder if CLI values provided
            clean = clean.replace("[Student Name — Student ID]", f"{author} — {student_id}")
            clean = clean.replace("**", "")
            p.add_run(clean)
        idx += 1

    doc.save(str(out_docx))
    return out_docx


def main() -> None:
    parser = argparse.ArgumentParser(description="Compile SENG435 final academic report to PDF and DOCX.")
    parser.add_argument("--author", type=str, default="[Student Name]", help="Author full name.")
    parser.add_argument("--student-id", type=str, default="[Student ID]", help="Student ID number.")
    args = parser.parse_args()

    artifacts = _load_verified_artifacts(PROJECT_ROOT)
    pdf_path = build_pdf_report(
        root=PROJECT_ROOT,
        art=artifacts,
        author=args.author,
        student_id=args.student_id,
        out_pdf=PROJECT_ROOT / "report" / "final_report.pdf",
    )
    docx_path = build_docx_report(
        root=PROJECT_ROOT,
        art=artifacts,
        author=args.author,
        student_id=args.student_id,
        out_docx=PROJECT_ROOT / "report" / "final_report.docx",
    )
    print(f"Generated PDF  : {pdf_path} ({pdf_path.stat().st_size} bytes)")
    print(f"Generated DOCX : {docx_path} ({docx_path.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
