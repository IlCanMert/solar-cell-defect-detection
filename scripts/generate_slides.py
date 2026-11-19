"""Generate a 10-slide landscape presentation PDF from verified experiment outputs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import yaml
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import (
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.generate_report import (  # noqa: E402
    _load_verified_artifacts,
    _register_fonts,
    _scaled_rl_image,
    _styled_table,
)

PAGE_W, PAGE_H = landscape(A4)
NAVY = colors.HexColor("#0f172a")
BLUE = colors.HexColor("#1e3a8a")
BLUE_SOFT = colors.HexColor("#eff6ff")
BLUE_BORDER = colors.HexColor("#bfdbfe")
GREEN_BG = colors.HexColor("#ecfdf5")
GREEN_BORDER = colors.HexColor("#a7f3d0")
AMBER_BG = colors.HexColor("#fffbeb")
AMBER_BORDER = colors.HexColor("#fde68a")
SLATE_BG = colors.HexColor("#f8fafc")
SLATE_BORDER = colors.HexColor("#cbd5e1")
TEXT_DARK = colors.HexColor("#111827")
TEXT_MUTED = colors.HexColor("#475569")


def _load_slide_extras(root: Path) -> dict[str, Any]:
    """Load experiment config and demo JSON alongside report artifacts."""
    with (root / "configs" / "experiment.yaml").open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    with (root / "results" / "sample_predictions" / "demo_output.json").open(
        "r", encoding="utf-8"
    ) as f:
        demo_out = json.load(f)
    return {"cfg": cfg, "demo": demo_out}


def _slide_chrome(canvas: Any, doc: Any) -> None:
    """Draw thin top accent bar and consistent slide footer with 1..10 numbering."""
    canvas.saveState()
    canvas.setFillColor(BLUE)
    canvas.rect(0, PAGE_H - 0.16 * cm, PAGE_W, 0.16 * cm, stroke=0, fill=1)

    canvas.setStrokeColor(SLATE_BORDER)
    canvas.setLineWidth(0.6)
    canvas.line(1.4 * cm, 1.08 * cm, PAGE_W - 1.4 * cm, 1.08 * cm)

    canvas.setFont("ReportSans", 8.2)
    canvas.setFillColor(TEXT_MUTED)
    canvas.drawString(
        1.4 * cm,
        0.68 * cm,
        "SENG435 Digital Image Processing | Intelligent Solar Cell Defect Classification (HOG + LBP + GA-SVM)",
    )
    canvas.setFont("ReportSansBold", 8.5)
    canvas.setFillColor(NAVY)
    canvas.drawRightString(PAGE_W - 1.4 * cm, 0.68 * cm, f"Slide {canvas.getPageNumber()} / 10")
    canvas.restoreState()


def _slide_header(
    section_tag: str,
    title: str,
    sans_b: str,
    sans: str,
) -> Table:
    """Create a uniform dark-navy slide header banner spanning 26.9 cm."""
    tag_style = ParagraphStyle(
        "SlideTag",
        fontName=sans_b,
        fontSize=8.5,
        leading=10.5,
        textColor=colors.HexColor("#93c5fd"),
    )
    title_style = ParagraphStyle(
        "SlideTitle",
        fontName=sans_b,
        fontSize=15.5,
        leading=19.0,
        textColor=colors.white,
    )
    tbl = Table(
        [[[Paragraph(section_tag.upper(), tag_style), Paragraph(title, title_style)]]],
        colWidths=[26.9 * cm],
    )
    tbl.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), NAVY),
                ("TOPPADDING", (0, 0), (-1, -1), 7),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
                ("LEFTPADDING", (0, 0), (-1, -1), 12),
                ("RIGHTPADDING", (0, 0), (-1, -1), 12),
                ("LINEBELOW", (0, 0), (-1, -1), 2.0, colors.HexColor("#3b82f6")),
            ]
        )
    )
    return tbl


def _callout_box(
    title_html: str,
    body_html: str,
    width_cm: float,
    sans: str,
    sans_b: str,
    bg_color: Any = BLUE_SOFT,
    border_color: Any = BLUE_BORDER,
    font_size: float = 9.5,
) -> Table:
    """Build a rounded-feel bordered callout card for key formulas or insights."""
    head_style = ParagraphStyle(
        "BoxHead",
        fontName=sans_b,
        fontSize=font_size + 0.5,
        leading=font_size + 3.2,
        textColor=NAVY,
        spaceAfter=3,
    )
    body_style = ParagraphStyle(
        "BoxBody",
        fontName=sans,
        fontSize=font_size,
        leading=font_size + 3.4,
        textColor=TEXT_DARK,
    )
    elems: list[Any] = []
    if title_html:
        elems.append(Paragraph(title_html, head_style))
    if body_html:
        elems.append(Paragraph(body_html, body_style))

    box = Table([[elems]], colWidths=[width_cm * cm])
    box.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), bg_color),
                ("BOX", (0, 0), (-1, -1), 0.8, border_color),
                ("TOPPADDING", (0, 0), (-1, -1), 6),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
                ("LEFTPADDING", (0, 0), (-1, -1), 9),
                ("RIGHTPADDING", (0, 0), (-1, -1), 9),
            ]
        )
    )
    return box


def _two_col(
    left_flowables: list[Any],
    right_flowables: list[Any],
    left_cm: float = 12.6,
    right_cm: float = 13.8,
) -> Table:
    """Arrange slide content in two crisp columns separated by a 0.5 cm gutter."""
    tbl = Table(
        [[left_flowables, "", right_flowables]],
        colWidths=[left_cm * cm, 0.5 * cm, right_cm * cm],
    )
    tbl.setStyle(
        TableStyle(
            [
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 0),
                ("RIGHTPADDING", (0, 0), (-1, -1), 0),
                ("TOPPADDING", (0, 0), (-1, -1), 0),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 0),
            ]
        )
    )
    return tbl


def _image_card(
    img_path: Path,
    caption: str,
    max_w_cm: float,
    max_h_cm: float,
    sans: str,
) -> Table:
    """Wrap a scaled PNG figure and concise caption in a clean bordered frame."""
    cap_style = ParagraphStyle(
        "SlideImgCap",
        fontName=sans,
        fontSize=8.3,
        leading=11.0,
        alignment=TA_CENTER,
        textColor=TEXT_MUTED,
        spaceBefore=4,
    )
    img = _scaled_rl_image(img_path, max_w_cm - 0.5, max_h_cm - 1.1)
    card = Table(
        [[[img, Paragraph(caption, cap_style)]]],
        colWidths=[max_w_cm * cm],
    )
    card.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), SLATE_BG),
                ("BOX", (0, 0), (-1, -1), 0.7, SLATE_BORDER),
                ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("TOPPADDING", (0, 0), (-1, -1), 6),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
                ("LEFTPADDING", (0, 0), (-1, -1), 5),
                ("RIGHTPADDING", (0, 0), (-1, -1), 5),
            ]
        )
    )
    return card


def build_slides_pdf(
    root: Path,
    art: dict[str, Any],
    extras: dict[str, Any],
    author: str,
    student_id: str,
    out_pdf: Path,
) -> Path:
    """Compile the 10-slide landscape presentation deck PDF."""
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    _, _, _, sans, sans_b = _register_fonts()

    bullet_style = ParagraphStyle(
        "SlideBullet",
        fontName=sans,
        fontSize=10.3,
        leading=14.0,
        alignment=TA_LEFT,
        leftIndent=11,
        firstLineIndent=-9,
        textColor=TEXT_DARK,
        spaceAfter=5,
    )
    subhead_style = ParagraphStyle(
        "SlideSubhead",
        fontName=sans_b,
        fontSize=11.2,
        leading=14.5,
        textColor=BLUE,
        spaceBefore=2,
        spaceAfter=4,
    )
    body_style = ParagraphStyle(
        "SlideBody",
        fontName=sans,
        fontSize=10.2,
        leading=13.8,
        textColor=TEXT_DARK,
        spaceAfter=5,
    )
    code_style = ParagraphStyle(
        "SlideCode",
        fontName="Courier",
        fontSize=9.2,
        leading=12.6,
        textColor=colors.HexColor("#f8fafc"),
    )

    ds = art["dataset"]
    fs = art["features"]
    cfg = extras["cfg"]
    ga_hist = art["ga_history"]
    ga_ms_runs = art["ga_multiseed"]["runs"]
    ga_ms = art["ga_multiseed"]["mean_std"]
    m = art["metrics"]
    e1 = m["E1"]
    e2 = m["E2_seed42"]
    e2m = m["E2_3seed_mean"]
    e3 = m["E3"]
    e4 = m["E4"]
    e5s = m["E5_seed42"]
    e5m = m["E5_5seed_mean"]
    err = art["error_summary"]
    ct = err["cell_type_breakdown"]
    dis = err["e1_vs_e2_disagreements"]
    demo_preds = extras["demo"]["predictions"]
    val_top = art["val_grid"][0]
    val_weak = next(
        (r for r in art["val_grid"] if r["class_weight"] == "balanced" and str(r["C"]) == "1.0"),
        art["val_grid"][-1],
    )

    doc = SimpleDocTemplate(
        str(out_pdf),
        pagesize=landscape(A4),
        leftMargin=1.4 * cm,
        rightMargin=1.4 * cm,
        topMargin=0.95 * cm,
        bottomMargin=1.35 * cm,
        title="SENG435 Solar Cell Defect Classification Presentation",
        author=f"{author} ({student_id})",
    )

    story: list[Any] = []

    # ------------------------------------------------------------------
    # SLIDE 1: Title & Problem Scope
    # ------------------------------------------------------------------
    cover_kicker = ParagraphStyle(
        "CoverKicker",
        fontName=sans_b,
        fontSize=10.5,
        leading=14.0,
        alignment=TA_CENTER,
        textColor=colors.HexColor("#93c5fd"),
        spaceAfter=8,
    )
    cover_title = ParagraphStyle(
        "CoverTitle",
        fontName=sans_b,
        fontSize=24.0,
        leading=29.5,
        alignment=TA_CENTER,
        textColor=colors.white,
        spaceAfter=10,
    )
    cover_sub = ParagraphStyle(
        "CoverSub",
        fontName=sans,
        fontSize=12.0,
        leading=16.5,
        alignment=TA_CENTER,
        textColor=colors.HexColor("#e2e8f0"),
    )
    hero = Table(
        [
            [
                [
                    Paragraph("SENG435 — INTRODUCTION TO DIGITAL IMAGE PROCESSING", cover_kicker),
                    Paragraph(
                        "Intelligent Solar Cell Defect Classification<br/>Using Genetic Algorithm-Based Feature Selection",
                        cover_title,
                    ),
                    Paragraph(
                        "ELPV solar cell images classified with HOG, LBP, and a linear SVM",
                        cover_sub,
                    ),
                ]
            ]
        ],
        colWidths=[26.9 * cm],
    )
    hero.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), NAVY),
                ("TOPPADDING", (0, 0), (-1, -1), 24),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 24),
                ("LEFTPADDING", (0, 0), (-1, -1), 22),
                ("RIGHTPADDING", (0, 0), (-1, -1), 22),
                ("LINEBELOW", (0, 0), (-1, -1), 3.0, colors.HexColor("#3b82f6")),
            ]
        )
    )
    story.append(hero)
    story.append(Spacer(1, 14))

    kpi_style_num = ParagraphStyle(
        "KpiNum", fontName=sans_b, fontSize=18.5, leading=22.0, alignment=TA_CENTER, textColor=BLUE
    )
    kpi_style_lbl = ParagraphStyle(
        "KpiLbl", fontName=sans_b, fontSize=9.8, leading=13.0, alignment=TA_CENTER, textColor=NAVY
    )
    kpi_style_sub = ParagraphStyle(
        "KpiSub", fontName=sans, fontSize=8.8, leading=12.0, alignment=TA_CENTER, textColor=TEXT_MUTED
    )

    def _kpi_card(num: str, lbl: str, sub: str) -> Table:
        c = Table(
            [[[Paragraph(num, kpi_style_num), Paragraph(lbl, kpi_style_lbl), Paragraph(sub, kpi_style_sub)]]],
            colWidths=[8.6 * cm],
        )
        c.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, -1), SLATE_BG),
                    ("BOX", (0, 0), (-1, -1), 0.9, BLUE_BORDER),
                    ("TOPPADDING", (0, 0), (-1, -1), 10),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 10),
                    ("LEFTPADDING", (0, 0), (-1, -1), 8),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 8),
                ]
            )
        )
        return c

    kpis = Table(
        [
            [
                _kpi_card(
                    f"{ds['binary_subset_total']:,} Cells",
                    "Definite Binary ELPV Subset",
                    f"{ds['binary_healthy']:,} Healthy / {ds['binary_defective']} Defective (mono + poly)",
                ),
                "",
                _kpi_card(
                    f"{ga_ms['feature_reduction_pct']['mean']:.2f}% fewer",
                    "Features retained after GA selection",
                    "Average reduction across three GA runs",
                ),
                "",
                _kpi_card(
                    f"{ga_ms['test_macro_f1']['mean']:.4f} Macro F1",
                    "Mean across three GA runs",
                    f"All-feature SVM {float(e1['test_macro_f1']):.4f} on the same test cells",
                ),
            ]
        ],
        colWidths=[8.6 * cm, 0.55 * cm, 8.6 * cm, 0.55 * cm, 8.6 * cm],
    )
    kpis.setStyle(TableStyle([("LEFTPADDING", (0, 0), (-1, -1), 0), ("RIGHTPADDING", (0, 0), (-1, -1), 0)]))
    story.append(kpis)
    story.append(Spacer(1, 14))

    meta_box = _callout_box(
        "Project and presenter",
        f"<b>Presenter:</b> {author} &nbsp;|&nbsp; <b>Student ID:</b> {student_id} &nbsp;|&nbsp; "
        f"<b>Instructor:</b> Assoc. Prof. Tuba \u00c7A\u011eLIKANTAR &nbsp;|&nbsp; <b>Date:</b> December 2026<br/>"
        "Does selecting fewer image features preserve classification performance on the held-out cells?",
        26.9,
        sans,
        sans_b,
        bg_color=BLUE_SOFT,
        border_color=BLUE_BORDER,
        font_size=10.0,
    )
    story.append(meta_box)
    story.append(PageBreak())

    # ------------------------------------------------------------------
    # SLIDE 2: Dataset & Strict Anti-Leakage Protocol
    # ------------------------------------------------------------------
    story.append(_slide_header("1. Dataset & Protocol", "ELPV binary labels and held-out cell-level evaluation", sans_b, sans))
    story.append(Spacer(1, 10))

    sp_tr = ds["splits"]["train"]
    sp_va = ds["splits"]["val"]
    sp_te = ds["splits"]["test"]
    split_tbl_data = [
        ["Partition", "Total", "Healthy", "Defective", "Mono", "Poly", "Role"],
        [
            "Train (70%)",
            f"{sp_tr['total']:,}",
            f"{sp_tr['healthy']:,}",
            f"{sp_tr['defective']}",
            f"{sp_tr['mono']}",
            f"{sp_tr['poly']}",
            "Tuning + GA 3-fold CV",
        ],
        [
            "Validation (15%)",
            f"{sp_va['total']}",
            f"{sp_va['healthy']}",
            f"{sp_va['defective']}",
            f"{sp_va['mono']}",
            f"{sp_va['poly']}",
            "Regularization C selection",
        ],
        [
            "Locked Test (15%)",
            f"{sp_te['total']}",
            f"{sp_te['healthy']}",
            f"{sp_te['defective']}",
            f"{sp_te['mono']}",
            f"{sp_te['poly']}",
            "Final E1-E5 eval only",
        ],
        [
            "Binary Total",
            f"{ds['binary_subset_total']:,}",
            f"{ds['binary_healthy']:,}",
            f"{ds['binary_defective']}",
            f"{sp_tr['mono'] + sp_va['mono'] + sp_te['mono']:,}",
            f"{sp_tr['poly'] + sp_va['poly'] + sp_te['poly']:,}",
            "Frozen data/splits.csv",
        ],
    ]
    split_tbl = _styled_table(
        split_tbl_data,
        [2.45 * cm, 1.1 * cm, 1.35 * cm, 1.45 * cm, 1.15 * cm, 1.15 * cm, 3.85 * cm],
        highlight_row=3,
        font_name=sans,
        bold_font=sans_b,
        font_size=7.7,
    )

    s2_left = [
        Paragraph("Definite Binary Subset & Stratification", subhead_style),
        Paragraph(
            f"&bull; <b>Raw ELPV Corpus:</b> {ds['raw_total_images']:,} grayscale 300&times;300 EL cells "
            f"(1,074 mono, 1,550 poly) with consensus labels {{0.0, 0.3333, 0.6667, 1.0}}.",
            bullet_style,
        ),
        Paragraph(
            f"&bull; <b>Clean Subset ({ds['binary_subset_total']:,}):</b> Retained definite 0.0 HEALTHY ({ds['binary_healthy']:,}) "
            f"&amp; 1.0 DEFECTIVE ({ds['binary_defective']}); excluded 401 ambiguous consensus cells.",
            bullet_style,
        ),
        Paragraph(
            "&bull; <b>Joint Stratification:</b> Frozen 70/15/15 split stratified by <i>(label &times; cell_type)</i> (seed=42).",
            bullet_style,
        ),
        Spacer(1, 5),
        split_tbl,
        Spacer(1, 8),
        _callout_box(
            "Model fitting and evaluation",
            "&bull; <b>StandardScaler inside Pipeline:</b> Mean/variance learned strictly inside Train/CV folds.<br/>"
            "&bull; <b>Locked Test Isolation:</b> Test cells never entered Grid Search or GA fitness scoring.<br/>"
            f"&bull; <b>Module-ID Limitation:</b> <i>has_module_id={str(ds['has_module_id']).lower()}</i> in ELPV metadata; "
            "cells from the same module may appear in different splits.",
            12.5,
            sans,
            sans_b,
            bg_color=AMBER_BG,
            border_color=AMBER_BORDER,
            font_size=8.8,
        ),
    ]
    s2_right = [
        _image_card(
            root / "results" / "figures" / "dataset_samples.png",
            "Figure 1. Representative 300x300 ELPV images across Healthy vs. Defective and Mono vs. Poly wafers.",
            13.8,
            15.6,
            sans,
        )
    ]
    story.append(_two_col(s2_left, s2_right, left_cm=12.6, right_cm=13.8))
    story.append(PageBreak())

    # ------------------------------------------------------------------
    # SLIDE 3: Classical Image Processing Features (HOG + LBP)
    # ------------------------------------------------------------------
    story.append(_slide_header("2. Image Processing Methodology", "Handcrafted HOG + Uniform LBP Representation (334 Dims)", sans_b, sans))
    story.append(Spacer(1, 10))

    hog_cfg = cfg["features"]["hog"]
    lbp_cfg = cfg["features"]["lbp"]
    s3_left = [
        Paragraph("Complementary Spatial Edge & Micro-Texture Cues", subhead_style),
        Paragraph(
            "&bull; <b>Standardized Resize:</b> Every 300&times;300 raw EL wafer cell is downsampled cleanly to 64&times;64 grayscale.",
            bullet_style,
        ),
        _callout_box(
            f"1. Histogram of Oriented Gradients — {fs['hog_dim']} Features (Indices 0–323)",
            f"&bull; Parameters: orientations={hog_cfg['orientations']}, pixels_per_cell=(16, 16), "
            f"cells_per_block=(2, 2), block_norm='{hog_cfg['block_norm']}'.<br/>"
            "&bull; Produces 3&times;3 overlapping blocks &times; (2&times;2&times;9 = 36 bins) = <b>324 structural descriptors</b>.<br/>"
            "&bull; Captures sharp directional gradients caused by hairline cracks, finger interruptions, and busbars.",
            12.5,
            sans,
            sans_b,
            bg_color=BLUE_SOFT,
            border_color=BLUE_BORDER,
            font_size=9.1,
        ),
        Spacer(1, 6),
        _callout_box(
            f"2. Uniform Local Binary Patterns — {fs['lbp_dim']} Features (Indices 324–333)",
            f"&bull; Parameters: P={lbp_cfg['P']} circular neighbors, radius R={lbp_cfg['R']}, method='{lbp_cfg['method']}'.<br/>"
            "&bull; L1-normalized <b>10-bin probability histogram</b> summarizing isotropic surface intensity patterns.<br/>"
            "&bull; Detects uniform inactive dark spots, shunts, and polycrystalline grain texture changes.",
            12.5,
            sans,
            sans_b,
            bg_color=GREEN_BG,
            border_color=GREEN_BORDER,
            font_size=9.1,
        ),
        Spacer(1, 6),
        _callout_box(
            "CPU Throughput Verification",
            f"&bull; Concatenated <b>{fs['total_dim']}-dim</b> matrix extracted across all {ds['binary_subset_total']:,} cells "
            f"in <b>{fs['extraction_time_s']:.4f} s</b> (<b>{fs['ms_per_image']:.4f} ms/image</b>), zero NaN/Inf.",
            12.5,
            sans,
            sans_b,
            bg_color=SLATE_BG,
            border_color=SLATE_BORDER,
            font_size=9.0,
        ),
    ]
    s3_right = [
        _image_card(
            root / "results" / "figures" / "feature_visualization.png",
            "Figure 2. Raw cell, 64x64 HOG gradients (324), Uniform LBP code map, and 10-bin histogram for healthy cell0004 (top) vs defective cracked cell0001 (bottom).",
            13.8,
            15.6,
            sans,
        )
    ]
    story.append(_two_col(s3_left, s3_right, left_cm=12.6, right_cm=13.8))
    story.append(PageBreak())

    # ------------------------------------------------------------------
    # SLIDE 4: Intelligent Technique: GA-Based Feature Selection
    # ------------------------------------------------------------------
    story.append(_slide_header("3. Intelligent Optimization", "Parsimony-Penalized Wrapper Genetic Algorithm (GA)", sans_b, sans))
    story.append(Spacer(1, 10))

    ga_cfg = cfg["ga"]
    ga_params_tbl = _styled_table(
        [
            ["GA Parameter", "Configured Value"],
            ["Chromosome Encoding", "334-bit binary mask m in {0, 1}^334"],
            ["Objective Scoring", "Train-only 3-fold Stratified CV Macro F1"],
            ["Population / Generations", f"{ga_cfg['population_size']} individuals \u00d7 {ga_cfg['generations']} generations"],
            ["Selection / Elitism", f"Tournament (k={ga_cfg['tournament_size']}) + Elitism ({ga_cfg['elitism_count']} unique best)"],
            ["Crossover / Mutation", f"Two-point (p={ga_cfg['crossover_prob']}) / Bit-flip (p={ga_cfg['mutation_prob']:.6f} ~ 1/334)"],
            ["Memoization & Speed", f"{ga_hist['n_cache_hits']} cache hits / {ga_hist['n_unique_evaluations']} unique evals in {ga_hist['search_time_s']:.2f} s"],
        ],
        [4.8 * cm, 7.6 * cm],
        highlight_row=6,
        font_name=sans,
        bold_font=sans_b,
        font_size=8.3,
    )

    s4_left = [
        Paragraph("Wrapper Formulation & Search Mechanics", subhead_style),
        _callout_box(
            "Penalized Fitness Objective (Train Split Only)",
            "<b>Fitness(<i>m</i>) = mean_macro_F1<sub>CV</sub>(<i>m</i>) &minus; &lambda; &middot; (||<i>m</i>||<sub>1</sub> / 334)</b>, "
            f"&nbsp;where <b>&lambda; = {ga_cfg['lambda_penalty']}</b><br/>"
            "Simultaneously rewards Macro F1 across imbalanced classes and removes redundant or noisy HOG blocks.",
            12.5,
            sans,
            sans_b,
            bg_color=BLUE_SOFT,
            border_color=BLUE_BORDER,
            font_size=9.4,
        ),
        Spacer(1, 8),
        ga_params_tbl,
        Spacer(1, 8),
        _callout_box(
            "Seed=42 Convergence Outcome",
            f"&bull; Best penalized fitness climbed smoothly to <b>{float(ga_hist['best_eval']['fitness']):.4f}</b>.<br/>"
            f"&bull; Train 3-fold CV Macro F1 improved from Baseline <b>{float(e1['train_cv_macro_f1']):.4f}</b> to "
            f"<b>{float(e2['train_cv_macro_f1']):.4f}</b> while pruning 126 redundant HOG bins.",
            12.5,
            sans,
            sans_b,
            bg_color=GREEN_BG,
            border_color=GREEN_BORDER,
            font_size=9.1,
        ),
    ]
    s4_right = [
        _image_card(
            root / "results" / "figures" / "ga_convergence.png",
            "Figure 3. GA convergence (seed=42): penalized fitness & train CV Macro F1 (left) and active feature count pruning down to 208/334 (right).",
            13.8,
            15.6,
            sans,
        )
    ]
    story.append(_two_col(s4_left, s4_right, left_cm=12.6, right_cm=13.8))
    story.append(PageBreak())

    # ------------------------------------------------------------------
    # SLIDE 5: Selected Feature Breakdown (208 / 334)
    # ------------------------------------------------------------------
    story.append(_slide_header("4. Feature Modality Insight", "Selected Feature Breakdown: Why GA Retained 10/10 Uniform LBP", sans_b, sans))
    story.append(Spacer(1, 10))

    feat_break_tbl = _styled_table(
        [
            ["Feature Subset", "HOG Kept (/324)", "LBP Kept (/10)", "Total (/334)", "Reduction"],
            ["E1 Baseline (All)", "324 (100.0%)", "10 (100.0%)", "334", "0.00%"],
            [
                "E2 GA-SVM (seed=42)",
                f"{int(float(e2['hog_features']))} ({int(float(e2['hog_features']))/324*100:.1f}%)",
                f"{int(float(e2['lbp_features']))} (100.0%)",
                f"{int(float(e2['selected_features']))}",
                f"{float(e2['feature_reduction_pct']):.2f}%",
            ],
            [
                "E2 GA (3-Seed Mean)",
                f"{float(e2m['hog_features']):.1f} ({float(e2m['hog_features'])/324*100:.1f}%)",
                f"{float(e2m['lbp_features']):.1f} ({float(e2m['lbp_features'])/10*100:.1f}%)",
                f"{float(e2m['selected_features']):.1f}",
                f"{float(e2m['feature_reduction_pct']):.2f}%",
            ],
        ],
        [3.3 * cm, 2.7 * cm, 2.4 * cm, 2.0 * cm, 2.0 * cm],
        highlight_row=2,
        font_name=sans,
        bold_font=sans_b,
        font_size=8.3,
    )

    s5_left = [
        Paragraph("Structural vs. Textural Modality Retention", subhead_style),
        feat_break_tbl,
        Spacer(1, 8),
        Paragraph(
            f"&bull; <b>Unconstrained Discovery:</b> GA received zero prior weighting between indices 0–323 (HOG) "
            f"and 324–333 (LBP), yet kept <b>100% (10/10) of Uniform LBP bins</b> on seed=42 (mean 8.0/10 across seeds).",
            bullet_style,
        ),
        Paragraph(
            "&bull; <b>LBP retention:</b> Seed 42 retained all ten texture bins. The three-run average was eight. "
            "The effect of individual bins would require a separate experiment.",
            bullet_style,
        ),
        Paragraph(
            "&bull; <b>HOG retention:</b> Seed 42 selected 198 of 324 edge features. The smaller input vector "
            "reduces the number of linear SVM weights.",
            bullet_style,
        ),
        Spacer(1, 5),
        _callout_box(
            "Takeaway for Feature Engineering",
            "Handcrafted multi-descriptor fusion benefits asymmetrically from evolutionary pruning: "
            "high-dimensional directional histograms (HOG) compress cleanly, while compact histogram textures (LBP) remain dense.",
            12.5,
            sans,
            sans_b,
            bg_color=GREEN_BG,
            border_color=GREEN_BORDER,
            font_size=9.1,
        ),
    ]
    s5_right = [
        _image_card(
            root / "results" / "figures" / "selected_features_breakdown.png",
            "Figure 4. Retained vs. pruned counts across HOG blocks and Uniform LBP bins under GA feature selection (seed=42).",
            13.8,
            15.6,
            sans,
        )
    ]
    story.append(_two_col(s5_left, s5_right, left_cm=12.6, right_cm=13.8))
    story.append(PageBreak())

    # ------------------------------------------------------------------
    # SLIDE 6: Baseline Tuning & Master E1–E5 Comparison
    # ------------------------------------------------------------------
    story.append(_slide_header("5. Quantitative Evaluation", "Regularization Tuning & Locked Test Set Benchmark (E1–E5)", sans_b, sans))
    story.append(Spacer(1, 8))

    story.append(
        _callout_box(
            "Validation Hyperparameter Tuning (Frozen Split, Test Excluded)",
            f"Sweep over C in {{0.001, 0.01, 0.1, 1.0, 10.0}} and class_weight in {{None, 'balanced'}}: "
            f"<b>class_weight='balanced', C={val_top['C']}</b> raised Val Macro F1 from <b>{float(val_weak['val_macro_f1']):.4f}</b> (C=1.0) to "
            f"<b>{float(val_top['val_macro_f1']):.4f}</b> (Train CV {float(val_top['cv_macro_f1_mean']):.4f} &plusmn; {float(val_top['cv_macro_f1_std']):.4f}, "
            f"Val Defective Recall <b>{float(val_top['val_defective_recall']):.4f}</b>). Identical hyperparameters froze E1–E5.",
            26.9,
            sans,
            sans_b,
            bg_color=BLUE_SOFT,
            border_color=BLUE_BORDER,
            font_size=9.2,
        )
    )
    story.append(Spacer(1, 8))

    def _m_row(exp_label: str, row: dict[str, Any], time_str: str) -> list[str]:
        feats_str = (
            f"{int(float(row['selected_features']))} ({int(float(row['hog_features']))}/{int(float(row['lbp_features']))})"
            if float(row["selected_features"]).is_integer()
            else f"{float(row['selected_features']):.1f} ({float(row['hog_features']):.1f}/{float(row['lbp_features']):.1f})"
        )
        return [
            exp_label,
            feats_str,
            f"{float(row['feature_reduction_pct']):.2f}%",
            f"{float(row['train_cv_macro_f1']):.4f}",
            f"{float(row['val_macro_f1']):.4f}",
            f"{float(row['test_macro_f1']):.4f}",
            f"{float(row['test_defective_f1']):.4f}",
            f"{float(row['test_defective_precision']):.4f}",
            f"{float(row['test_defective_recall']):.4f}",
            f"{float(row['test_accuracy']):.4f}",
            time_str,
        ]

    master_tbl_data = [
        [
            "Experiment",
            "Feats (HOG/LBP)",
            "Red.",
            "Train CV",
            "Val F1",
            "Test F1",
            "Def F1",
            "Def Prec",
            "Def Rec",
            "Test Acc",
            "Search/Fit",
        ],
        _m_row("E1 Baseline (All)", e1, f"{float(e1['fit_time_s']):.4f}s fit"),
        _m_row("E2 GA-SVM (seed=42)", e2, f"{float(e2['train_or_search_time_s']):.2f}s GA"),
        _m_row("E2 GA (3-Seed Mean)", e2m, f"{float(e2m['train_or_search_time_s']):.2f}s GA"),
        _m_row("E3 HOG-Only", e3, f"{float(e3['fit_time_s']):.4f}s fit"),
        _m_row("E4 LBP-Only", e4, f"{float(e4['fit_time_s']):.4f}s fit"),
        _m_row("E5 Random-208 (s=42)", e5s, f"{float(e5s['fit_time_s']):.4f}s fit"),
        _m_row("E5 Random-208 (5-Mean)", e5m, f"{float(e5m['fit_time_s']):.4f}s fit"),
    ]
    master_tbl = _styled_table(
        master_tbl_data,
        [4.1 * cm, 2.85 * cm, 1.5 * cm, 1.95 * cm, 1.85 * cm, 1.95 * cm, 1.95 * cm, 2.1 * cm, 2.05 * cm, 2.1 * cm, 2.5 * cm],
        highlight_row=2,
        font_name=sans,
        bold_font=sans_b,
        font_size=8.4,
    )
    story.append(master_tbl)
    story.append(Spacer(1, 10))

    takeaways = Table(
        [
            [
                _callout_box(
                    "1. Best Single-Run Generalization",
                    f"<b>E2 GA-SVM (seed=42)</b> leads on <i>every</i> test metric: Macro F1 <b>{float(e2['test_macro_f1']):.4f}</b> "
                    f"(vs E1 {float(e1['test_macro_f1']):.4f}), Defective F1 <b>{float(e2['test_defective_f1']):.4f}</b>, and Accuracy "
                    f"<b>{float(e2['test_accuracy']):.4f}</b> using 37.72% fewer features.",
                    8.6,
                    sans,
                    sans_b,
                    bg_color=GREEN_BG,
                    border_color=GREEN_BORDER,
                    font_size=8.9,
                ),
                "",
                _callout_box(
                    "2. Beats Random-208 Baseline",
                    f"Matched-size <b>E5 Random-208 (5-mean)</b> drops to <b>{float(e5m['test_macro_f1']):.4f}</b> Macro F1 and "
                    f"<b>{float(e5m['test_accuracy']):.4f}</b> accuracy on the same test cells.",
                    8.6,
                    sans,
                    sans_b,
                    bg_color=SLATE_BG,
                    border_color=SLATE_BORDER,
                    font_size=8.9,
                ),
                "",
                _callout_box(
                    "3. Fusion Fixes False Alarms",
                    f"LBP-Only (E4) achieves high Defective Recall (<b>{float(e4['test_defective_recall']):.4f}</b>) only with poor "
                    f"Precision (<b>{float(e4['test_defective_precision']):.4f}</b>); combining HOG + LBP resolves false positives cleanly.",
                    8.6,
                    sans,
                    sans_b,
                    bg_color=AMBER_BG,
                    border_color=AMBER_BORDER,
                    font_size=8.9,
                ),
            ]
        ],
        colWidths=[8.6 * cm, 0.55 * cm, 8.6 * cm, 0.55 * cm, 8.6 * cm],
    )
    takeaways.setStyle(TableStyle([("LEFTPADDING", (0, 0), (-1, -1), 0), ("RIGHTPADDING", (0, 0), (-1, -1), 0)]))
    story.append(takeaways)
    story.append(PageBreak())

    # ------------------------------------------------------------------
    # SLIDE 7: Multi-Seed Stability & Honest Trade-off Analysis
    # ------------------------------------------------------------------
    story.append(_slide_header("6. Robustness Audit", "GA Multi-Seed Stability (42, 123, 999) & Compression Trade-Off", sans_b, sans))
    story.append(Spacer(1, 10))

    ms_rows = [
        ["GA Seed", "Feats (HOG/LBP)", "Reduction", "Fitness", "Train CV F1", "Test Macro F1", "Def Rec", "Test Acc", "Search (s)"]
    ]
    for r in ga_ms_runs:
        ms_rows.append(
            [
                f"seed = {r['seed']}",
                f"{r['selected_features']} ({r['hog_features']}/{r['lbp_features']})",
                f"{r['feature_reduction_pct']:.2f}%",
                f"{r['best_fitness']:.4f}",
                f"{r['train_cv_macro_f1']:.4f}",
                f"{r['test_macro_f1']:.4f}",
                f"{r['test_defective_recall']:.4f}",
                f"{r['test_accuracy']:.4f}",
                f"{r['search_time_s']:.2f}",
            ]
        )
    ms_rows.append(
        [
            "Mean \u00b1 Std",
            f"{ga_ms['selected_features']['mean']:.1f}\n(\u00b1{ga_ms['selected_features']['std']:.1f})",
            f"{ga_ms['feature_reduction_pct']['mean']:.2f}%\n(\u00b1{ga_ms['feature_reduction_pct']['std']:.2f}%)",
            f"{ga_ms['best_fitness']['mean']:.4f}\n(\u00b1{ga_ms['best_fitness']['std']:.4f})",
            f"{ga_ms['train_cv_macro_f1']['mean']:.4f}",
            f"{ga_ms['test_macro_f1']['mean']:.4f}\n(\u00b1{ga_ms['test_macro_f1']['std']:.4f})",
            f"{ga_ms['test_defective_recall']['mean']:.4f}",
            f"{ga_ms['test_accuracy']['mean']:.4f}",
            f"{ga_ms['search_time_s']['mean']:.2f}",
        ]
    )
    ms_tbl = _styled_table(
        ms_rows,
        [2.2 * cm, 2.75 * cm, 2.35 * cm, 2.05 * cm, 1.95 * cm, 2.4 * cm, 1.7 * cm, 1.75 * cm, 1.75 * cm],
        highlight_row=4,
        font_name=sans,
        bold_font=sans_b,
        font_size=8.1,
    )

    s7_left = [
        Paragraph("Independent GA Runs Across Random Seeds", subhead_style),
        ms_tbl,
        Spacer(1, 8),
        _callout_box(
            "Feature reduction across three runs",
            f"The mean test Macro F1 was <b>{ga_ms['test_macro_f1']['mean']:.4f} &plusmn; {ga_ms['test_macro_f1']['std']:.4f}</b> "
            f"versus <b>{float(e1['test_macro_f1']):.4f}</b> for the all-feature baseline. The GA runs used "
            f"<b>{ga_ms['feature_reduction_pct']['mean']:.2f}%</b> fewer features on average.",
            18.9,
            sans,
            sans_b,
            bg_color=AMBER_BG,
            border_color=AMBER_BORDER,
            font_size=9.2,
        ),
    ]
    s7_right = [
        _callout_box(
            "Training fitness and test performance",
            "Seed 999 had the highest training fitness (0.7285) with 155 features. Its test Macro F1 was 0.7191. "
            "Seed 42 kept 208 features and reached 0.7325 on the same test cells.",
            7.5,
            sans,
            sans_b,
            bg_color=BLUE_SOFT,
            border_color=BLUE_BORDER,
            font_size=9.0,
        )
    ]
    story.append(_two_col(s7_left, s7_right, left_cm=18.9, right_cm=7.5))
    story.append(PageBreak())

    # ------------------------------------------------------------------
    # SLIDE 8: Confusion Matrices & Error Analysis (Mono vs Poly)
    # ------------------------------------------------------------------
    story.append(_slide_header("7. Qualitative Error Analysis", "Confusion Matrices & Monocrystalline vs. Polycrystalline Errors", sans_b, sans))
    story.append(Spacer(1, 8))

    e1_cm = err["e1_outcome_counts"]
    e2_cm = err["e2_outcome_counts"]
    ct_mono = ct["mono"]
    ct_poly = ct["poly"]

    err_tbl = _styled_table(
        [
            ["Subgroup (Test)", "N (H / D)", "E1 Errors (%)", "E2 Errors (%)", "Net Change"],
            [
                "Mono ('mono')",
                f"{ct_mono['total']} ({ct_mono['true_healthy']}/{ct_mono['true_defective']})",
                f"{ct_mono['e1_errors']} ({ct_mono['e1_error_rate']*100:.2f}%)",
                f"{ct_mono['e2_errors']} ({ct_mono['e2_error_rate']*100:.2f}%)",
                "+4 errors",
            ],
            [
                "Poly ('poly')",
                f"{ct_poly['total']} ({ct_poly['true_healthy']}/{ct_poly['true_defective']})",
                f"{ct_poly['e1_errors']} ({ct_poly['e1_error_rate']*100:.2f}%)",
                f"{ct_poly['e2_errors']} ({ct_poly['e2_error_rate']*100:.2f}%)",
                "-6 errors (improved)",
            ],
            [
                "All Locked Test",
                f"{err['test_total']} (226/108)",
                f"{err['e1_total_errors']} (24.55%)",
                f"{err['e2_total_errors']} (23.95%)",
                f"-{err['e1_total_errors'] - err['e2_total_errors']} net errors",
            ],
        ],
        [3.15 * cm, 2.15 * cm, 2.15 * cm, 2.15 * cm, 2.9 * cm],
        highlight_row=3,
        font_name=sans,
        bold_font=sans_b,
        font_size=7.8,
    )

    s8_left = [
        Paragraph("Confusion Outcomes & Disagreement Audit", subhead_style),
        Paragraph(
            f"&bull; <b>E1 Confusion Counts:</b> TP={e1_cm['TP']}, TN={e1_cm['TN']}, FP={e1_cm['FP']}, FN={e1_cm['FN']} ({err['e1_total_errors']} errors).",
            bullet_style,
        ),
        Paragraph(
            f"&bull; <b>E2 GA-SVM Counts:</b> TP={e2_cm['TP']}, TN={e2_cm['TN']}, FP={e2_cm['FP']}, FN={e2_cm['FN']} ({err['e2_total_errors']} errors).",
            bullet_style,
        ),
        Paragraph(
            f"&bull; <b>Model Disagreements ({dis['total_disagreements']} cells):</b> E2 fixed <b>{dis['e2_fixed_e1_error']}</b> E1 errors "
            f"and regressed on <b>{dis['e2_regressed_vs_e1']}</b> (net <b>+{dis['net_improvement']}</b>).",
            bullet_style,
        ),
        Spacer(1, 4),
        err_tbl,
        Spacer(1, 6),
        _callout_box(
            "Why Polycrystalline Wafer Errors Dominate (28.1% vs 17.8%)",
            "Polycrystalline silicon exhibits random multi-crystal grain boundaries and patchy reflectance. "
            "At 64&times;64 resolution, sharp grain edges imitate hairline cracks in HOG gradients, elevating false alarms; "
            "GA pruning mitigates 6 polycrystalline errors (-6 net poly errors).",
            12.5,
            sans,
            sans_b,
            bg_color=AMBER_BG,
            border_color=AMBER_BORDER,
            font_size=8.8,
        ),
    ]
    s8_right = [
        _image_card(
            root / "results" / "figures" / "confusion_matrices.png",
            "Figure 5. Locked test confusion matrices: E1 Baseline (left) vs E2 GA-SVM seed=42 (right).",
            13.8,
            6.8,
            sans,
        ),
        Spacer(1, 5),
        _image_card(
            root / "results" / "figures" / "error_analysis_grid.png",
            "Figure 6. Qualitative error grid (seed=42 GA-SVM): TP, TN, FP (grain/busbar artifacts), and FN (subtle micro-cracks).",
            13.8,
            8.3,
            sans,
        ),
    ]
    story.append(_two_col(s8_left, s8_right, left_cm=12.6, right_cm=13.8))
    story.append(PageBreak())

    # ------------------------------------------------------------------
    # SLIDE 9: Live CLI Inference Demo Showcase
    # ------------------------------------------------------------------
    story.append(_slide_header("8. Live Demonstration", "Two model predictions beside the ELPV label", sans_b, sans))
    story.append(Spacer(1, 8))

    demo_rows = [
        ["Cell Tag (Split)", "Type", "True", "E1 Margin \u2192 Pred", "E2 Margin \u2192 Pred", "Latency"]
    ]
    for p in demo_preds:
        meta = p["meta"]
        tag = meta["sample_tag"]
        c_type = meta["cell_type"]
        true_lbl = "DEF" if meta["true_label"] == 1 else "HLTH"
        e1_m = f"{p['baseline_margin']:+.4f} \u2192 {'DEF' if p['baseline_label'] == 1 else 'HLTH'}"
        e2_tag = "DEF" if p["ga_label"] == 1 else "HLTH"
        fix_note = " (FIXED)" if (not p["models_agree"] and p["ga_label"] == meta["true_label"]) else ""
        e2_m = f"{p['ga_margin']:+.4f} \u2192 {e2_tag}{fix_note}"
        lat = f"{p['timing_ms']['ga_total_ms']:.3f} ms"
        demo_rows.append([tag, c_type, true_lbl, e1_m, e2_m, lat])

    demo_tbl = _styled_table(
        demo_rows,
        [3.1 * cm, 0.9 * cm, 0.95 * cm, 2.45 * cm, 3.45 * cm, 1.65 * cm],
        highlight_row=3,
        font_name=sans,
        bold_font=sans_b,
        font_size=7.4,
    )

    s9_left = [
        Paragraph("Browser demonstration", subhead_style),
        _callout_box(
            "",
            "<b>python scripts/app.py</b><br/><br/>"
            "Select <b>Örnek hücreyi incele</b> for cell #1576. Compare both predictions with the healthy label. "
            "Open <b>Başka bir örnek seç</b> for #850, a defect both models missed.<br/><br/>"
            "Command-line demo: <b>python scripts/demo.py</b>",
            12.5,
            sans,
            sans_b,
            bg_color=SLATE_BG,
            border_color=SLATE_BORDER,
            font_size=8.6,
        ),
        Spacer(1, 6),
        demo_tbl,
        Spacer(1, 6),
        _callout_box(
            "Decision scores",
            "A score above zero indicates a defective prediction. A score at or below zero indicates a healthy prediction. "
            "Probability estimates would require separate calibration.",
            12.5,
            sans,
            sans_b,
            bg_color=BLUE_SOFT,
            border_color=BLUE_BORDER,
            font_size=8.7,
        ),
    ]
    s9_right = [
        _image_card(
            root / "results" / "sample_predictions" / "demo_showcase.png",
            "Figure 7. Generated demo diagnostic card comparing E1 Baseline vs E2 GA-SVM across #470 (TP), #1180 (TN), and #1576 (E1 FP fixed by E2).",
            13.8,
            15.6,
            sans,
        )
    ]
    story.append(_two_col(s9_left, s9_right, left_cm=12.6, right_cm=13.8))
    story.append(PageBreak())

    # ------------------------------------------------------------------
    # SLIDE 10: Key Takeaways, Limitations & Full Reproducibility
    # ------------------------------------------------------------------
    story.append(_slide_header("9. Conclusions & Reproducibility", "Key Contributions, Honest Scope & End-to-End CPU Pipeline", sans_b, sans))
    story.append(Spacer(1, 10))

    s10_left = [
        _callout_box(
            "Major Technical Contributions",
            f"&bull; <b>Compact Classical Pipeline:</b> 64&times;64 HOG (324) + Uniform LBP (10) extracts features in "
            f"<b>{fs['ms_per_image']:.2f} ms/cell</b> without GPU or pretrained weights.<br/><br/>"
            f"&bull; <b>Feature reduction:</b> GA used <b>42.31%</b> fewer features on average across three runs "
            f"with test Macro F1 <b>{ga_ms['test_macro_f1']['mean']:.4f}</b> versus {float(e1['test_macro_f1']):.4f} "
            f"for the all-feature SVM on the same cells.<br/><br/>"
            f"&bull; <b>Individual project:</b> {author}. Development and revision assistance is described in the report.",
            13.1,
            sans,
            sans_b,
            bg_color=GREEN_BG,
            border_color=GREEN_BORDER,
            font_size=9.5,
        )
    ]
    s10_right = [
        _callout_box(
            "Evaluation scope",
            f"&bull; <b>Dataset Scope:</b> Binary ELPV subset only (2,223 definite cells; 401 ambiguous 0.333/0.667 samples excluded).<br/><br/>"
            f"&bull; <b>Module identifiers:</b> Cells from the same PV module may appear in different splits. "
            f"Performance on unseen modules needs separate evaluation.<br/><br/>"
            f"&bull; <b>Cell type:</b> Seed 42 test error was {ct_poly['e2_error_rate']*100:.1f}% on polycrystalline "
            f"cells and {ct_mono['e2_error_rate']*100:.1f}% on monocrystalline cells.",
            13.3,
            sans,
            sans_b,
            bg_color=AMBER_BG,
            border_color=AMBER_BORDER,
            font_size=9.5,
        )
    ]
    story.append(_two_col(s10_left, s10_right, left_cm=13.1, right_cm=13.3))
    story.append(Spacer(1, 12))

    cmd_banner = Table(
        [
            [
                [
                    Paragraph(
                        "END-TO-END REPRODUCIBLE CPU PIPELINE (ZERO GPU OR PRETRAINED WEIGHTS REQUIRED)",
                        ParagraphStyle("CmdHead", fontName=sans_b, fontSize=9.2, textColor=colors.HexColor("#93c5fd"), spaceAfter=4),
                    ),
                    Paragraph(
                        "python scripts/prepare_data.py  &&  python -m src.features  &&  python scripts/train_baseline.py  &&  "
                        "python scripts/run_ga.py<br/>python scripts/evaluate_models.py  &&  python scripts/demo.py  &&  "
                        "python scripts/generate_report.py  &&  python scripts/generate_slides.py",
                        code_style,
                    ),
                ]
            ]
        ],
        colWidths=[26.9 * cm],
    )
    cmd_banner.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), NAVY),
                ("TOPPADDING", (0, 0), (-1, -1), 10),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 10),
                ("LEFTPADDING", (0, 0), (-1, -1), 12),
                ("RIGHTPADDING", (0, 0), (-1, -1), 12),
            ]
        )
    )
    story.append(cmd_banner)
    story.append(Spacer(1, 10))
    story.append(
        Paragraph(
            "<b>Thank You!</b> &nbsp; All artifacts, models, reports, and presentation assets are ready in the repository. Questions &amp; Live Demo welcome.",
            ParagraphStyle("ThanksLine", fontName=sans_b, fontSize=11.5, alignment=TA_CENTER, textColor=BLUE),
        )
    )

    doc.build(story, onFirstPage=_slide_chrome, onLaterPages=_slide_chrome)
    return out_pdf


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for author metadata and output destination."""
    parser = argparse.ArgumentParser(
        description="Compile 10-slide SENG435 presentation PDF from verified experiment outputs."
    )
    parser.add_argument(
        "--author",
        default="Student Name (Edit via --author before submission)",
        help="Author / presenter full name displayed on Slide 1.",
    )
    parser.add_argument(
        "--student-id",
        default="Student ID (Edit via --student-id)",
        help="Student ID displayed on Slide 1.",
    )
    parser.add_argument(
        "--out-pdf",
        type=Path,
        default=PROJECT_ROOT / "presentation" / "slides.pdf",
        help="Output slide deck PDF path.",
    )
    return parser.parse_args()


def main() -> None:
    """Load verified experiment artifacts and build presentation/slides.pdf."""
    args = parse_args()
    art = _load_verified_artifacts(PROJECT_ROOT)
    extras = _load_slide_extras(PROJECT_ROOT)
    out_pdf = build_slides_pdf(
        PROJECT_ROOT, art, extras, args.author, args.student_id, args.out_pdf
    )
    size_bytes = out_pdf.stat().st_size
    print(f"Presentation PDF compiled successfully: {out_pdf} ({size_bytes:,} bytes)")


if __name__ == "__main__":
    main()
