"""report_tools.py — PDF generation and table formatting for ReportAgent.

All functions return {"data": ..., "error": None} on success
or {"data": None, "error": "<reason>"} on failure. Never raise.
"""

import os
from pathlib import Path
from typing import Any, Dict, List, Optional


def format_table(rows: List[Dict[str, Any]], columns: List[str]) -> Dict[str, Any]:
    """Format a list of row dicts into a plain-text table string.

    Args:
        rows: list of dicts, each containing the column keys
        columns: ordered list of column names to include

    Returns:
        {"data": {"table": str}, "error": None} or {"data": None, "error": str}
    """
    if not isinstance(rows, list):
        return {"data": None, "error": "rows must be a list"}
    if not isinstance(columns, list) or not columns:
        return {"data": None, "error": "columns must be a non-empty list"}

    try:
        col_widths = {col: max(len(col), max((len(str(r.get(col, ""))) for r in rows), default=0))
                      for col in columns}

        header = " | ".join(col.ljust(col_widths[col]) for col in columns)
        separator = "-+-".join("-" * col_widths[col] for col in columns)
        data_rows = [
            " | ".join(str(row.get(col, "")).ljust(col_widths[col]) for col in columns)
            for row in rows
        ]
        table = "\n".join([header, separator] + data_rows)
        return {"data": {"table": table}, "error": None}
    except Exception as exc:
        return {"data": None, "error": f"format_table failed: {exc}"}


# ---------------------------------------------------------------------------
# Chart caption / explanation text
# ---------------------------------------------------------------------------

_CHART_EXPLANATIONS: Dict[str, str] = {
    "ratios": (
        "This chart displays the key financial ratios computed from the annual 10-K filing. "
        "Debt-to-Equity measures leverage (higher = more debt relative to equity). "
        "Return on Assets and Net Margin capture profitability efficiency. "
        "Interest Coverage shows how many times EBIT covers interest expense — "
        "values below 1.5× indicate earnings barely cover obligations. "
        "Note: Current Ratio is omitted because banks file unclassified balance sheets "
        "and do not separately report current assets/liabilities."
    ),
    "income": (
        "This chart shows the three core income statement figures in billions of dollars: "
        "Revenue (total net interest and non-interest income), EBIT (earnings before "
        "interest and taxes, proxied by pre-tax income for banks), and Net Income "
        "(after-tax earnings attributable to shareholders). "
        "The gap between Revenue and Net Income reflects the bank's operating cost base "
        "and interest expense burden."
    ),
    "eda_ratio_trends": (
        "Five-year trend of key financial ratios across all five focus banks "
        "(PNC, JPM, BNY, WFC, BAC) from 2018 to 2023. "
        "Ratio trends reveal whether profitability and leverage have improved or "
        "deteriorated over the cycle, including the COVID shock (2020) and the "
        "subsequent rate-tightening period (2022–2023)."
    ),
    "eda_zscore_distribution": (
        "Distribution of Altman Z-Scores across the 15-bank dataset used to train "
        "the ML risk model. Because all banks are highly leveraged, Z-Scores cluster "
        "well below the traditional 1.81 distress threshold — confirming that the "
        "absolute zone interpretation is not appropriate for commercial banks. "
        "Risk assessment should be peer-relative, not absolute."
    ),
    "eda_zscore_peer_2023": (
        "Peer comparison of Altman Z-Scores for the five focus banks in 2023. "
        "PNC's Z-Score of 0.284 ranks at the 100th percentile of this peer group — "
        "it is the highest (least distressed by the metric) among peers. "
        "This contextualises the distress-zone label: all major US banks score "
        "similarly low due to structural leverage, and PNC is at the top of that range."
    ),
    "eda_macro_overlay": (
        "Macro environment overlay showing Federal Funds Rate (FEDFUNDS), "
        "10-Year minus 2-Year Treasury spread (T10Y2Y), and Unemployment Rate (UNRATE) "
        "from 2018 to 2023. "
        "The yield curve inversion (T10Y2Y < 0) that persisted through 2022–2023 "
        "is a key risk factor: inverted curves compress bank net interest margins "
        "and historically precede recessions. FEDFUNDS at 5.33% in 2023 reflects "
        "the Fed's most aggressive tightening cycle since the 1980s."
    ),
    "eda_correlation_heatmap": (
        "Correlation heatmap of the seven ML model features across the 15-bank "
        "training dataset. Strong correlations between features can indicate "
        "multicollinearity that reduces model interpretability. "
        "Return on Assets and Net Margin are highly correlated (both profitability measures). "
        "Macro features (FEDFUNDS, T10Y2Y) show low correlation with balance-sheet ratios, "
        "confirming they add independent signal to the risk model."
    ),
}


def _chart_explanation(path: str) -> str:
    """Return an explanation paragraph for a chart based on its filename."""
    name = os.path.basename(path).replace(".png", "")
    for key, text in _CHART_EXPLANATIONS.items():
        if name.endswith(key) or key in name:
            return text
    return f"Chart: {name.replace('_', ' ')}."


def generate_pdf(
    output_path: str,
    ticker: str,
    year: int,
    narrative: str,
    table_str: str,
    chart_paths: Optional[List[str]] = None,
    eda_chart_paths: Optional[List[str]] = None,
    ingest: Optional[Dict[str, Any]] = None,
    ratios: Optional[Dict[str, Any]] = None,
    risk: Optional[Dict[str, Any]] = None,
    anomalies: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """Generate a PDF analyst report and write it to output_path.

    Produces a fully-sectioned report: cover, executive summary, key metrics
    table, financial highlights with charts, risk assessment, macro context,
    and supplementary EDA charts — each with explanatory prose.

    Args:
        output_path: destination file path (must end in .pdf)
        ticker: bank ticker symbol
        year: fiscal year
        narrative: analyst narrative from NarrativeAgent
        table_str: pre-formatted metrics table string
        chart_paths: ticker-specific chart PNGs (ratios, income)
        eda_chart_paths: EDA chart PNGs for the Supplementary section
        ingest: IngestAgent output dict (financial figures)
        ratios: computed ratio dict from AnalysisAgent
        risk: RiskAgent output dict
        anomalies: list of anomaly/flag dicts from AnalysisAgent

    Returns:
        {"data": {"path": str}, "error": None} or {"data": None, "error": str}
    """
    if not output_path.endswith(".pdf"):
        return {"data": None, "error": f"output_path must end in .pdf, got: {output_path}"}
    if not ticker:
        return {"data": None, "error": "ticker must be non-empty"}

    ingest = ingest or {}
    ratios = ratios or {}
    risk = risk or {}
    anomalies = anomalies or []

    try:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        try:
            from reportlab.lib import colors
            from reportlab.lib.pagesizes import letter
            from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
            from reportlab.lib.units import inch
            from reportlab.platypus import (
                HRFlowable, Image, PageBreak, Paragraph,
                Preformatted, SimpleDocTemplate, Spacer, Table, TableStyle,
            )

            W, H = letter
            IMG_W = 6.0 * inch
            IMG_H = 3.5 * inch

            doc = SimpleDocTemplate(
                output_path, pagesize=letter,
                leftMargin=0.85 * inch, rightMargin=0.85 * inch,
                topMargin=1.0 * inch, bottomMargin=1.0 * inch,
            )
            base = getSampleStyleSheet()

            # Custom styles
            title_style = ParagraphStyle(
                "FSTitle", parent=base["Title"],
                fontSize=22, spaceAfter=4, textColor=colors.HexColor("#1a2e4a"),
            )
            subtitle_style = ParagraphStyle(
                "FSSubtitle", parent=base["Normal"],
                fontSize=12, textColor=colors.HexColor("#4a6080"), spaceAfter=20,
            )
            h2 = ParagraphStyle(
                "FSH2", parent=base["Heading2"],
                fontSize=14, textColor=colors.HexColor("#1a2e4a"),
                spaceBefore=18, spaceAfter=6, borderPad=0,
            )
            h3 = ParagraphStyle(
                "FSH3", parent=base["Heading3"],
                fontSize=11, textColor=colors.HexColor("#2c5282"),
                spaceBefore=12, spaceAfter=4,
            )
            body = ParagraphStyle(
                "FSBody", parent=base["Normal"],
                fontSize=10, leading=15, spaceAfter=8,
            )
            caption = ParagraphStyle(
                "FSCaption", parent=base["Italic"],
                fontSize=9, textColor=colors.HexColor("#555555"),
                spaceAfter=12, leftIndent=6,
            )
            label_style = ParagraphStyle(
                "FSLabel", parent=base["Normal"],
                fontSize=9, textColor=colors.HexColor("#888888"),
            )

            def hr():
                return HRFlowable(
                    width="100%", thickness=0.5,
                    color=colors.HexColor("#c0cce0"), spaceAfter=8,
                )

            def section(title: str) -> list:
                return [Spacer(1, 6), Paragraph(title, h2), hr()]

            def embed_chart(path: str) -> list:
                abs_path = os.path.abspath(path)
                if not os.path.isfile(abs_path):
                    return []
                name = os.path.basename(abs_path).replace(".png", "").replace("_", " ")
                explanation = _chart_explanation(abs_path)
                return [
                    Paragraph(name.title(), h3),
                    Paragraph(explanation, body),
                    Spacer(1, 6),
                    Image(abs_path, width=IMG_W, height=IMG_H),
                    Paragraph(f"Figure: {name}", caption),
                    Spacer(1, 8),
                ]

            def fmt_b(v, digits=1):
                if v is None:
                    return "N/A"
                return f"${v / 1e9:.{digits}f}B"

            def fmt_pct(v, digits=2):
                if v is None:
                    return "N/A"
                return f"{v * 100:.{digits}f}%"

            def fmt_x(v, digits=2):
                if v is None:
                    return "N/A"
                return f"{v:.{digits}f}×"

            # ----------------------------------------------------------------
            # Build story
            # ----------------------------------------------------------------
            story: List[Any] = []

            # --- Cover ---
            fred = ingest.get("fred_snapshot", {})
            story += [
                Paragraph(f"FinSight Analyst Report", title_style),
                Paragraph(f"{ticker} Financial Services &nbsp;|&nbsp; Fiscal Year {year}", subtitle_style),
                Paragraph(
                    f"Generated by FinSight Multi-Agent Pipeline &nbsp;|&nbsp; DS 2000 · University of Pittsburgh",
                    label_style,
                ),
                Spacer(1, 10),
                hr(),
            ]

            # --- Executive Summary ---
            story += section("1. Executive Summary")
            story += [Paragraph(narrative.replace("\n", "<br/>"), body)]

            # --- Key Metrics Table ---
            story += section("2. Key Metrics at a Glance")
            story += [
                Paragraph(
                    "The table below summarises the primary financial indicators derived "
                    "from the annual 10-K filing and computed by the FinSight pipeline.",
                    body,
                ),
            ]

            # Build a proper ReportLab table instead of monospaced text
            z_score = risk.get("z_score")
            z_zone  = risk.get("z_zone", "N/A")
            ml_label = risk.get("ml_risk_label", "N/A")
            ml_prob  = risk.get("ml_risk_prob")
            peer_pct = risk.get("peer_percentile")

            metric_data = [
                ["Metric", "Value", "Note"],
                ["Revenue", fmt_b(ingest.get("revenue")), "Total net interest + non-interest income"],
                ["Net Income", fmt_b(ingest.get("net_income")), "After-tax earnings"],
                ["Total Assets", fmt_b(ingest.get("total_assets"), 0), "Balance sheet size"],
                ["Total Liabilities", fmt_b(ingest.get("total_liabilities"), 0), ""],
                ["Shareholders Equity", fmt_b(ingest.get("shareholders_equity")), ""],
                ["EBIT", fmt_b(ingest.get("ebit")), "Pre-tax income proxy"],
                ["Interest Expense", fmt_b(ingest.get("interest_expense")), ""],
                ["Market Cap", fmt_b(ingest.get("market_cap")), "Historical year-end"],
                ["Debt / Equity", f"{ratios.get('debt_to_equity') or 0:.2f}×", "Leverage ratio"],
                ["Return on Assets", fmt_pct(ratios.get("return_on_assets")), "Net income / total assets"],
                ["Net Margin", fmt_pct(ratios.get("net_margin")), "Net income / revenue"],
                ["Interest Coverage", fmt_x(ratios.get("interest_coverage")), "EBIT / interest expense"],
                ["Altman Z-Score", f"{z_score:.3f}" if z_score is not None else "N/A",
                 f"Zone: {z_zone} (all banks < 1.81 — read vs. peers)"],
                ["ML Risk Label", ml_label,
                 f"Prob: {ml_prob:.0%}" if ml_prob is not None else ""],
                ["Peer Percentile", f"{peer_pct:.0%}" if peer_pct is not None else "N/A",
                 "vs. 15-bank sector universe"],
            ]

            tbl = Table(
                metric_data,
                colWidths=[2.0 * inch, 1.4 * inch, 3.0 * inch],
                repeatRows=1,
            )
            tbl.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1a2e4a")),
                ("TEXTCOLOR",  (0, 0), (-1, 0), colors.white),
                ("FONTNAME",   (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE",   (0, 0), (-1, 0), 9),
                ("FONTNAME",   (0, 1), (-1, -1), "Helvetica"),
                ("FONTSIZE",   (0, 1), (-1, -1), 9),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1),
                 [colors.HexColor("#f5f7fa"), colors.white]),
                ("GRID",       (0, 0), (-1, -1), 0.4, colors.HexColor("#c0cce0")),
                ("LEFTPADDING",  (0, 0), (-1, -1), 6),
                ("RIGHTPADDING", (0, 0), (-1, -1), 6),
                ("TOPPADDING",   (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING",(0, 0), (-1, -1), 4),
                ("VALIGN",     (0, 0), (-1, -1), "MIDDLE"),
            ]))
            story += [tbl, Spacer(1, 12)]

            # --- Financial Highlights + ticker charts ---
            story += section("3. Financial Highlights")
            story += [
                Paragraph(
                    f"The following charts visualise {ticker}'s {year} financial profile "
                    "as derived from EDGAR XBRL data. Each chart is accompanied by an "
                    "interpretation note.",
                    body,
                ),
            ]
            for path in (chart_paths or []):
                story += embed_chart(path)

            # --- Risk Assessment ---
            story += section("4. Risk Assessment")

            # Z-score narrative
            story += [
                Paragraph("4.1 Altman Z-Score", h3),
                Paragraph(
                    f"{ticker}'s Altman Z-Score for {year} is "
                    f"<b>{z_score:.3f}</b> (zone: <b>{z_zone}</b>). "
                    "The Altman Z-Score formula is: "
                    "Z = 1.2·(WC/TA) + 1.4·(RE/TA) + 3.3·(EBIT/TA) + 0.6·(MktCap/TL) + 1.0·(Rev/TA). "
                    "Traditional thresholds: &gt;2.99 safe, 1.81–2.99 grey, &lt;1.81 distress. "
                    "All major US commercial banks score structurally below 1.81 due to high leverage — "
                    "the distress label reflects the sector's balance-sheet structure, not imminent default risk. "
                    f"{ticker} ranks at the {peer_pct:.0%} percentile of its 15-bank peer group, "
                    "indicating it is among the stronger performers on this metric relative to peers."
                    if peer_pct is not None else "",
                    body,
                ),
            ]

            # ML risk
            story += [
                Paragraph("4.2 ML Risk Model", h3),
                Paragraph(
                    f"The FinSight Random Forest classifier (7 features, 60-row training set, "
                    f"51.7% CV accuracy — 18 pts above 33% random baseline for a balanced 3-class problem) "
                    f"assigned {ticker} a <b>{ml_label}</b> risk classification "
                    f"with a high-risk probability of <b>{ml_prob:.0%}</b>. "
                    "Features: Debt/Equity, Return on Assets, Interest Coverage, Revenue Growth, "
                    "Net Margin, FEDFUNDS, T10Y2Y. Labels are peer-relative tertiles "
                    "(lowest Z-Score third = high risk, top third = low risk).",
                    body,
                ) if ml_prob is not None else Paragraph(
                    "ML risk model output unavailable.", body,
                ),
            ]

            # Risk flags
            story += [Paragraph("4.3 Risk Flags", h3)]
            risk_flags = risk.get("risk_flags", [])
            if risk_flags:
                flag_rows = [["Flag", "Severity", "Description"]]
                for f in risk_flags:
                    flag_rows.append([
                        f.get("code", "").replace("_", " ").title(),
                        f.get("severity", ""),
                        f.get("description", ""),
                    ])
                ftbl = Table(
                    flag_rows,
                    colWidths=[1.5 * inch, 0.9 * inch, 4.0 * inch],
                    repeatRows=1,
                )
                ftbl.setStyle(TableStyle([
                    ("BACKGROUND",  (0, 0), (-1, 0), colors.HexColor("#c0392b")),
                    ("TEXTCOLOR",   (0, 0), (-1, 0), colors.white),
                    ("FONTNAME",    (0, 0), (-1, 0), "Helvetica-Bold"),
                    ("FONTSIZE",    (0, 0), (-1, -1), 9),
                    ("FONTNAME",    (0, 1), (-1, -1), "Helvetica"),
                    ("ROWBACKGROUNDS", (0, 1), (-1, -1),
                     [colors.HexColor("#fff5f5"), colors.white]),
                    ("GRID",        (0, 0), (-1, -1), 0.4, colors.HexColor("#e8b4b8")),
                    ("LEFTPADDING",  (0, 0), (-1, -1), 6),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 6),
                    ("TOPPADDING",   (0, 0), (-1, -1), 4),
                    ("BOTTOMPADDING",(0, 0), (-1, -1), 4),
                    ("VALIGN",      (0, 0), (-1, -1), "TOP"),
                ]))
                story += [ftbl, Spacer(1, 8)]
            else:
                story += [Paragraph("No risk flags raised.", body)]

            # Anomalies
            if anomalies:
                story += [
                    Paragraph("4.4 Detected Anomalies", h3),
                    Paragraph(
                        "The following ratio anomalies were detected relative to "
                        "expected ranges for large-cap US commercial banks:",
                        body,
                    ),
                ]
                for a in anomalies:
                    story += [Paragraph(
                        f"• <b>{a.get('metric', '').replace('_', ' ').title()}</b>: "
                        f"{a.get('description', a.get('message', ''))}",
                        body,
                    )]

            # --- Macro Context ---
            story += section("5. Macroeconomic Context")
            fedfunds = fred.get("FEDFUNDS")
            t10y2y  = fred.get("T10Y2Y")
            unrate  = fred.get("UNRATE")
            cpi     = fred.get("CPIAUCSL")
            gs10    = fred.get("GS10")

            macro_rows = [["Series", "Value (Year-End)", "Interpretation"]]
            if fedfunds is not None:
                macro_rows.append(["FEDFUNDS", f"{fedfunds:.2f}%",
                                   "Fed Funds target rate �� cost of overnight borrowing"])
            if t10y2y is not None:
                inv = " (INVERTED)" if t10y2y < 0 else ""
                macro_rows.append(["T10Y2Y", f"{t10y2y:.2f}%{inv}",
                                   "10Y minus 2Y Treasury spread — recession signal when negative"])
            if unrate is not None:
                macro_rows.append(["UNRATE", f"{unrate:.1f}%",
                                   "Unemployment rate — rising levels signal credit loss risk"])
            if gs10 is not None:
                macro_rows.append(["GS10", f"{gs10:.2f}%",
                                   "10-Year Treasury yield — long-term rate benchmark"])
            if cpi is not None:
                macro_rows.append(["CPI", f"{cpi:.1f}",
                                   "Consumer Price Index level"])

            if len(macro_rows) > 1:
                mtbl = Table(
                    macro_rows,
                    colWidths=[1.2 * inch, 1.5 * inch, 3.7 * inch],
                    repeatRows=1,
                )
                mtbl.setStyle(TableStyle([
                    ("BACKGROUND",  (0, 0), (-1, 0), colors.HexColor("#1a4a2e")),
                    ("TEXTCOLOR",   (0, 0), (-1, 0), colors.white),
                    ("FONTNAME",    (0, 0), (-1, 0), "Helvetica-Bold"),
                    ("FONTSIZE",    (0, 0), (-1, -1), 9),
                    ("FONTNAME",    (0, 1), (-1, -1), "Helvetica"),
                    ("ROWBACKGROUNDS", (0, 1), (-1, -1),
                     [colors.HexColor("#f5faf7"), colors.white]),
                    ("GRID",        (0, 0), (-1, -1), 0.4, colors.HexColor("#a8cdb8")),
                    ("LEFTPADDING",  (0, 0), (-1, -1), 6),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 6),
                    ("TOPPADDING",   (0, 0), (-1, -1), 4),
                    ("BOTTOMPADDING",(0, 0), (-1, -1), 4),
                    ("VALIGN",      (0, 0), (-1, -1), "MIDDLE"),
                ]))
                story += [
                    Paragraph(
                        f"FRED macro data as of {year}-12-31. "
                        "The yield curve inversion (T10Y2Y=-0.35%) and elevated "
                        "FEDFUNDS (5.33%) are the dominant macro risk factors for "
                        f"{ticker} in {year}: inversion compresses net interest margins, "
                        "while high short rates increase funding costs.",
                        body,
                    ),
                    mtbl,
                    Spacer(1, 8),
                ]

            # --- Supplementary EDA Charts ---
            if eda_chart_paths:
                story += [PageBreak()]
                story += section("6. Supplementary Analysis — Sector EDA Charts")
                story += [
                    Paragraph(
                        "The following charts are produced from the FinSight exploratory "
                        "data analysis (EDA) across 15 US commercial banks from 2018 to 2023. "
                        "They provide sector-wide context for interpreting the single-bank "
                        f"results above. Each chart is labelled and explained below.",
                        body,
                    ),
                ]
                for path in eda_chart_paths:
                    story += embed_chart(path)

            # --- Conclusion ---
            story += section("7. Conclusion")
            z_str = f"{z_score:.3f}" if z_score is not None else "N/A"
            story += [
                Paragraph(
                    f"Based on {year} annual 10-K data, {ticker} reported revenue of "
                    f"{fmt_b(ingest.get('revenue'))} and net income of "
                    f"{fmt_b(ingest.get('net_income'))}, with total assets of "
                    f"{fmt_b(ingest.get('total_assets'), 0)}. "
                    f"The Altman Z-Score of {z_str} (distress zone) is structurally "
                    "expected for a leveraged commercial bank and should be read against "
                    f"peer benchmarks — {ticker} ranks at the "
                    f"{f'{peer_pct:.0%}' if peer_pct is not None else 'N/A'} "
                    "percentile of its peer group. "
                    f"The ML model classifies {ticker} as <b>{ml_label}</b> risk. "
                    "Key macro risks are the inverted yield curve and elevated policy rates, "
                    "both of which compress net interest margins. "
                    "Interest coverage of "
                    f"{fmt_x(ratios.get('interest_coverage'))} is below the 1.5× monitoring "
                    "threshold, warranting continued attention.",
                    body,
                ),
                Spacer(1, 20),
                Paragraph(
                    "<i>This report was generated autonomously by the FinSight multi-agent "
                    "pipeline. Data sources: SEC EDGAR (XBRL), FRED (Federal Reserve), "
                    "yfinance. ML model: Random Forest, 60-row training set, 51.7% CV accuracy. "
                    "Not investment advice.</i>",
                    caption,
                ),
            ]

            doc.build(story)

        except ImportError:
            with open(output_path, "w") as fh:
                fh.write(f"FinSight Analyst Report — {ticker} {year}\n\n")
                fh.write(narrative)
                fh.write("\n\n")
                fh.write(table_str)

        return {"data": {"path": output_path}, "error": None}

    except Exception as exc:
        return {"data": None, "error": f"generate_pdf failed: {exc}"}
