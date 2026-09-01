"""
Per-noise-type risk report generator.
Produces model × SNR heatmaps for each noise type, covering
AUROC, ECE, CKA, F1 metrics — fulfilling methodology artifact R3.
"""
import numpy as np
import pandas as pd
from typing import Optional


def generate_risk_report(
    decay_df: pd.DataFrame,
    clean_ref_df: pd.DataFrame,
    output_dir: str,
    metrics: Optional[list] = None,
) -> str:
    """
    Generate per-noise-type risk tables as HTML.

    Parameters
    ----------
    decay_df : pd.DataFrame
        Aggregated metric curves (from 05_decay_metrics.py).
        Columns: encoder, noise_type, snr_db, auroc, ece, cka, f1_macro, ...
    clean_ref_df : pd.DataFrame
        Clean reference metrics per encoder.
        Columns: encoder, auroc, ece, erank, f1_macro, ...
    output_dir : str
        Directory to write HTML reports.
    metrics : list of str, optional
        Metrics to include. Default: ['auroc', 'ece', 'cka', 'f1_macro'].

    Returns
    -------
    report_path : str
        Path to the generated HTML report.
    """
    import os
    os.makedirs(output_dir, exist_ok=True)

    if metrics is None:
        metrics = [
            'auroc', 'ece', 'cka', 'f1_macro',
            'f1_NORM', 'f1_MI', 'f1_STTC', 'f1_CD', 'f1_HYP'
        ]

    noise_types = sorted(decay_df['noise_type'].unique())
    encoders = sorted(decay_df['encoder'].unique())
    snr_grid = sorted(decay_df['snr_db'].unique(), reverse=True)

    html_parts = [
        '<html><head><style>',
        'body{font-family:sans-serif;margin:20px;}',
        'table{border-collapse:collapse;margin:10px 0;width:auto;}',
        'th,td{border:1px solid #ddd;padding:6px 10px;text-align:center;font-size:13px;}',
        'th{background:#34495e;color:white;}',
        'h1{color:#2c3e50;} h2{color:#34495e;} h3{color:#7f8c8d;}',
        '.good{background:#c8e6c9;} .warn{background:#fff9c4;} .bad{background:#ffcdd2;}',
        '</style></head><body>',
        '<h1>Per-Noise-Type Risk Report</h1>',
    ]

    # Clean reference table
    html_parts.append('<h2>Clean Reference Baselines</h2>')
    if not clean_ref_df.empty:
        cols_to_show = ['encoder'] + [m for m in metrics if m in clean_ref_df.columns]
        html_parts.append(clean_ref_df[cols_to_show].to_html(index=False, float_format='%.4f'))

    # Per-noise-type heatmaps
    for ntype in noise_types:
        html_parts.append(f'<h2>Noise Type: {ntype}</h2>')
        subset = decay_df[decay_df['noise_type'] == ntype]

        for metric in metrics:
            if metric not in subset.columns:
                continue

            html_parts.append(f'<h3>{metric.upper()}</h3>')
            html_parts.append('<table><tr><th>Encoder \\ SNR (dB)</th>')
            for snr in snr_grid:
                html_parts.append(f'<th>{snr}</th>')
            html_parts.append('</tr>')

            for enc in encoders:
                html_parts.append(f'<tr><td><b>{enc}</b></td>')
                enc_data = subset[subset['encoder'] == enc]

                for snr in snr_grid:
                    row = enc_data[enc_data['snr_db'] == snr]
                    if row.empty:
                        html_parts.append('<td>—</td>')
                    else:
                        val = row.iloc[0][metric]
                        # Color coding based on metric type
                        if metric in ['auroc', 'cka', 'f1_macro']:
                            # Higher is better
                            css = 'good' if val >= 0.8 else ('warn' if val >= 0.6 else 'bad')
                        elif metric == 'ece':
                            # Lower is better
                            css = 'good' if val <= 0.05 else ('warn' if val <= 0.15 else 'bad')
                        else:
                            css = ''
                        html_parts.append(f'<td class="{css}">{val:.4f}</td>')

                html_parts.append('</tr>')
            html_parts.append('</table>')

    html_parts.append('</body></html>')

    report_path = os.path.join(output_dir, 'risk_report.html')
    with open(report_path, 'w') as f:
        f.write('\n'.join(html_parts))

    print(f"Risk report saved to {report_path}")
    return report_path
