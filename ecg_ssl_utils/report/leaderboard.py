"""HTML leaderboard generator with hardware-ceiling filtering."""
import pandas as pd
from typing import Optional


def generate_leaderboard(dss_df, deploy_df, save_path='leaderboard.html',
                         max_latency=None, max_memory=None,
                         min_auroc=None, max_ece=None):
    """Generate filterable HTML leaderboard from DSS and deployment data."""
    merged = dss_df.copy()
    if max_latency:
        merged = merged[merged['latency_p50'] <= max_latency]
    if max_memory:
        merged = merged[merged['memory_mb'] <= max_memory]
    if min_auroc:
        merged = merged[merged['auroc'] >= min_auroc]
    if max_ece:
        merged = merged[merged['ece'] <= max_ece]

    merged = merged.sort_values('robustness_score', ascending=False)

    html = '<html><head><style>'
    html += 'table{border-collapse:collapse;width:100%;font-family:sans-serif;}'
    html += 'th,td{border:1px solid #ddd;padding:8px;text-align:center;}'
    html += 'th{background:#2c3e50;color:white;}'
    html += 'tr:nth-child(even){background:#f2f2f2;}'
    html += 'tr:hover{background:#ddd;}'
    html += '.pass{color:green;font-weight:bold;}'
    html += '.fail{color:red;font-weight:bold;}'
    html += '</style></head><body>'
    html += '<h1>ECG SSL Robustness Leaderboard</h1>'
    html += merged.to_html(index=False, escape=False, classes='leaderboard')
    html += '</body></html>'

    with open(save_path, 'w') as f:
        f.write(html)
    print(f"Leaderboard saved to {save_path}")
    return save_path
