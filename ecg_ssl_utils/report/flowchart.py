"""Decision flowchart generator (Mermaid syntax)."""

def generate_decision_flowchart(leaderboard_df, save_path='decision_flowchart.md'):
    """
    Generate a decision flowchart based on the leaderboard data using Mermaid JS.
    """
    # Simple logic to determine "best" models under different constraints
    
    best_overall = leaderboard_df.iloc[0]['model_id'] if not leaderboard_df.empty else 'N/A'
    
    # Best under latency constraint
    fast = leaderboard_df[leaderboard_df['latency_p50'] < 0.005]
    best_fast = fast.iloc[0]['model_id'] if not fast.empty else best_overall
    
    # Best calibration
    calib = leaderboard_df[leaderboard_df['ece'] < 0.05]
    best_calib = calib.iloc[0]['model_id'] if not calib.empty else best_overall
    
    # Best noise robustness (e.g. at -6dB)
    # Assuming this dataframe is pre-filtered for worst-case noise
    best_robust = leaderboard_df.sort_values('auroc', ascending=False).iloc[0]['model_id'] if not leaderboard_df.empty else best_overall

    mermaid = f"""```mermaid
graph TD
    Start[New ECG Task] --> Q1{{Is latency critical <5ms?}}
    
    Q1 -- Yes --> Q2{{Is strict calibration <5% ECE required?}}
    Q1 -- No --> Q3{{Is noise tolerance -6dB SNR critical?}}
    
    Q2 -- Yes --> M1[{best_fast} (if it passes ECE gate) else fallback]
    Q2 -- No --> M2[{best_fast}]
    
    Q3 -- Yes --> M3[{best_robust}]
    Q3 -- No --> M4[{best_overall} (Best balanced DSS)]
```
"""
    with open(save_path, 'w') as f:
        f.write(mermaid)
    return save_path
