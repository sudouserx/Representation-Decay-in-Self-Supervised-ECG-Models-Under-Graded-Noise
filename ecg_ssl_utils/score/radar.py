"""Radar plot data preparation for 5-axis DSS components."""
import numpy as np

def prepare_radar_data(dss_results_df):
    """
    Given a dataframe of DSS results, prepare data for radar plots.
    Axes: CKA, Effective Rank, Calibration (1-ECE), Task (AUROC), Speed (1-Latency).
    Higher is always better.
    """
    # Assuming dss_results_df has columns:
    # model_id, noise_type, snr_db, cka, erank, ece, auroc, latency
    
    # Normalization should happen before passing here or we just use raw
    # But for a good radar plot, everything should be 0-1
    # We expect these values to already be in [0, 1] range via reference_anchored_normalize
    
    radar_data = {}
    for idx, row in dss_results_df.iterrows():
        model = row['model_id']
        condition = f"{row['noise_type']}@{row['snr_db']}dB"
        
        # Orient so higher is better on the plot
        # delta_cka and delta_er are usually 0 for perfect, 1 for worst
        # So we use 1 - delta
        
        values = [
            1.0 - row.get('delta_cka', 0.0),      # CKA retention
            1.0 - row.get('delta_erank', 0.0),    # Dimensionality retention
            1.0 - row.get('ece', 0.0),            # Calibration
            row.get('auroc', 0.5),                # Task performance
            1.0 - row.get('latency_norm', 0.0)    # Speed
        ]
        
        if model not in radar_data:
            radar_data[model] = {}
        radar_data[model][condition] = values
        
    axis_labels = ['Representation (CKA)', 'Dimensionality (ER)', 'Calibration (1-ECE)', 'Task (AUROC)', 'Speed (1-Lat)']
    
    return radar_data, axis_labels
