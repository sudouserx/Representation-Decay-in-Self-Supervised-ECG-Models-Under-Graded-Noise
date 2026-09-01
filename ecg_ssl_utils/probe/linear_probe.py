"""Linear probe for SSL evaluation + temperature scaling calibration."""
import torch, torch.nn as nn
import numpy as np
from sklearn.metrics import roc_auc_score
from typing import Optional, Tuple


class LinearProbe(nn.Module):
    """Single linear layer: frozen representations → class logits."""
    def __init__(self, in_dim=384, n_classes=71):
        super().__init__()
        self.fc = nn.Linear(in_dim, n_classes)
        self.temperature = nn.Parameter(torch.ones(1), requires_grad=False)

    def forward(self, h):
        return self.fc(h) / self.temperature

    def predict_proba(self, h):
        return torch.sigmoid(self.forward(h))


def train_probe(representations, labels, val_repr, val_labels,
                in_dim=384, n_classes=71, epochs=50, batch_size=512,
                lr=1e-2, patience=10, device='cuda'):
    """Train linear probe on frozen representations. Returns probe + metrics."""
    probe = LinearProbe(in_dim, n_classes).to(device)
    opt = torch.optim.Adam(probe.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    criterion = nn.BCEWithLogitsLoss()

    reps_t = torch.tensor(representations, dtype=torch.float32)
    labs_t = torch.tensor(labels, dtype=torch.float32)
    val_r = torch.tensor(val_repr, dtype=torch.float32).to(device)
    val_l = torch.tensor(val_labels, dtype=torch.float32).to(device)

    ds = torch.utils.data.TensorDataset(reps_t, labs_t)
    dl = torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=True)

    best_auroc, wait, best_state = 0, 0, None
    for ep in range(epochs):
        probe.train()
        for h, y in dl:
            h, y = h.to(device), y.to(device)
            loss = criterion(probe.fc(h), y)
            opt.zero_grad(); loss.backward(); opt.step()
        sched.step()
        # Val
        probe.eval()
        with torch.no_grad():
            logits = probe.fc(val_r)
            probs = torch.sigmoid(logits).cpu().numpy()
        try:
            auroc = roc_auc_score(val_labels, probs, average='macro', multi_class='ovr')
        except ValueError:
            auroc = 0.5
        if auroc > best_auroc:
            best_auroc = auroc; wait = 0
            best_state = {k: v.clone() for k, v in probe.state_dict().items()}
        else:
            wait += 1
            if wait >= patience: break

    if best_state: probe.load_state_dict(best_state)
    return probe, best_auroc
