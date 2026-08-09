"""
CLOCS: Contrastive Learning of Cardiac Signals
Temporal, spatial (lead), and patient-level contrastive learning.

Loss: L_CLOCS = λ_T·L_temporal + λ_S·L_spatial + λ_P·L_patient
Each component uses NT-Xent with different positive pair definitions.

Reference: Kiyasseh et al., ICML 2021.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List, Dict
from .simclr import nt_xent_loss


class CLOCSTrainer:
    """
    CLOCS training wrapper with temporal, spatial, and patient contrastive losses.

    Positive pairs:
    - Temporal: two non-overlapping segments from the same recording
    - Spatial: different lead groups from the same recording
    - Patient: different recordings from the same patient
    """

    def __init__(
        self,
        encoder: nn.Module,
        projector: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[object] = None,
        temperature: float = 0.1,
        lambda_temporal: float = 1.0,
        lambda_spatial: float = 1.0,
        lambda_patient: float = 0.5,
        segment_length: int = 2500,
        lead_group_a: List[int] = None,
        lead_group_b: List[int] = None,
        use_amp: bool = True,
        device: str = 'cuda',
    ):
        self.encoder = encoder.to(device)
        self.projector = projector.to(device)
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.temperature = temperature
        self.lambda_temporal = lambda_temporal
        self.lambda_spatial = lambda_spatial
        self.lambda_patient = lambda_patient
        self.segment_length = segment_length
        self.lead_group_a = lead_group_a or [0, 1, 2, 3, 4, 5]
        self.lead_group_b = lead_group_b or [6, 7, 8, 9, 10, 11]
        self.use_amp = use_amp
        self.device = device
        self.scaler = torch.amp.GradScaler('cuda') if use_amp else None

    def _temporal_loss(self, batch: torch.Tensor) -> torch.Tensor:
        """
        Temporal contrastive loss: two halves of the same recording
        are positive pairs.
        """
        B, C, L = batch.shape
        half = self.segment_length

        seg1 = batch[:, :, :half]     # first half
        seg2 = batch[:, :, half:2*half] if 2*half <= L else batch[:, :, L-half:]

        # Pad segments to match encoder input
        seg1 = F.pad(seg1, (0, L - half))
        seg2 = F.pad(seg2, (0, L - half))

        h1 = self.encoder(seg1)
        h2 = self.encoder(seg2)
        z1 = self.projector(h1)
        z2 = self.projector(h2)

        return nt_xent_loss(z1, z2, self.temperature)

    def _spatial_loss(self, batch: torch.Tensor) -> torch.Tensor:
        """
        Spatial/lead contrastive loss: two lead groups from the same
        recording are positive pairs.
        """
        B, C, L = batch.shape

        # Select lead groups, zero out the other leads
        view_a = batch.clone()
        view_a[:, self.lead_group_b, :] = 0.0

        view_b = batch.clone()
        view_b[:, self.lead_group_a, :] = 0.0

        h_a = self.encoder(view_a)
        h_b = self.encoder(view_b)
        z_a = self.projector(h_a)
        z_b = self.projector(h_b)

        return nt_xent_loss(z_a, z_b, self.temperature)

    def _patient_loss(
        self, batch: torch.Tensor, patient_ids: torch.Tensor
    ) -> Optional[torch.Tensor]:
        """
        Patient contrastive loss: different recordings from the same
        patient are positive pairs.

        Returns None if no patient has multiple recordings in the batch.
        """
        B = batch.shape[0]

        # Find patients with multiple recordings in this batch
        unique_patients, counts = torch.unique(patient_ids, return_counts=True)
        multi_patients = unique_patients[counts >= 2]

        if len(multi_patients) == 0:
            return None

        # Collect pairs
        anchors = []
        positives = []
        for pid in multi_patients:
            indices = (patient_ids == pid).nonzero(as_tuple=True)[0]
            # Use first two recordings as pair
            anchors.append(indices[0])
            positives.append(indices[1])

        if len(anchors) < 2:
            return None

        anchors = torch.stack(anchors)
        positives = torch.stack(positives)

        h_a = self.encoder(batch[anchors])
        h_p = self.encoder(batch[positives])
        z_a = self.projector(h_a)
        z_p = self.projector(h_p)

        return nt_xent_loss(z_a, z_p, self.temperature)

    def train_step(
        self,
        batch: torch.Tensor,
        patient_ids: Optional[torch.Tensor] = None,
    ) -> Dict[str, float]:
        """
        One training step.

        Parameters
        ----------
        batch : torch.Tensor
            Shape (B, 12, 5000).
        patient_ids : torch.Tensor, optional
            Shape (B,). Patient IDs for patient contrastive loss.

        Returns
        -------
        losses : dict with 'total', 'temporal', 'spatial', 'patient' keys.
        """
        self.encoder.train()
        self.projector.train()

        batch = batch.to(self.device)
        self.optimizer.zero_grad()

        with torch.amp.autocast('cuda', enabled=self.use_amp):
            loss_t = self._temporal_loss(batch)
            loss_s = self._spatial_loss(batch)

            loss_total = self.lambda_temporal * loss_t + self.lambda_spatial * loss_s

            loss_p_val = 0.0
            if patient_ids is not None:
                patient_ids = patient_ids.to(self.device)
                loss_p = self._patient_loss(batch, patient_ids)
                if loss_p is not None:
                    loss_total = loss_total + self.lambda_patient * loss_p
                    loss_p_val = loss_p.item()

        if self.scaler:
            self.scaler.scale(loss_total).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            loss_total.backward()
            self.optimizer.step()

        return {
            'total': loss_total.item(),
            'temporal': loss_t.item(),
            'spatial': loss_s.item(),
            'patient': loss_p_val,
        }
