# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from collections.abc import Sequence
from pathlib import Path
from typing import Any, ClassVar

import numpy as np
import pytorch_lightning as pl
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig
from torch import nn
from torch.utils.data import ConcatDataset, DataLoader
from torchmetrics import MetricCollection

from emg2qwerty import utils
from emg2qwerty.charset import charset
from emg2qwerty.data import LabelData, WindowedEMGDataset
from emg2qwerty.metrics import CharacterErrorRates
from emg2qwerty.modules import (
    MultiBandRotationInvariantMLP,
    SpectrogramNorm,
    TDSConvEncoder,
    TDSConvEncoderWithDropout,
    TDSConvEncoderNoSkip,
    BiLSTMBlock
)
from emg2qwerty.transforms import Transform


class WindowedEMGDataModule(pl.LightningDataModule):
    def __init__(
        self,
        window_length: int,
        padding: tuple[int, int],
        batch_size: int,
        num_workers: int,
        train_sessions: Sequence[Path],
        val_sessions: Sequence[Path],
        test_sessions: Sequence[Path],
        train_transform: Transform[np.ndarray, torch.Tensor],
        val_transform: Transform[np.ndarray, torch.Tensor],
        test_transform: Transform[np.ndarray, torch.Tensor],
    ) -> None:
        super().__init__()

        self.window_length = window_length
        self.padding = padding

        self.batch_size = batch_size
        self.num_workers = num_workers

        self.train_sessions = train_sessions
        self.val_sessions = val_sessions
        self.test_sessions = test_sessions

        self.train_transform = train_transform
        self.val_transform = val_transform
        self.test_transform = test_transform

    def setup(self, stage: str | None = None) -> None:
        self.train_dataset = ConcatDataset(
            [
                WindowedEMGDataset(
                    hdf5_path,
                    transform=self.train_transform,
                    window_length=self.window_length,
                    padding=self.padding,
                    jitter=True,
                )
                for hdf5_path in self.train_sessions
            ]
        )
        self.val_dataset = ConcatDataset(
            [
                WindowedEMGDataset(
                    hdf5_path,
                    transform=self.val_transform,
                    window_length=self.window_length,
                    padding=self.padding,
                    jitter=False,
                )
                for hdf5_path in self.val_sessions
            ]
        )
        self.test_dataset = ConcatDataset(
            [
                WindowedEMGDataset(
                    hdf5_path,
                    transform=self.test_transform,
                    # Feed the entire session at once without windowing/padding
                    # at test time for more realism
                    window_length=None,
                    padding=(0, 0),
                    jitter=False,
                )
                for hdf5_path in self.test_sessions
            ]
        )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=WindowedEMGDataset.collate,
            pin_memory=True,
            persistent_workers=True,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=WindowedEMGDataset.collate,
            pin_memory=True,
            persistent_workers=True,
        )

    def test_dataloader(self) -> DataLoader:
        # Test dataset does not involve windowing and entire sessions are
        # fed at once. Limit batch size to 1 to fit within GPU memory and
        # avoid any influence of padding (while collating multiple batch items)
        # in test scores.
        return DataLoader(
            self.test_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=WindowedEMGDataset.collate,
            pin_memory=True,
            persistent_workers=True,
        )


class TDSConvCTCModule(pl.LightningModule):
    NUM_BANDS: ClassVar[int] = 2
    ELECTRODE_CHANNELS: ClassVar[int] = 16

    def __init__(
        self,
        in_features: int,
        mlp_features: Sequence[int],
        block_channels: Sequence[int],
        kernel_width: int,
        optimizer: DictConfig,
        lr_scheduler: DictConfig,
        decoder: DictConfig,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        num_features = self.NUM_BANDS * mlp_features[-1]

        # Model
        # inputs: (T, N, bands=2, electrode_channels=16, freq)
        self.model = nn.Sequential(
            # (T, N, bands=2, C=16, freq)
            SpectrogramNorm(channels=self.NUM_BANDS * self.ELECTRODE_CHANNELS),
            # (T, N, bands=2, mlp_features[-1])
            MultiBandRotationInvariantMLP(
                in_features=in_features,
                mlp_features=mlp_features,
                num_bands=self.NUM_BANDS,
            ),
            # (T, N, num_features)
            nn.Flatten(start_dim=2),
            TDSConvEncoder(
                num_features=num_features,
                block_channels=block_channels,
                kernel_width=kernel_width,
            ),
            # (T, N, num_classes)
            nn.Linear(num_features, charset().num_classes),
            nn.LogSoftmax(dim=-1),
        )

        # Criterion
        self.ctc_loss = nn.CTCLoss(blank=charset().null_class)

        # Decoder
        self.decoder = instantiate(decoder)

        # Metrics
        metrics = MetricCollection([CharacterErrorRates()])
        self.metrics = nn.ModuleDict(
            {
                f"{phase}_metrics": metrics.clone(prefix=f"{phase}/")
                for phase in ["train", "val", "test"]
            }
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.model(inputs)

    def _step(
        self, phase: str, batch: dict[str, torch.Tensor], *args, **kwargs
    ) -> torch.Tensor:
        inputs = batch["inputs"]
        targets = batch["targets"]
        input_lengths = batch["input_lengths"]
        target_lengths = batch["target_lengths"]
        N = len(input_lengths)  # batch_size

        emissions = self.forward(inputs)

        # Shrink input lengths by an amount equivalent to the conv encoder's
        # temporal receptive field to compute output activation lengths for CTCLoss.
        # NOTE: This assumes the encoder doesn't perform any temporal downsampling
        # such as by striding.
        T_diff = inputs.shape[0] - emissions.shape[0]
        emission_lengths = input_lengths - T_diff

        loss = self.ctc_loss(
            log_probs=emissions,  # (T, N, num_classes)
            targets=targets.transpose(0, 1),  # (T, N) -> (N, T)
            input_lengths=emission_lengths,  # (N,)
            target_lengths=target_lengths,  # (N,)
        )

        # Decode emissions
        predictions = self.decoder.decode_batch(
            emissions=emissions.detach().cpu().numpy(),
            emission_lengths=emission_lengths.detach().cpu().numpy(),
        )

        # Update metrics
        metrics = self.metrics[f"{phase}_metrics"]
        targets = targets.detach().cpu().numpy()
        target_lengths = target_lengths.detach().cpu().numpy()
        for i in range(N):
            # Unpad targets (T, N) for batch entry
            target = LabelData.from_labels(targets[: target_lengths[i], i])
            metrics.update(prediction=predictions[i], target=target)

        self.log(f"{phase}/loss", loss, batch_size=N, sync_dist=True)
        return loss

    def _epoch_end(self, phase: str) -> None:
        metrics = self.metrics[f"{phase}_metrics"]
        self.log_dict(metrics.compute(), sync_dist=True)
        metrics.reset()

    def training_step(self, *args, **kwargs) -> torch.Tensor:
        return self._step("train", *args, **kwargs)

    def validation_step(self, *args, **kwargs) -> torch.Tensor:
        return self._step("val", *args, **kwargs)

    def test_step(self, *args, **kwargs) -> torch.Tensor:
        return self._step("test", *args, **kwargs)

    def on_train_epoch_end(self) -> None:
        self._epoch_end("train")

    def on_validation_epoch_end(self) -> None:
        self._epoch_end("val")

    def on_test_epoch_end(self) -> None:
        self._epoch_end("test")

    def configure_optimizers(self) -> dict[str, Any]:
        return utils.instantiate_optimizer_and_scheduler(
            self.parameters(),
            optimizer_config=self.hparams.optimizer,
            lr_scheduler_config=self.hparams.lr_scheduler,
        )



class DeepCNNEncoder(nn.Module):
    """A deeper CNN stack with residual connections and dropout."""
    def __init__(
        self,
        in_features: int,
        block_channels: Sequence[int],
        kernel_width: int = 9,
        dropout_rate: float = 0.2,
    ):
        super().__init__()
        initial_out = block_channels[0]
        self.input_proj = nn.Linear(in_features, initial_out) if in_features != initial_out else nn.Identity()

        blocks = []
        prev_c = initial_out
        for c in block_channels:
            blocks.append(ResConv1DBlock(prev_c, c, kernel_width, dropout_rate))
            blocks.append(nn.Dropout(p=dropout_rate))  # Dropout between blocks
            prev_c = c

        self.conv_stack = nn.Sequential(*blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_proj(x)
        x = self.conv_stack(x)
        return x


class ResConv1DBlock(nn.Module):
    """Residual 1D CNN Block with optional dropout and depthwise-separable convolutions."""
    def __init__(self, in_channels, out_channels, kernel_size, dropout_rate=0.2):
        super().__init__()
        padding = (kernel_size - 1) // 2

        self.conv = nn.Conv1d(
            in_channels, out_channels,
            kernel_size=kernel_size,
            padding=padding,
        )
        self.act = nn.ReLU()
        self.layernorm = nn.LayerNorm(out_channels)
        self.dropout = nn.Dropout(p=dropout_rate)

        # Projection for residual connection if needed
        self.skip_proj = None
        if in_channels != out_channels:
            self.skip_proj = nn.Conv1d(in_channels, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(0, 1).transpose(1, 2)  # (N, C, T)
        out = self.conv(x)
        out = self.act(out)
        out = self.dropout(out)  # Dropout before residual connection
        if self.skip_proj is not None:
            skip = self.skip_proj(x)
        else:
            skip = x
        out = out + skip  # Residual add
        out = out.transpose(1, 2).transpose(0, 1)  # (T, N, out_channels)
        out = self.layernorm(out)
        return out


class DeepCNNCTCModule(pl.LightningModule):
    """CTC Model with a deep residual CNN encoder."""
    NUM_BANDS: ClassVar[int] = 2
    ELECTRODE_CHANNELS: ClassVar[int] = 16

    def __init__(
        self,
        in_features: int,
        mlp_features: Sequence[int],
        block_channels: Sequence[int],
        kernel_width: int,
        dropout_rate: float,  
        optimizer: DictConfig,
        lr_scheduler: DictConfig,
        decoder: DictConfig,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        num_features = self.NUM_BANDS * mlp_features[-1]

        self.model = nn.Sequential(
            SpectrogramNorm(channels=self.NUM_BANDS * self.ELECTRODE_CHANNELS),
            MultiBandRotationInvariantMLP(
                in_features=in_features,
                mlp_features=mlp_features,
                num_bands=self.NUM_BANDS,
            ),
            nn.Flatten(start_dim=2),
            DeepCNNEncoder(
                in_features=num_features,
                block_channels=block_channels,
                kernel_width=kernel_width,
                dropout_rate=dropout_rate
            ),
            nn.Linear(block_channels[-1], charset().num_classes),
            nn.LogSoftmax(dim=-1),
        )

        self.ctc_loss = nn.CTCLoss(blank=charset().null_class)
        self.decoder = instantiate(decoder)

        metrics = MetricCollection([CharacterErrorRates()])
        self.metrics = nn.ModuleDict({
            f"{phase}_metrics": metrics.clone(prefix=f"{phase}/")
            for phase in ["train", "val", "test"]
        })

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.model(inputs)

    def _step(self, phase: str, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        """Handles forward pass, loss calculation, and metric updates."""
        inputs = batch["inputs"]           # shape (T, N, 2, 16, freq)
        targets = batch["targets"]         # shape (T, N)
        input_lengths = batch["input_lengths"]
        target_lengths = batch["target_lengths"]
        N = len(input_lengths)  # batch_size

        emissions = self.forward(inputs)    # shape (T, N, num_classes)
        T_diff = inputs.shape[0] - emissions.shape[0]
        emission_lengths = input_lengths - T_diff

        loss = self.ctc_loss(
            log_probs=emissions,          # (T, N, num_classes)
            targets=targets.transpose(0, 1),  # (N, T)
            input_lengths=emission_lengths,
            target_lengths=target_lengths,
        )

        # Decode emissions for metrics
        predictions = self.decoder.decode_batch(
            emissions=emissions.detach().cpu().numpy(),
            emission_lengths=emission_lengths.detach().cpu().numpy(),
        )

        metrics = self.metrics[f"{phase}_metrics"]
        tgt_cpu = targets.detach().cpu().numpy()
        tgt_lens_cpu = target_lengths.detach().cpu().numpy()
        for i in range(N):
            tgt_i = LabelData.from_labels(tgt_cpu[:tgt_lens_cpu[i], i])
            metrics.update(prediction=predictions[i], target=tgt_i)

        self.log(f"{phase}/loss", loss, batch_size=N, sync_dist=True)
        return loss
    
    def _epoch_end(self, phase: str) -> None:
        """Ensure all computed metrics, including CER, are logged at the end of an epoch."""
        metrics = self.metrics[f"{phase}_metrics"]
    
        # Log all computed metrics (ensures val/CER is logged)
        self.log_dict(metrics.compute(), sync_dist=True)

        # Reset for the next epoch
        metrics.reset()

    def training_step(self, batch, batch_idx) -> torch.Tensor:
        return self._step("train", batch)

    def validation_step(self, batch, batch_idx) -> torch.Tensor:
        return self._step("val", batch)

    def test_step(self, batch, batch_idx) -> torch.Tensor:
        return self._step("test", batch)

    def on_train_epoch_end(self) -> None:
        self._epoch_end("train")

    def on_validation_epoch_end(self) -> None:
        self._epoch_end("val")

    def on_test_epoch_end(self) -> None:
        self._epoch_end("test")

    def configure_optimizers(self) -> dict[str, Any]:
        return utils.instantiate_optimizer_and_scheduler(
            self.parameters(),
            optimizer_config=self.hparams.optimizer,
            lr_scheduler_config=self.hparams.lr_scheduler,
        )


class TDSConvCTCModuleWithDropout(pl.LightningModule):
    """CTC Model using the original TDS architecture but with dropout and skip connections
    integrated into the encoder blocks.
    
    The overall architecture is identical to the original TDSConvCTCModule:
      SpectrogramNorm -> MultiBandRotationInvariantMLP -> Flatten ->
      TDSConvEncoderWithDropout -> Linear -> LogSoftmax
    
    This module plugs into the same Hydra configuration system.
    """
    NUM_BANDS: ClassVar[int] = 2
    ELECTRODE_CHANNELS: ClassVar[int] = 16

    def __init__(
        self,
        in_features: int,
        mlp_features: Sequence[int],
        block_channels: Sequence[int],
        kernel_width: int,
        dropout_rate: float,
        optimizer: DictConfig,
        lr_scheduler: DictConfig,
        decoder: DictConfig,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        num_features = self.NUM_BANDS * mlp_features[-1]

        self.model = nn.Sequential(
            SpectrogramNorm(channels=self.NUM_BANDS * self.ELECTRODE_CHANNELS),
            MultiBandRotationInvariantMLP(
                in_features=in_features,
                mlp_features=mlp_features,
                num_bands=self.NUM_BANDS,
            ),
            nn.Flatten(start_dim=2),
            TDSConvEncoderWithDropout(
                num_features=num_features,
                block_channels=block_channels,
                kernel_width=kernel_width,
                dropout_rate=dropout_rate
            ),
            nn.Linear(num_features, charset().num_classes),
            nn.LogSoftmax(dim=-1),
        )

        self.ctc_loss = nn.CTCLoss(blank=charset().null_class)
        self.decoder = instantiate(decoder)

        metrics = MetricCollection([CharacterErrorRates()])
        self.metrics = nn.ModuleDict({
            f"{phase}_metrics": metrics.clone(prefix=f"{phase}/")
            for phase in ["train", "val", "test"]
        })

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.model(inputs)

    def _step(self, phase: str, batch: dict[str, torch.Tensor], *args, **kwargs) -> torch.Tensor:
        inputs = batch["inputs"]
        targets = batch["targets"]
        input_lengths = batch["input_lengths"]
        target_lengths = batch["target_lengths"]
        N = len(input_lengths)

        emissions = self.forward(inputs)
        T_diff = inputs.shape[0] - emissions.shape[0]
        emission_lengths = input_lengths - T_diff

        loss = self.ctc_loss(
            log_probs=emissions,
            targets=targets.transpose(0, 1),
            input_lengths=emission_lengths,
            target_lengths=target_lengths,
        )

        predictions = self.decoder.decode_batch(
            emissions=emissions.detach().cpu().numpy(),
            emission_lengths=emission_lengths.detach().cpu().numpy(),
        )

        metrics = self.metrics[f"{phase}_metrics"]
        tgt_cpu = targets.detach().cpu().numpy()
        tgt_lens_cpu = target_lengths.detach().cpu().numpy()
        for i in range(N):
            target = LabelData.from_labels(tgt_cpu[:tgt_lens_cpu[i], i])
            metrics.update(prediction=predictions[i], target=target)

        self.log(f"{phase}/loss", loss, batch_size=N, sync_dist=True)
        return loss

    def _epoch_end(self, phase: str) -> None:
        metrics = self.metrics[f"{phase}_metrics"]
        self.log_dict(metrics.compute(), sync_dist=True)
        metrics.reset()

    def training_step(self, *args, **kwargs) -> torch.Tensor:
        return self._step("train", *args, **kwargs)

    def validation_step(self, *args, **kwargs) -> torch.Tensor:
        return self._step("val", *args, **kwargs)

    def test_step(self, *args, **kwargs) -> torch.Tensor:
        return self._step("test", *args, **kwargs)

    def on_train_epoch_end(self) -> None:
        self._epoch_end("train")

    def on_validation_epoch_end(self) -> None:
        self._epoch_end("val")

    def on_test_epoch_end(self) -> None:
        self._epoch_end("test")

    def configure_optimizers(self) -> dict[str, Any]:
        return utils.instantiate_optimizer_and_scheduler(
            self.parameters(),
            optimizer_config=self.hparams.optimizer,
            lr_scheduler_config=self.hparams.lr_scheduler,
        )
    
class TDSConvCTCModuleNoSkip(pl.LightningModule):
    """
    CTC Model using the TDS architecture, but with dropout added and **no skip connections**.
    This maintains the overall pipeline:
      SpectrogramNorm -> MultiBandRotationInvariantMLP -> Flatten ->
      TDSConvEncoderNoSkip -> Linear -> LogSoftmax
    """
    NUM_BANDS: ClassVar[int] = 2
    ELECTRODE_CHANNELS: ClassVar[int] = 16

    def __init__(
        self,
        in_features: int,
        mlp_features: Sequence[int],
        block_channels: Sequence[int],
        kernel_width: int,
        dropout_rate: float,
        optimizer: DictConfig,
        lr_scheduler: DictConfig,
        decoder: DictConfig,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        num_features = self.NUM_BANDS * mlp_features[-1]
        self.model = nn.Sequential(
            SpectrogramNorm(channels=self.NUM_BANDS * self.ELECTRODE_CHANNELS),
            MultiBandRotationInvariantMLP(
                in_features=in_features,
                mlp_features=mlp_features,
                num_bands=self.NUM_BANDS,
            ),
            nn.Flatten(start_dim=2),
            TDSConvEncoderNoSkip(
                num_features=num_features,
                block_channels=block_channels,
                kernel_width=kernel_width,
                dropout_rate=dropout_rate
            ),
            nn.Linear(num_features, charset().num_classes),
            nn.LogSoftmax(dim=-1),
        )
        self.ctc_loss = nn.CTCLoss(blank=charset().null_class)
        self.decoder = instantiate(decoder)
        metrics = MetricCollection([CharacterErrorRates()])
        self.metrics = nn.ModuleDict({
            f"{phase}_metrics": metrics.clone(prefix=f"{phase}/")
            for phase in ["train", "val", "test"]
        })

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.model(inputs)

    def _step(self, phase: str, batch: dict[str, torch.Tensor], *args, **kwargs) -> torch.Tensor:
        inputs = batch["inputs"]
        targets = batch["targets"]
        input_lengths = batch["input_lengths"]
        target_lengths = batch["target_lengths"]
        N = len(input_lengths)
        emissions = self.forward(inputs)
        T_diff = inputs.shape[0] - emissions.shape[0]
        emission_lengths = input_lengths - T_diff
        loss = self.ctc_loss(
            log_probs=emissions,
            targets=targets.transpose(0, 1),
            input_lengths=emission_lengths,
            target_lengths=target_lengths,
        )
        predictions = self.decoder.decode_batch(
            emissions=emissions.detach().cpu().numpy(),
            emission_lengths=emission_lengths.detach().cpu().numpy(),
        )
        metrics = self.metrics[f"{phase}_metrics"]
        tgt_cpu = targets.detach().cpu().numpy()
        tgt_lens_cpu = target_lengths.detach().cpu().numpy()
        for i in range(N):
            target = LabelData.from_labels(tgt_cpu[:tgt_lens_cpu[i], i])
            metrics.update(prediction=predictions[i], target=target)
        self.log(f"{phase}/loss", loss, batch_size=N, sync_dist=True)
        return loss

    def _epoch_end(self, phase: str) -> None:
        metrics = self.metrics[f"{phase}_metrics"]
        self.log_dict(metrics.compute(), sync_dist=True)
        metrics.reset()

    def training_step(self, *args, **kwargs) -> torch.Tensor:
        return self._step("train", *args, **kwargs)

    def validation_step(self, *args, **kwargs) -> torch.Tensor:
        return self._step("val", *args, **kwargs)

    def test_step(self, *args, **kwargs) -> torch.Tensor:
        return self._step("test", *args, **kwargs)

    def on_train_epoch_end(self) -> None:
        self._epoch_end("train")

    def on_validation_epoch_end(self) -> None:
        self._epoch_end("val")

    def on_test_epoch_end(self) -> None:
        self._epoch_end("test")

    def configure_optimizers(self) -> dict[str, Any]:
        return utils.instantiate_optimizer_and_scheduler(
            self.parameters(),
            optimizer_config=self.hparams.optimizer,
            lr_scheduler_config=self.hparams.lr_scheduler,
        )
    

class TDSBiLSTMCTCModuleWithDropout(pl.LightningModule):
    NUM_BANDS: ClassVar[int] = 2
    ELECTRODE_CHANNELS: ClassVar[int] = 16

    def __init__(
        self,
        in_features: int,
        mlp_features: Sequence[int],
        block_channels: Sequence[int],
        kernel_width: int,
        dropout_rate: float,
        lstm_hidden: int,
        lstm_layers: int,
        optimizer: DictConfig,
        lr_scheduler: DictConfig,
        decoder: DictConfig,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        # After the MLP, the output shape is (T, N, NUM_BANDS, mlp_features[-1])
        # Flattening gives (T, N, lstm_in_dim) where lstm_in_dim = NUM_BANDS * mlp_features[-1]
        lstm_in_dim = self.NUM_BANDS * mlp_features[-1]
        
        self.model = nn.Sequential(
            SpectrogramNorm(channels=self.NUM_BANDS * self.ELECTRODE_CHANNELS),
            MultiBandRotationInvariantMLP(
                in_features=in_features,
                mlp_features=mlp_features,
                num_bands=self.NUM_BANDS,
            ),
            nn.Flatten(start_dim=2),  # shape: (T, N, lstm_in_dim)
            TDSConvEncoderWithDropout(
                num_features=lstm_in_dim,
                block_channels=block_channels,
                kernel_width=kernel_width,
                dropout_rate=dropout_rate
            )
        )
        
        self.bilstm = BiLSTMBlock(
            input_dim=lstm_in_dim,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            dropout_rate=dropout_rate
        )
        self.classifier = nn.Sequential(
            nn.Linear(2 * lstm_hidden, charset().num_classes),
            nn.LogSoftmax(dim=-1)
        )
        
        self.ctc_loss = nn.CTCLoss(blank=charset().null_class)
        self.decoder = instantiate(decoder)
        metrics = MetricCollection([CharacterErrorRates()])
        self.metrics = nn.ModuleDict({
            f"{phase}_metrics": metrics.clone(prefix=f"{phase}/")
            for phase in ["train", "val", "test"]
        })

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        # inputs shape: (T, N, num_bands, electrode_channels, frequency_bins)
        x = self.model(inputs)        # x shape: (T, N, lstm_in_dim)
        x = self.bilstm(x)            # x shape: (T, N, 2 * lstm_hidden)
        x = self.classifier(x)        # x shape: (T, N, num_classes)
        return x

    def _step(self, phase: str, batch: dict[str, torch.Tensor], *args, **kwargs) -> torch.Tensor:
        inputs = batch["inputs"]
        targets = batch["targets"]
        input_lengths = batch["input_lengths"]
        target_lengths = batch["target_lengths"]
        N = len(input_lengths)
        emissions = self.forward(inputs)
        T_diff = inputs.shape[0] - emissions.shape[0]
        emission_lengths = input_lengths - T_diff
        loss = self.ctc_loss(
            log_probs=emissions,
            targets=targets.transpose(0,1),
            input_lengths=emission_lengths,
            target_lengths=target_lengths,
        )
        predictions = self.decoder.decode_batch(
            emissions=emissions.detach().cpu().numpy(),
            emission_lengths=emission_lengths.detach().cpu().numpy(),
        )
        metrics = self.metrics[f"{phase}_metrics"]
        tgt_cpu = targets.detach().cpu().numpy()
        tgt_lens_cpu = target_lengths.detach().cpu().numpy()
        for i in range(N):
            target = LabelData.from_labels(tgt_cpu[:tgt_lens_cpu[i], i])
            metrics.update(prediction=predictions[i], target=target)
        self.log(f"{phase}/loss", loss, batch_size=N, sync_dist=True)
        return loss

    def _epoch_end(self, phase: str) -> None:
        metrics = self.metrics[f"{phase}_metrics"]
        self.log_dict(metrics.compute(), sync_dist=True)
        metrics.reset()

    def training_step(self, *args, **kwargs) -> torch.Tensor:
        return self._step("train", *args, **kwargs)

    def validation_step(self, *args, **kwargs) -> torch.Tensor:
        return self._step("val", *args, **kwargs)

    def test_step(self, *args, **kwargs) -> torch.Tensor:
        return self._step("test", *args, **kwargs)

    def on_train_epoch_end(self) -> None:
        self._epoch_end("train")

    def on_validation_epoch_end(self) -> None:
        self._epoch_end("val")

    def on_test_epoch_end(self) -> None:
        self._epoch_end("test")

    def configure_optimizers(self) -> dict[str, Any]:
        return utils.instantiate_optimizer_and_scheduler(
            self.parameters(),
            optimizer_config=self.hparams.optimizer,
            lr_scheduler_config=self.hparams.lr_scheduler,
        )
    


class SqueezeExcite(nn.Module):
    """
    Learns per-channel scaling weights via global average pooling over time.
    """
    def __init__(self, num_channels: int, reduction_ratio: int = 8):
        super().__init__()
        reduced_dim = max(num_channels // reduction_ratio, 4)
        self.fc = nn.Sequential(
            nn.Linear(num_channels, reduced_dim, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(reduced_dim, num_channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x shape: (T, N, C). We pool over T => (N, C), learn weights, then scale x.
        """
        T, N, C = x.shape
        x_pooled = x.mean(dim=0)  # (N, C)
        w = self.fc(x_pooled)     # (N, C)
        w = w.unsqueeze(0)        # (1, N, C)
        return x * w             # broadcast scale on each channel


class TDSConv2dBlockWithDropoutSE(nn.Module):

    def __init__(self, channels: int, width: int, kernel_width: int, dropout_rate: float = 0.3):
        super().__init__()
        # We rely on your existing TDSConv2dBlockWithDropout with skip connections
        from emg2qwerty.modules import TDSConv2dBlockWithDropout
        self.tds_conv = TDSConv2dBlockWithDropout(channels, width, kernel_width, dropout_rate)
        self.se = SqueezeExcite(channels * width, reduction_ratio=8)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        # Normal TDS conv+skip
        x = self.tds_conv(inputs)
        # Then apply S/E
        x = self.se(x)
        return x

class TDSConvEncoderWithDropoutSE(nn.Module):

    def __init__(
        self,
        num_features: int,
        block_channels: Sequence[int],
        kernel_width: int,
        dropout_rate: float,
    ) -> None:
        super().__init__()
        from emg2qwerty.modules import TDSConv2dBlockWithDropout, TDSFullyConnectedBlockWithDropout

        blocks = []
        for c in block_channels:
            assert num_features % c == 0, "block_channels must evenly divide num_features"
            width = num_features // c
            # TDSConv2dBlockWithDropout = conv + skip + dropout
            blocks.append(TDSConv2dBlockWithDropout(c, width, kernel_width, dropout_rate))
            # Then we insert an SE step. c*width = total channels in (T, N, c*width).
            blocks.append(SqueezeExcite(c * width))
            # TDSFullyConnectedBlockWithDropout = two linear layers + skip
            blocks.append(TDSFullyConnectedBlockWithDropout(num_features, dropout_rate))

        self.tds_blocks = nn.Sequential(*blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.tds_blocks(x)
    
class PostLSTMMultiHeadAttention(nn.Module):
    """
    After BiLSTM, we have shape (T, N, 2*h). We do a multi-head self-attention 
    that reweights each time step using global context. 
    This is a minimal overhead approach with big gains. 
    """
    def __init__(self, embed_dim: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.mha = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=False)
        self.layer_norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (T, N, embed_dim)
        # We do standard multi-head self-attn
        attn_out, _ = self.mha(x, x, x)  # no mask
        out = x + self.dropout(attn_out)
        out = self.layer_norm(out)
        return out


class TDSBiLSTMAttnSECTCModule(pl.LightningModule):
    """
    1) SpectrogramNorm
    2) MultiBandRotationInvariantMLP
    3) Flatten
    4) TDS with S/E
    5) BiLSTM
    6) Post-LSTM MultiHead Self-Attn
    7) Linear -> LogSoftmax
    8) CTC loss
    """
    NUM_BANDS: ClassVar[int] = 2
    ELECTRODE_CHANNELS: ClassVar[int] = 16

    def __init__(
        self,
        in_features: int,
        mlp_features: Sequence[int],
        block_channels: Sequence[int],
        kernel_width: int,
        dropout_rate: float,
        lstm_hidden: int,
        lstm_layers: int,
        num_heads: int,
        optimizer: DictConfig,
        lr_scheduler: DictConfig,
        decoder: DictConfig,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        from emg2qwerty.modules import SpectrogramNorm, MultiBandRotationInvariantMLP, BiLSTMBlock
        from emg2qwerty.charset import charset
        from torchmetrics import MetricCollection
        from emg2qwerty.metrics import CharacterErrorRates

        # 1) Input normalization
        self.norm = SpectrogramNorm(channels=self.NUM_BANDS * self.ELECTRODE_CHANNELS)
        # 2) MLP
        self.mlp = MultiBandRotationInvariantMLP(
            in_features=in_features,
            mlp_features=mlp_features,
            num_bands=self.NUM_BANDS,
        )
        # Flatten => after MLP we have shape (T, N, num_bands, mlp_features[-1])
        # so the dimension is num_bands * mlp_features[-1]
        self.num_features = self.NUM_BANDS * mlp_features[-1]

        # 3) TDS with S/E
        self.tds_encoder = TDSConvEncoderWithDropoutSE(
            num_features=self.num_features,
            block_channels=block_channels,
            kernel_width=kernel_width,
            dropout_rate=dropout_rate,
        )

        # 4) BiLSTM
        self.bilstm = BiLSTMBlock(
            input_dim=self.num_features,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            dropout_rate=dropout_rate,
        )
        # output of BiLSTM is shape (T, N, 2*lstm_hidden)

        # 5) Multi-Head Attn
        self.post_lstm_attn = PostLSTMMultiHeadAttention(embed_dim=2*lstm_hidden, num_heads=num_heads, dropout=dropout_rate)

        # 6) Classifier
        self.classifier = nn.Sequential(
            nn.Linear(2 * lstm_hidden, charset().num_classes),
            nn.LogSoftmax(dim=-1),
        )
        # CTC
        self.ctc_loss = nn.CTCLoss(blank=charset().null_class)
        self.decoder = instantiate(decoder)

        # Metrics
        metrics = MetricCollection([CharacterErrorRates()])
        self.metrics = nn.ModuleDict({
            f"{phase}_metrics": metrics.clone(prefix=f"{phase}/")
            for phase in ["train", "val", "test"]
        })

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        inputs shape: (T, N, bands=2, electrode_channels=16, freq)
        """
        x = self.norm(inputs)    # (T, N, 2, 16, freq)
        x = self.mlp(x)          # (T, N, 2, mlp_features[-1])
        x = x.flatten(start_dim=2)  # (T, N, num_bands*mlp_features[-1])
        x = self.tds_encoder(x)  # (T, N, num_features)
        x = self.bilstm(x)       # (T, N, 2*lstm_hidden)
        x = self.post_lstm_attn(x)  # (T, N, 2*lstm_hidden), reweighted
        x = self.classifier(x)   # (T, N, vocab_size)
        return x

    def _step(self, phase: str, batch: dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """
        Common step for train/val/test, must accept batch_idx as well.
        """

        inputs = batch["inputs"]
        targets = batch["targets"]
        input_lengths = batch["input_lengths"]
        target_lengths = batch["target_lengths"]
        N = len(input_lengths)

        emissions = self.forward(inputs)  # (T, N, vocab_size)
        # If TDS downsamples, T_out <= T_in
        T_diff = inputs.shape[0] - emissions.shape[0]
        emission_lengths = input_lengths - T_diff

        loss = self.ctc_loss(
            log_probs=emissions,
            targets=targets.transpose(0, 1),
            input_lengths=emission_lengths,
            target_lengths=target_lengths,
        )

        # Decode for CER
        preds = self.decoder.decode_batch(
            emissions=emissions.detach().cpu().numpy(),
            emission_lengths=emission_lengths.detach().cpu().numpy(),
        )
        metrics = self.metrics[f"{phase}_metrics"]

        tgt_cpu = targets.detach().cpu().numpy()
        tgt_lens_cpu = target_lengths.detach().cpu().numpy()
        for i in range(N):
            target = LabelData.from_labels(tgt_cpu[:tgt_lens_cpu[i], i])
            metrics.update(prediction=preds[i], target=target)

        self.log(f"{phase}/loss", loss, batch_size=N, sync_dist=True)
        return loss

    def training_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        return self._step("train", batch, batch_idx)

    def validation_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        return self._step("val", batch, batch_idx)

    def test_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        return self._step("test", batch, batch_idx)

    def _epoch_end(self, phase: str) -> None:
        metrics = self.metrics[f"{phase}_metrics"]
        self.log_dict(metrics.compute(), sync_dist=True)
        metrics.reset()

    def on_train_epoch_end(self) -> None:
        self._epoch_end("train")

    def on_validation_epoch_end(self) -> None:
        self._epoch_end("val")

    def on_test_epoch_end(self) -> None:
        self._epoch_end("test")

    def configure_optimizers(self) -> dict[str, Any]:
        from emg2qwerty import utils
        return utils.instantiate_optimizer_and_scheduler(
            self.parameters(),
            optimizer_config=self.hparams.optimizer,
            lr_scheduler_config=self.hparams.lr_scheduler,
        )
    




class SqueezeExcite(nn.Module):

    def __init__(self, num_channels: int, reduction_ratio: int = 8):
        super().__init__()
        reduced_dim = max(num_channels // reduction_ratio, 4)
        self.fc = nn.Sequential(
            nn.Linear(num_channels, reduced_dim, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(reduced_dim, num_channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x shape: (T, N, C). We pool over T => (N, C), learn weights, then scale x.
        """
        T, N, C = x.shape
        x_pooled = x.mean(dim=0)  # (N, C)
        w = self.fc(x_pooled)     # (N, C)
        w = w.unsqueeze(0)        # (1, N, C)
        return x * w             # broadcast scale on each channel



class TDSConvEncoderWithDropoutSE(nn.Module):

    def __init__(
        self,
        num_features: int,
        block_channels: Sequence[int],
        kernel_width: int,
        dropout_rate: float,
    ) -> None:
        super().__init__()
        # We'll replicate TDSConvEncoderWithDropout's logic, but add S/E
        from emg2qwerty.modules import TDSConv2dBlockWithDropout, TDSFullyConnectedBlockWithDropout

        blocks = []
        for c in block_channels:
            assert num_features % c == 0
            width = num_features // c
            # TDSConv2dBlockWithDropout = conv + skip + dropout
            blocks.append(TDSConv2dBlockWithDropout(c, width, kernel_width, dropout_rate))
            # Then we insert an SE step. c*width = total channels in (T, N, c*width).
            blocks.append(SqueezeExcite(c * width))
            # TDSFullyConnectedBlockWithDropout = two linear layers + skip
            blocks.append(TDSFullyConnectedBlockWithDropout(num_features, dropout_rate))

        self.tds_blocks = nn.Sequential(*blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.tds_blocks(x)


class TDSBiLSTMCTCModuleWithDropoutSE(pl.LightningModule):

    NUM_BANDS: ClassVar[int] = 2
    ELECTRODE_CHANNELS: ClassVar[int] = 16

    def __init__(
        self,
        in_features: int,
        mlp_features: Sequence[int],
        block_channels: Sequence[int],
        kernel_width: int,
        dropout_rate: float,
        lstm_hidden: int,
        lstm_layers: int,
        optimizer: DictConfig,
        lr_scheduler: DictConfig,
        decoder: DictConfig,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        # After MLP => shape (T, N, NUM_BANDS, mlp_features[-1])
        # Flatten => (T, N, lstm_in_dim)
        lstm_in_dim = self.NUM_BANDS * mlp_features[-1]

        # 1) The front-end (same as baseline, but TDSConvEncoderWithDropoutSE)
        self.model = nn.Sequential(
            SpectrogramNorm(channels=self.NUM_BANDS * self.ELECTRODE_CHANNELS),
            MultiBandRotationInvariantMLP(
                in_features=in_features,
                mlp_features=mlp_features,
                num_bands=self.NUM_BANDS,
            ),
            nn.Flatten(start_dim=2),
            TDSConvEncoderWithDropoutSE(
                num_features=lstm_in_dim,
                block_channels=block_channels,
                kernel_width=kernel_width,
                dropout_rate=dropout_rate,
            ),
        )

        # 2) BiLSTM
        self.bilstm = BiLSTMBlock(
            input_dim=lstm_in_dim,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            dropout_rate=dropout_rate,
        )

        # 3) Classifier
        self.classifier = nn.Sequential(
            nn.Linear(2 * lstm_hidden, charset().num_classes),
            nn.LogSoftmax(dim=-1),
        )

        # 4) CTC Loss
        self.ctc_loss = nn.CTCLoss(blank=charset().null_class)

        # 5) Decoder + Metrics
        self.decoder = instantiate(decoder)
        metrics = MetricCollection([CharacterErrorRates()])
        self.metrics = nn.ModuleDict({
            f"{phase}_metrics": metrics.clone(prefix=f"{phase}/")
            for phase in ["train", "val", "test"]
        })

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        # inputs shape: (T, N, bands=2, electrode_channels=16, freq)
        x = self.model(inputs)   # => (T, N, lstm_in_dim)
        x = self.bilstm(x)       # => (T, N, 2*lstm_hidden)
        x = self.classifier(x)   # => (T, N, num_classes)
        return x

    def _step(self, phase: str, batch: dict[str, torch.Tensor], *args, **kwargs) -> torch.Tensor:
        inputs = batch["inputs"]
        targets = batch["targets"]
        input_lengths = batch["input_lengths"]
        target_lengths = batch["target_lengths"]
        N = len(input_lengths)

        emissions = self.forward(inputs)   # shape (T_out, N, vocab)
        T_diff = inputs.shape[0] - emissions.shape[0]
        emission_lengths = input_lengths - T_diff

        loss = self.ctc_loss(
            log_probs=emissions,
            targets=targets.transpose(0, 1),
            input_lengths=emission_lengths,
            target_lengths=target_lengths,
        )

        # Decode for CER
        preds = self.decoder.decode_batch(
            emissions=emissions.detach().cpu().numpy(),
            emission_lengths=emission_lengths.detach().cpu().numpy(),
        )
        m = self.metrics[f"{phase}_metrics"]

        tgt_cpu = targets.detach().cpu().numpy()
        tgt_lens_cpu = target_lengths.detach().cpu().numpy()
        for i in range(N):
            lbl_data = LabelData.from_labels(tgt_cpu[:tgt_lens_cpu[i], i])
            m.update(prediction=preds[i], target=lbl_data)

        self.log(f"{phase}/loss", loss, batch_size=N, sync_dist=True)
        return loss

    def training_step(self, *args, **kwargs) -> torch.Tensor:
        return self._step("train", *args, **kwargs)

    def validation_step(self, *args, **kwargs) -> torch.Tensor:
        return self._step("val", *args, **kwargs)

    def test_step(self, *args, **kwargs) -> torch.Tensor:
        return self._step("test", *args, **kwargs)

    def _epoch_end(self, phase: str) -> None:
        m = self.metrics[f"{phase}_metrics"]
        self.log_dict(m.compute(), sync_dist=True)
        m.reset()

    def on_train_epoch_end(self) -> None:
        self._epoch_end("train")

    def on_validation_epoch_end(self) -> None:
        self._epoch_end("val")

    def on_test_epoch_end(self) -> None:
        self._epoch_end("test")

    def configure_optimizers(self) -> Any:
        return utils.instantiate_optimizer_and_scheduler(
            self.parameters(),
            optimizer_config=self.hparams.optimizer,
            lr_scheduler_config=self.hparams.lr_scheduler,
        )
