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
    #TDSConvEncoderwithDropout,
    TDSConvEncoderNoSkip
)
from emg2qwerty.modules import TDSConvEncoderWithDropout  # Ensure capitalization is correct
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
        hidden_size: int, # Added GRU hidden size
        num_layers: int, # Number of GRU Layers
        optimizer: DictConfig,
        lr_scheduler: DictConfig,
        decoder: DictConfig,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        num_features = self.NUM_BANDS * mlp_features[-1]

        ### ORIGINAL CODE MODEL ###
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
        # #IGNORE THIS
        # self.cnn = nn.Sequential(
        #     SpectrogramNorm(channels=self.NUM_BANDS * self.ELECTRODE_CHANNELS),
        #     MultiBandRotationInvariantMLP( in_features=in_features, mlp_features=mlp_features, 
        #     num_bands=self.NUM_BANDS,
        #     ),
        #     nn.Flatten(start_dim=2),
        #     TDSConvEncoder(
        #         num_features=num_features,
        #         block_channels=block_channels,
        #         kernel_width=kernel_width,
        #     ),
        # )

        # # Feature extraction layers
        # self.feature_extractor = nn.Sequential(
        #     SpectrogramNorm(channels=self.NUM_BANDS * self.ELECTRODE_CHANNELS),
        #     MultiBandRotationInvariantMLP(
        #         in_features=in_features,
        #         mlp_features=mlp_features,
        #         num_bands=self.NUM_BANDS,
        #     ),
        #     nn.Flatten(start_dim=2),  # (T, N, num_features)
        # )
        # self.cnn = nn.Sequential(
        #     nn.Conv1d(in_channels=num_features, out_channels=block_channels[0], kernel_size=kernel_width, padding=kernel_width // 2),
        #     nn.BatchNorm1d(block_channels[0]), # Batchnorm added 3/7
        #     nn.ReLU(),
        #     nn.Conv1d(in_channels=block_channels[0], out_channels=block_channels[1], kernel_size=kernel_width, padding=kernel_width // 2),
        #     nn.BatchNorm1d(block_channels[1]), #Batchnorm added
        #     nn.ReLU()
        # )

        # self.gru = nn.GRU(

        #     input_size=block_channels[1],  # Match output from CNN encoder
        #     hidden_size=hidden_size,  # Hidden dimension for GRU
        #     num_layers=num_layers,  # Stacked GRU layers
        #     batch_first=False,  # Input shape: (T, N, C)
        #     bidirectional=True,  # Enables BiGRU
        #     dropout=0 if num_layers == 1 else 0.3,
        #     #dropout = 0.3 # added on 3/6/2025 after reducing to 64 hidden size - note it didn't help if anything it made it worse
        # )

        # self.layer_norm = nn.LayerNorm(2 * hidden_size)

        #Added Final Linear Projection Layer (Bidirectional GRU outputs 2x hidden_size)
        # self.fc = nn.Linear(2 * hidden_size, charset().num_classes)

        # # Combine all into self.model ## NEEDED THIS BECAUSE OF A BIG ERROR
        # self.model = nn.Sequential(
        #     self.feature_extractor,
        #     self.cnn,
        #     self.gru,
        #     self.fc,
        # )

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
        # #return self.model(inputs) # ORIGINAL CODE
        # x = self.feature_extractor(inputs)  # Shape: (T, N, num_features)

        # # Permute for CNN (Batch First format)
        # x = x.permute(1, 2, 0)  # (N, num_features, T)
        # x = self.cnn(x)  # Apply CNN layers
        # x = x.permute(2, 0, 1)  # (T, N, num_channels)

        # # Apply GRU
        # x, _ = self.gru(x)  # Output shape: (T, N, 2 * hidden_size)
        # x = self.layer_norm(x)

        # # Apply linear layer
        # x = self.fc(x)  # Shape: (T, N, num_classes)

        # return nn.functional.log_softmax(x, dim=-1)

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
        # T_diff = inputs.shape[0] - gru_out.shape[0]  # Use GRU output
        # emission_lengths = input_lengths - T_diff


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




### DONT CHANGE ANYTHING BELOW THIS REALLY
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
        kernel_width: int = 9, #small kernel: 3-5, large kernel 7-9
        dropout_rate: float = 0,
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
    def __init__(self, in_channels, out_channels, kernel_size, dropout_rate=0):
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

        from torchmetrics import MetricCollection
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

class TDSGRUCTCModuleWithDropout(pl.LightningModule):
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
        dropout_rate: float = 0,
        gru_hidden: int = 256,
        gru_layers: int = 2,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        # After the MLP, the output shape is (T, N, NUM_BANDS, mlp_features[-1])
        # Flattening gives (T, N, lstm_in_dim) where lstm_in_dim = NUM_BANDS * mlp_features[-1]
        # Flattening gives (T, N, gru_in_dim) where gru_in_dim = NUM_BANDS * mlp_features[-1]
        gru_in_dim = self.NUM_BANDS * mlp_features[-1]
        
        self.model = nn.Sequential(
            SpectrogramNorm(channels=self.NUM_BANDS * self.ELECTRODE_CHANNELS),
            MultiBandRotationInvariantMLP(
                in_features=in_features,
                mlp_features=mlp_features,
                num_bands=self.NUM_BANDS,
            ),
            nn.Flatten(start_dim=2),  # shape: (T, N, lstm_in_dim)
            TDSConvEncoderWithDropout(
                num_features=gru_in_dim,
                block_channels=block_channels,
                kernel_width=kernel_width,
                dropout_rate=dropout_rate
            )
        )
        
        self.gru = nn.GRU(
            input_size=gru_in_dim,
            hidden_size=gru_hidden,
            num_layers=gru_layers,
            dropout=dropout_rate,
            bidirectional=True,  # Keep bidirectional GRU
            batch_first=False  # Keep the shape as (T, N, 2 * gru_hidden)
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(2 * gru_hidden, charset().num_classes),
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
        x, _ = self.gru(x)            # x shape: (T, N, 2 * gru_hidden)
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
