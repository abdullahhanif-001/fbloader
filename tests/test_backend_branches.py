from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

import fbloader
from fbloader.backends import wds as wds_backend
from tests.shardutil import write_test_shard


def test_dali_explicit_without_runtime_raises(tmp_path: Path) -> None:
    shard = write_test_shard(tmp_path / "s0.tar", n=8)
    expected = "Linux" if sys.platform != "linux" else "DALI failed to start"
    with pytest.raises(RuntimeError, match=expected):
        fbloader.DataLoader(
            str(shard),
            batch_size=4,
            num_workers=0,
            backend="dali",
            steps=2,
            crop=8,
        )


def test_streaming_backend_requires_package() -> None:
    with patch("fbloader.backends.streaming.streaming_available", return_value=True):
        with patch("fbloader.backends.streaming.looks_like_mds", return_value=True):
            with patch("fbloader.backends.streaming.build_streaming_dataset") as build:
                build.side_effect = RuntimeError("mosaicml-streaming is not installed")
                with pytest.raises(RuntimeError, match="mosaicml-streaming"):
                    fbloader.DataLoader(
                        "s3://bucket/data.mds",
                        batch_size=4,
                        num_workers=0,
                        backend="streaming",
                        steps=1,
                    )


def test_remote_webdataset_pipeline_branch() -> None:
    fake_pipeline = MagicMock()
    fake_pipeline.shuffle.return_value = fake_pipeline
    fake_pipeline.map.return_value = fake_pipeline
    fake_pipeline.batched.return_value = fake_pipeline
    fake_pipeline.with_epoch.return_value = fake_pipeline
    fake_wds = MagicMock()
    fake_wds.WebDataset.return_value = fake_pipeline
    with patch.object(wds_backend, "expand_urls", return_value=["s3://bucket/shard.tar"]):
        with patch.object(wds_backend, "_all_local", return_value=False):
            with patch.dict("sys.modules", {"webdataset": fake_wds}):
                ds = wds_backend.build_wds_dataset(
                    "s3://bucket/shard.tar",
                    batch_size=2,
                    drop_last=True,
                    shuffle=True,
                    crop=8,
                    seed=1,
                    steps=3,
                    handler="warn",
                )
    assert ds is fake_pipeline
    fake_wds.WebDataset.assert_called_once()
