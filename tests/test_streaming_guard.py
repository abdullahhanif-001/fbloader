from __future__ import annotations

import pytest

from fbloader.backends import streaming as st


def test_streaming_availability_flag() -> None:
    assert isinstance(st.streaming_available(), bool)


def test_looks_like_mds_false_for_tar() -> None:
    assert st.looks_like_mds("/data/shards.tar") is False


def test_build_streaming_requires_package() -> None:
    if st.streaming_available():
        pytest.skip("streaming installed")
    with pytest.raises(RuntimeError, match="mosaicml-streaming"):
        st.build_streaming_dataset("s3://bucket/mds", batch_size=4, shuffle=False)
