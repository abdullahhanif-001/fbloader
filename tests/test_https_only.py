from __future__ import annotations

import pytest

from fbloader.backends import wds as wds_backend


def test_cleartext_http_rejected() -> None:
    with pytest.raises(ValueError, match="Unsupported or cleartext"):
        wds_backend.is_remote_url("http://example.com/shard.tar")


def test_https_still_remote() -> None:
    assert wds_backend.is_remote_url("https://example.com/shard.tar") is True
