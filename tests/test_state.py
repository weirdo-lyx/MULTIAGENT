"""M3 非文本状态传递测试。"""
from __future__ import annotations

from multiprocessing import shared_memory

import numpy as np
import pytest

from src.eval.metrics import get_metrics
from src.state import Embedder, StateChannel


def test_embedder_mock_is_offline_deterministic_and_512_dim() -> None:
    """确定性兜底嵌入不依赖网络，且满足 M3 的 512 维接口契约。"""
    embedder = Embedder(model_name="mock")

    first = embedder.encode(["多智能体通信", "共享内存状态传递"])
    second = embedder.encode(["多智能体通信", "共享内存状态传递"])

    assert first.shape == (2, 512)
    assert first.dtype == np.float32
    np.testing.assert_allclose(first, second)


def test_state_channel_put_get_reads_same_array_and_records_metrics() -> None:
    """put 后另一处 get 能读回同一数组，并记录共享内存传输字节数。"""
    metrics = get_metrics()
    metrics.reset()
    channel = StateChannel(metrics=metrics, prefix="test_state")
    arr = np.arange(12, dtype=np.float32).reshape(3, 4)

    handle = channel.put(arr, meta={"source": "unit-test"})
    view = channel.get(handle)

    try:
        assert handle["shape"] == (3, 4)
        assert handle["dtype"] == "float32"
        assert handle["meta"] == {"source": "unit-test"}
        np.testing.assert_array_equal(view, arr)

        # view 是共享内存零拷贝视图，修改共享内存后再次 attach 能看到同一份数据。
        view[0, 0] = 99.0
        again = channel.get(handle)
        try:
            assert again[0, 0] == pytest.approx(99.0)
        finally:
            del again

        summary = metrics.summary()
        assert summary["state_transfers"]["count"] == 1
        assert summary["state_transfers"]["bytes"] == arr.nbytes
    finally:
        del view
        channel.release(handle)


def test_state_channel_release_unlinks_shared_memory() -> None:
    """release 后共享内存名不可再 attach，证明没有系统级残留。"""
    channel = StateChannel(prefix="test_state")
    handle = channel.put(np.ones((2, 2), dtype=np.float32), meta={})
    shm_name = handle["shm_name"]

    channel.release(handle)

    with pytest.raises(FileNotFoundError):
        shared_memory.SharedMemory(name=shm_name, create=False)
