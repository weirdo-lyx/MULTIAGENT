"""M3 共享内存状态通道：用句柄传递大数组，消息里不塞向量正文。"""
from __future__ import annotations

import uuid
import weakref
from multiprocessing import shared_memory
from typing import Any

import numpy as np

from src.eval.metrics import Metrics, get_metrics


class _SharedNDArray(np.ndarray):
    """带共享内存引用的 ndarray 子类，防止底层 buffer 被提前回收。"""

    _shm: shared_memory.SharedMemory | None

    def __new__(
        cls,
        shape: tuple[int, ...],
        dtype: np.dtype,
        shm: shared_memory.SharedMemory,
    ) -> "_SharedNDArray":
        obj = np.ndarray.__new__(cls, shape, dtype=dtype, buffer=shm.buf)
        obj._shm = shm
        return obj

    def __array_finalize__(self, obj: Any) -> None:
        self._shm = getattr(obj, "_shm", None)


def _close_shm(shm: shared_memory.SharedMemory) -> None:
    """尽力关闭共享内存句柄；数组视图仍存活时由后续 GC 再释放。"""
    try:
        shm.close()
    except BufferError:
        pass
    except FileNotFoundError:
        pass


class StateChannel:
    """共享内存状态通道。

    ``put`` 将 numpy 数组复制到共享内存并返回小句柄；``get`` 根据句柄 attach
    并返回零拷贝 ndarray 视图；``release`` 负责 unlink 共享内存，避免多轮实验泄漏。
    """

    def __init__(self, metrics: Metrics | None = None, prefix: str = "ma_state") -> None:
        self.metrics = metrics or get_metrics()
        self.prefix = prefix
        self._segments: dict[str, list[shared_memory.SharedMemory]] = {}

    def put(self, arr: np.ndarray, meta: dict[str, Any] | None = None) -> dict[str, Any]:
        """写入数组到共享内存，返回可放入 Message.state_handle 的句柄。

        Args:
            arr: 待共享的 numpy 数组。非连续数组会先转为 C 连续布局。
            meta: 业务元数据，会原样放进句柄，供接收方理解状态来源。

        Returns:
            ``{"shm_name": str, "shape": tuple, "dtype": str, "meta": dict}``
        """
        if not isinstance(arr, np.ndarray):
            raise TypeError("StateChannel.put 只接受 np.ndarray")

        contiguous = np.ascontiguousarray(arr)
        shm_name = f"{self.prefix}_{uuid.uuid4().hex}"
        shm = shared_memory.SharedMemory(
            name=shm_name,
            create=True,
            size=contiguous.nbytes,
        )
        target = np.ndarray(contiguous.shape, dtype=contiguous.dtype, buffer=shm.buf)
        target[...] = contiguous

        self._segments.setdefault(shm_name, []).append(shm)
        self.metrics.record_state_transfer(int(contiguous.nbytes))

        return {
            "shm_name": shm_name,
            "shape": tuple(int(x) for x in contiguous.shape),
            "dtype": str(contiguous.dtype),
            "meta": dict(meta or {}),
            "nbytes": int(contiguous.nbytes),
        }

    def get(self, handle: dict[str, Any]) -> np.ndarray:
        """根据句柄 attach 共享内存并返回零拷贝数组视图。"""
        shm_name = self._require_name(handle)
        shape = tuple(int(x) for x in handle["shape"])
        dtype = np.dtype(handle["dtype"])

        shm = shared_memory.SharedMemory(name=shm_name, create=False)
        view = _SharedNDArray(shape, dtype, shm)
        self._segments.setdefault(shm_name, []).append(shm)

        # 返回值是共享内存视图；finalize 确保调用方丢弃数组后底层 fd 能关闭。
        weakref.finalize(view, _close_shm, shm)
        return view

    def release(self, handle: dict[str, Any]) -> None:
        """释放句柄对应共享内存。

        ``unlink`` 会先执行，确保名字从系统共享内存表移除；若仍有 ndarray 视图活着，
        close 可能被延后到视图 GC，但不会再留下可被新进程 attach 的共享内存对象。
        """
        shm_name = self._require_name(handle)
        segments = self._segments.pop(shm_name, [])

        unlinked = False
        for shm in segments:
            if not unlinked:
                try:
                    shm.unlink()
                except FileNotFoundError:
                    pass
                unlinked = True
            _close_shm(shm)

        if not segments:
            # 支持由另一个 StateChannel 实例释放只拿到 handle 的共享内存。
            try:
                shm = shared_memory.SharedMemory(name=shm_name, create=False)
            except FileNotFoundError:
                return
            try:
                shm.unlink()
            except FileNotFoundError:
                pass
            _close_shm(shm)

    @staticmethod
    def _require_name(handle: dict[str, Any]) -> str:
        """校验并取出共享内存名，给错误调用提供清晰失败信息。"""
        try:
            shm_name = handle["shm_name"]
        except KeyError as exc:
            raise KeyError("state handle 缺少 shm_name") from exc
        if not isinstance(shm_name, str) or not shm_name:
            raise ValueError("state handle 的 shm_name 必须是非空字符串")
        return shm_name
