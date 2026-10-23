# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# Adapted from TorchData and PyTorch under their BSD-style licenses.
# See licenses/torchdata.txt and licenses/pytorch.txt in the source distribution
# (or the installed distribution's .dist-info/licenses directory).
# Sources: https://github.com/pytorch/data/blob/v0.11.0/torchdata/nodes/pin_memory.py
#          https://github.com/pytorch/pytorch/blob/v2.9.1/torch/utils/data/_utils/pin_memory.py

#
# NOTE: This is from the torchdata PinMemory implementation, with some fixes.
#
# The torchdata implementation of PinMemory is kinda bugged. It doesn't actaully set the cuda.device_index
# because they are missing an `if device is None` statement. That statement is does exist in the offical
# torch.utils.data._utils.pin_memory import _pin_memory_loop. However, it's mising from torchdata.
#
# This fixes this, but just always using `torch.accelerator.set_device_index`
# which will be the behavior in the next version of torch.
# See: https://github.com/pytorch/pytorch/pull/146821/files
#

from typing import Any
import queue
import functools
import threading

from torch.utils.data._utils.pin_memory import pin_memory
from torchdata.nodes.base_node import BaseNode, T
from torchdata.nodes.exception_wrapper import ExceptionWrapper, StartupExceptionWrapper
from torchdata.nodes.map import _SingleThreadedMapper
from torchdata.nodes.snapshot_store import MonotonicIndex, SnapshotStore
import torch
import torch.multiprocessing


def _pin_memory_loop(
    source: BaseNode[object],
    q: queue.Queue[tuple[object, int]],
    snapshot_store: SnapshotStore,
    snapshot_frequency: int,
    semaphore: threading.BoundedSemaphore,
    stop_event: threading.Event,
    device_id: int | str,
) -> None:
    """Pin items from `source` without wrapping them in index tuples.

    Adapted from `torch.utils.data._utils.pin_memory._pin_memory_loop`. Setting the thread's device
    and limiting it to one Torch thread keeps `pin_memory` from using every CPU core.
    """
    idx = MonotonicIndex()

    def _put(
        item: object,
        *,
        block: bool = True,
        snapshot: dict[str, Any] | StartupExceptionWrapper | None = None,
    ) -> None:
        version = idx.get()
        if snapshot:
            snapshot_store.append(snapshot=snapshot, version=version)
        q.put((item, version), block=block)

    try:
        torch.set_num_threads(1)

        torch.multiprocessing._set_thread_name("pt_data_pin")
        torch.accelerator.set_device_index(device_id)

        assert (  # noqa: PT018
            isinstance(snapshot_frequency, int) and snapshot_frequency >= 0
        ), f"snapshot_frequency must be non-negative integer! Got {snapshot_frequency}"
        snapshot_store.append_initial_snapshot(snapshot=source.state_dict())
    except Exception:
        e = StartupExceptionWrapper(where=f"in _pin_memory_loop startup for device {device_id}")
        print(e)
        snapshot_store.append_initial_snapshot(snapshot=e)
        return

    yielded = 0
    while not stop_event.is_set():
        if not semaphore.acquire(blocking=True, timeout=0.1):
            continue
        try:
            item = next(source)
            item = pin_memory(item)
            yielded += 1
            snapshot = None
            if snapshot_frequency > 0 and yielded % snapshot_frequency == 0:
                snapshot = source.state_dict()
            _put(item, block=False, snapshot=snapshot)
        except StopIteration as e:
            item = e
            _put(item, block=False)
            break
        except Exception:
            item = ExceptionWrapper(where=f"in _pin_memory_loop for device {device_id}")
            _put(item, block=False)
            break


class PinMemory(BaseNode[T]):
    """Pin each item from the source node into page-locked host memory.

    The worker pins memory for `torch.accelerator.current_device_index()` captured when this node is
    constructed. Backed by `torch.utils.data._utils.pin_memory.pin_memory`.
    """

    def __init__(
        self,
        source: BaseNode[T],
        snapshot_frequency: int = 1,
    ) -> None:
        """Create a pin-memory node.

        Args:
            source (BaseNode[T]): Node that yields the values to pin.
            snapshot_frequency (int): Number of yielded items per source snapshot. Defaults to 1, so the
                source is snapshotted after every item.
        """
        super().__init__()
        self.source = source
        self.snapshot_frequency = snapshot_frequency
        self._current_device = torch.accelerator.current_device_index()
        self._it: _SingleThreadedMapper[T] | None = None

    def reset(self, initial_state: dict[str, Any] | None = None) -> None:
        super().reset(initial_state)
        if self._it is not None:
            self._it._shutdown()
            del self._it
        self._it = _SingleThreadedMapper(
            source=self.source,
            prefetch_factor=1,
            worker=functools.partial(_pin_memory_loop, device_id=self._current_device),
            snapshot_frequency=self.snapshot_frequency,
            initial_state=initial_state,
        )

    def next(self) -> T:
        if self._it is None:
            raise RuntimeError("PinMemory.reset() was not called")
        return next(self._it)

    def get_state(self) -> dict[str, Any]:
        if self._it is None:
            raise RuntimeError("PinMemory.reset() was not called")
        return self._it.get_state()
