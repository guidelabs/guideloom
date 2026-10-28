from collections.abc import Callable, Iterable, Mapping
from dataclasses import fields, is_dataclass, replace
from typing import Any, cast

import torch


def _move_data_to_device[T: object](data: T, device: torch.device, *, non_blocking: bool = False) -> T:
    match data:
        case torch.Tensor():
            return data.to(device, non_blocking=non_blocking)  # pyright: ignore[reportReturnType]
        case _ if is_dataclass(data) and not isinstance(data, type):
            return replace(
                data,
                **{
                    item.name: _move_data_to_device(
                        getattr(data, item.name), device, non_blocking=non_blocking
                    )
                    for item in fields(data)
                    if item.init
                },
            )
        case Mapping():
            constructor = cast("Callable[[Iterable[tuple[Any, Any]]], T]", type(data))
            return constructor(
                (key, _move_data_to_device(value, device, non_blocking=non_blocking))
                for key, value in data.items()
            )
        case tuple() if hasattr(data, "_fields"):
            # A namedtuple takes its fields as positional arguments, not as one iterable.
            return type(data)(
                *(_move_data_to_device(value, device, non_blocking=non_blocking) for value in data)
            )
        case list() | tuple():
            return type(data)(
                _move_data_to_device(value, device, non_blocking=non_blocking) for value in data
            )
        case _:
            return data
