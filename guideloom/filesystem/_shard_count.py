from collections.abc import Awaitable, Callable


def num_shards_sync(shard_to_reader: Callable[[int], object]) -> int:
    def exists(idx: int) -> bool:
        try:
            shard_to_reader(idx)
        except FileNotFoundError:
            return False
        else:
            return True

    # No shards at all
    if not exists(0):
        return 0

    # Exponential search
    lo = 0
    hi = 1
    while exists(hi):
        lo = hi
        hi *= 2

    # Binary search in (lo, hi)
    while lo + 1 < hi:
        mid = (lo + hi) // 2
        if exists(mid):
            lo = mid
        else:
            hi = mid

    return lo + 1


async def num_shards_async(shard_to_reader: Callable[[int], Awaitable[object]]) -> int:
    async def exists(idx: int) -> bool:
        try:
            await shard_to_reader(idx)
        except FileNotFoundError:
            return False
        else:
            return True

    # No shards at all
    if not await exists(0):
        return 0

    # Exponential search
    lo = 0
    hi = 1
    while await exists(hi):
        lo = hi
        hi *= 2

    # Binary search in (lo, hi)
    while lo + 1 < hi:
        mid = (lo + hi) // 2
        if await exists(mid):
            lo = mid
        else:
            hi = mid

    return lo + 1
