from __future__ import annotations

from collections.abc import Iterable, Iterator

try:
    from tqdm import tqdm as _tqdm
except ImportError:  # pragma: no cover - cluster/runtime fallback
    _tqdm = None


class _PlainProgress:
    def __init__(self, iterable: Iterable, **_: object):
        self._iterable = iterable

    def __iter__(self) -> Iterator:
        return iter(self._iterable)

    def set_postfix(self, **_: object) -> None:
        return

    def close(self) -> None:
        return


def tqdm(iterable: Iterable, **kwargs: object):
    if _tqdm is None:
        return _PlainProgress(iterable, **kwargs)
    return _tqdm(iterable, **kwargs)
