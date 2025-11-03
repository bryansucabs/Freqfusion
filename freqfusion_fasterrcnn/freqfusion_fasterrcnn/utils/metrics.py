"""Utility helpers for logging detection training statistics."""
from __future__ import annotations

import datetime
import time
from collections import defaultdict, deque
from typing import DefaultDict, Dict, Iterable, Iterator, List, Optional

import torch


class SmoothedValue:
    """Track a series of values and provide access to smoothed statistics."""

    def __init__(self, window_size: int = 20, fmt: str | None = None) -> None:
        self.deque: deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt or "{median:.4f} ({global_avg:.4f})"

    def update(self, value: float, n: int = 1) -> None:
        self.deque.append(value)
        self.count += n
        self.total += value * n

    @property
    def median(self) -> float:
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self) -> float:
        d = torch.tensor(list(self.deque), dtype=torch.float32)
        return d.mean().item()

    @property
    def global_avg(self) -> float:
        return self.total / max(1, self.count)

    @property
    def max(self) -> float:
        return max(self.deque)

    @property
    def value(self) -> float:
        return self.deque[-1]

    def __str__(self) -> str:
        return self.fmt.format(
            median=self.median,
            avg=self.avg,
            global_avg=self.global_avg,
            max=self.max,
            value=self.value,
        )


class MetricLogger:
    def __init__(self, delimiter: str = "\t") -> None:
        self.meters: DefaultDict[str, SmoothedValue] = defaultdict(SmoothedValue)
        self.delimiter = delimiter

    def update(self, **kwargs: float) -> None:
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            self.meters[k].update(v)

    def __getattr__(self, attr: str):
        if attr in self.meters:
            return self.meters[attr]
        return super().__getattribute__(attr)

    def __str__(self) -> str:
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append(f"{name}: {meter}")
        return self.delimiter.join(loss_str)

    def log_every(
        self,
        iterable: Iterable,
        print_freq: int,
        header: Optional[str] = None,
    ) -> Iterator:
        i = 0
        if header is None:
            header = ""
        start_time = time.time()
        end = time.time()
        iter_time = SmoothedValue(fmt="{avg:.4f}")
        data_time = SmoothedValue(fmt="{avg:.4f}")
        space_fmt = "{:>6d}/{:d}"
        for obj in iterable:
            data_time.update(time.time() - end)
            yield obj
            iter_time.update(time.time() - end)
            if i % print_freq == 0:
                eta_seconds = iter_time.global_avg * (len(iterable) - i)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                print(
                    f"{header} [{space_fmt.format(i, len(iterable))}] "
                    f"eta: {eta_string} {iter_time.fmt.format(avg=iter_time.avg)} "
                    f"{data_time.fmt.format(avg=data_time.avg)} {self}"
                )
            i += 1
            end = time.time()
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print(f"{header} Total time: {total_time_str} ({total_time/len(iterable):.4f} s/it)")

    def synchronize_between_processes(self) -> None:
        """Placeholder for distributed training compatibility."""

        pass
