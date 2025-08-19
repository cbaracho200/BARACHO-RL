from __future__ import annotations
from typing import List
from .base import Transition

class ReplayBuffer:
    def __init__(self, capacity: int = 100_000):
        self.capacity = capacity
        self.storage: List[Transition] = []
        self.ptr = 0
    def add(self, tr: Transition):
        if len(self.storage) < self.capacity:
            self.storage.append(tr)
        else:
            self.storage[self.ptr] = tr
            self.ptr = (self.ptr + 1) % self.capacity
    def all(self) -> List[Transition]:
        return list(self.storage)
    def clear(self):
        self.storage.clear()
        self.ptr = 0
