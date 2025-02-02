import ctypes
from collections import deque
from typing import Optional, Deque

from pydantic import BaseModel


class TextWindowData(BaseModel):
    screen_dc: int
    mem_dc: int
    offscreen_mem_dc: int
    x: int
    y: int
    final_width: int
    final_height: int
    font: str
    font_size: int
    color: tuple[int, int, int]
    text_deque: Deque = deque()
    speed: float  # Speed in seconds between words
    opacity: int
    graphics: Optional[ctypes.c_void_p] = None
    blend: Optional[ctypes.c_void_p] = None

    class Config:
        arbitrary_types_allowed = True
