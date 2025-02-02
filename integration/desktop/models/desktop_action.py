from __future__ import annotations

from typing import Optional, Tuple, Callable

from pydantic import BaseModel


class DesktopAction(BaseModel):
    uid: Optional[str] = None
    name: Optional[str] = None
    icon_path: Optional[str] = None
    color_rgb: Optional[Tuple[int, int, int]] = (127, 127, 127)
    shortcut_key: Optional[str] = None
    function: Optional[Callable] = None
    callback: Optional[Callable] = None
    args: Optional[...] = None

    class Config:
        arbitrary_types_allowed = True
        from_attributes = True  # Replace this and the 'model_validate()' crap with py-automap?
