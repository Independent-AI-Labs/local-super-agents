from typing import List, Optional

from pydantic import BaseModel


class IconWindowData(BaseModel):
    screen_dc: int
    mem_dc: int
    offscreen_mem_dc: int
    x: int
    y: int
    gdi_frames: List
    final_width: int
    final_height: int
    draw_outline: bool
    is_animated: bool
    frame_count: int
    frame_delays: List[int]
    run_animation: bool
    opacity: int
    graphics: Optional[any] = None
    outline_pen: Optional[any] = None
    blend: Optional[any] = None
    pt_dest: Optional[any] = None
    sz: Optional[any] = None
    hdc_screens: Optional[List] = []
    gdi_graphics: Optional[List] = None

    class Config:
        arbitrary_types_allowed = True
