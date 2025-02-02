from typing import Optional

from pydantic import BaseModel


class DesktopTarget(BaseModel):
    x: float
    y: float
    description: str = "destination"
    icon_path: Optional[str] = None
