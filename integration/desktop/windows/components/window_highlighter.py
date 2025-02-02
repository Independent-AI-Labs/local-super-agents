import time
import threading
from typing import Dict, Tuple

import win32gui

from integration.desktop.models.desktop_action import DesktopAction

# -------------------------------------------------------------------
# GLOBALS
# -------------------------------------------------------------------

# Store HWND -> DesktopAction
# The DesktopAction contains the color and any other attributes needed
selected_hwnds: Dict[int, DesktopAction] = {}

# Control flag for stopping the fade loop
_stop_fade_loop = False


# -------------------------------------------------------------------
# DRAWING LOGIC
# -------------------------------------------------------------------

def draw_fading_border(hwnd: int, alpha: float, color: Tuple[int, int, int]):
    """
    Draw a border around the given HWND using the specified color and alpha.
    This is where you'd do your minimal GDI or layered-window draw calls.

    For example (pseudocode):
       - If you have a transparent overlay window for each HWND, update
         the overlay’s alpha via UpdateLayeredWindow(...)
       - Or paint the border directly using GDI calls with an alpha pen/brush
         (requires some extended approach).
    """
    # TODO: implement your actual drawing logic here
    # For demonstration, we’ll just print:
    print(f"Drawing border on hwnd={hwnd} color={color}, alpha={alpha:.2f}")


def fade_border_loop(sleep_interval: float = 0.05, fade_step: float = 0.05):
    """
    Continuously fade between 0 and 1 alpha for all valid selected_hwnds.
    Minimizes draw calls by iterating once per loop.
    """
    global _stop_fade_loop

    alpha = 0.0
    direction = fade_step  # either +fade_step or -fade_step

    while not _stop_fade_loop:
        # Update the alpha (bounce at 0 or 1)
        alpha += direction
        if alpha > 1.0:
            alpha = 1.0
            direction = -direction
        elif alpha < 0.0:
            alpha = 0.0
            direction = -direction

        # Collect invalid hwnds to remove
        invalid_hwnds = []
        for hwnd, action in selected_hwnds.items():
            if not win32gui.IsWindow(hwnd):
                invalid_hwnds.append(hwnd)
            else:
                # Draw the border with current alpha
                color = action.color_rgb or (127, 127, 127)
                draw_fading_border(hwnd, alpha, color)

        # Remove invalid hwnds
        for hwnd in invalid_hwnds:
            del selected_hwnds[hwnd]

        # Small sleep so we don’t thrash the CPU
        time.sleep(sleep_interval)


# -------------------------------------------------------------------
# THREADING WRAPPER
# -------------------------------------------------------------------

def start_fade_thread() -> threading.Thread:
    """
    Start the fade loop in a background thread and return the thread object.
    """
    fade_thread = threading.Thread(target=fade_border_loop, daemon=True)
    fade_thread.start()
    return fade_thread


def stop_fade_loop():
    """
    Signal the fade loop to stop.
    """
    global _stop_fade_loop
    _stop_fade_loop = True
