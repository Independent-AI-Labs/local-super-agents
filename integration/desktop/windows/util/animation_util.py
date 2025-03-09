import ctypes
import os
import random
import time
from ctypes import byref
from ctypes import c_void_p
from ctypes.wintypes import RECT
from ctypes.wintypes import SIZE, POINT
from typing import List, Dict

import win32con
import win32gui
from PIL import Image

from compliance.services.logging_service import DEFAULT_LOGGER
from integration.desktop.models.desktop_action import DesktopAction
from integration.desktop.models.icon_window_data import IconWindowData
from integration.desktop.models.text_window_data import TextWindowData
from integration.desktop.windows.util.win_constants import gdiplus, BLENDFUNCTION, AC_SRC_OVER, AC_SRC_ALPHA, ULW_ALPHA, \
    SmoothingModeHighQuality, InterpolationModeHighQualityBicubic, gdi32, user32, UnitPixel, BITMAPINFO, \
    BITMAPINFOHEADER, PixelFormat32bppARGB, SetLayeredWindowAttributes, get_mouse_pos


def await_next_frame_if_needed(start_time, spring_frame_time):
    # end_time = time.time() - start_time
    # remaining_time = end_time - spring_frame_time
    # DEFAULT_LOGGER.log_debug(f"WAITING FOR: {remaining_time:.2f} s")
    #
    # if remaining_time > 0:
    #     time.sleep(spring_frame_time)

    time.sleep(spring_frame_time)


SPRING_CONSTANT = 0.1
SPRING_DAMPING_FACTOR = 0.8
SPRING_FRAME_TIME = 1.0 / 240.0
SPRING_THRESHOLD = 0.1


def spring_animation(
        hwnd: int,
        start_width: int, end_width: int,
        start_height: int, end_height: int,
        start_alpha: int, end_alpha: int,
        start_x: int = None, end_x: int = None,
        start_y: int = None, end_y: int = None
) -> None:
    """
    Animates the window from its initial state to the specified target dimensions, position, and alpha
    using a spring-based approach. Supports smooth resizing, repositioning, and transparency changes.

    Args:
        hwnd (int): Handle to the window to animate.
        start_width (int): Initial width of the window.
        end_width (int): Target width of the window.
        start_height (int): Initial height of the window.
        end_height (int): Target height of the window.
        start_alpha (int): Initial alpha transparency (0-255).
        end_alpha (int): Target alpha transparency (0-255).
        start_x (int, optional): Initial x position of the window.
        end_x (int, optional): Target x position of the window.
        start_y (int, optional): Initial y position of the window.
        end_y (int, optional): Target y position of the window.
    """
    start_time = time.time()

    # Current properties (floats for smoother intermediate updates)
    current_width = float(start_width)
    current_height = float(start_height)
    current_alpha = float(start_alpha)
    current_x = float(start_x) if start_x is not None else None
    current_y = float(start_y) if start_y is not None else None

    # Velocities
    vel_width = vel_height = vel_alpha = vel_x = vel_y = 0.0

    # Max frames to avoid infinite loops
    max_frames = int(3.0 / SPRING_FRAME_TIME)
    frames = 0

    while frames < max_frames:
        frames += 1

        # --- WIDTH ---
        diff_w = end_width - current_width
        vel_width += diff_w * SPRING_CONSTANT
        vel_width *= SPRING_DAMPING_FACTOR
        current_width += vel_width

        if (diff_w > 0 and current_width > end_width) or (diff_w < 0 and current_width < end_width):
            current_width = end_width
            vel_width = 0

        # --- HEIGHT ---
        diff_h = end_height - current_height
        vel_height += diff_h * SPRING_CONSTANT
        vel_height *= SPRING_DAMPING_FACTOR
        current_height += vel_height

        if (diff_h > 0 and current_height > end_height) or (diff_h < 0 and current_height < end_height):
            current_height = end_height
            vel_height = 0

        # --- ALPHA ---
        diff_a = end_alpha - current_alpha
        vel_alpha += diff_a * SPRING_CONSTANT
        vel_alpha *= SPRING_DAMPING_FACTOR
        current_alpha += vel_alpha

        if (diff_a > 0 and current_alpha > end_alpha) or (diff_a < 0 and current_alpha < end_alpha):
            current_alpha = end_alpha
            vel_alpha = 0

        # Clamp alpha
        clamped_alpha = max(0, min(255, int(current_alpha)))

        # --- POSITION X ---
        if start_x is not None and end_x is not None:
            diff_x = end_x - current_x
            vel_x += diff_x * SPRING_CONSTANT
            vel_x *= SPRING_DAMPING_FACTOR
            current_x += vel_x

            if (diff_x > 0 and current_x > end_x) or (diff_x < 0 and current_x < end_x):
                current_x = end_x
                vel_x = 0

        # --- POSITION Y ---
        if start_y is not None and end_y is not None:
            diff_y = end_y - current_y
            vel_y += diff_y * SPRING_CONSTANT
            vel_y *= SPRING_DAMPING_FACTOR
            current_y += vel_y

            if (diff_y > 0 and current_y > end_y) or (diff_y < 0 and current_y < end_y):
                current_y = end_y
                vel_y = 0

        # --- Apply changes to the window ---
        win32gui.SetWindowPos(
            hwnd,
            win32con.HWND_TOPMOST,
            int(current_x) if current_x is not None else win32gui.GetWindowRect(hwnd)[0],
            int(current_y) if current_y is not None else win32gui.GetWindowRect(hwnd)[1],
            int(current_width),
            int(current_height),
            win32con.SWP_NOACTIVATE | win32con.SWP_SHOWWINDOW
        )
        set_multiply_transparency(hwnd, clamped_alpha, 0x000000)

        # --- Check if we can stop early ---
        if (
                abs(diff_w) < SPRING_THRESHOLD and
                abs(diff_h) < SPRING_THRESHOLD and
                abs(diff_a) < SPRING_THRESHOLD and
                (start_x is None or abs(diff_x) < SPRING_THRESHOLD) and
                (start_y is None or abs(diff_y) < SPRING_THRESHOLD)
        ):
            break

        # Ensure ~60fps
        await_next_frame_if_needed(start_time, SPRING_FRAME_TIME)

    set_multiply_transparency(hwnd, end_alpha, 0x000000)


def calculate_next_action_animation_step(actions: List[DesktopAction],
                                         spring_data: Dict,
                                         current_action: DesktopAction,
                                         max_delta: int = 32,
                                         spring_factor: float = 0.24,
                                         friction: float = 0.69,
                                         settling_threshold: float = 0.8):
    """
    Animate the size of each action using a spring effect.

    Args:
        actions:
        spring_data (dict): A dictionary mapping action.uid to [pos_x, pos_y, vel_x, vel_y].
        current_action: The currently active action.
        max_delta (int): The maximum size for the icon.
        spring_factor (float): Spring pull factor.
        friction (float): Damping factor for velocity.
        settling_threshold (float): Threshold for settling.

    Returns:
        dict: Updated spring data.
    """
    for action in actions:
        pos_x, pos_y, vel_x, vel_y = spring_data[action.uid]

        # Determine the target size
        if current_action == action:
            target_x, target_y = max_delta, max_delta
        else:
            target_x, target_y = 0.0, 0.0

        # Calculate the distance to the target
        diff_x = target_x - pos_x
        diff_y = target_y - pos_y

        # Apply spring acceleration
        acc_x = diff_x * spring_factor
        acc_y = diff_y * spring_factor

        # Update velocity with friction (damping) and acceleration
        vel_x = vel_x * friction + acc_x
        vel_y = vel_y * friction + acc_y

        # Update the current position (size) based on velocity
        pos_x += vel_x
        pos_y += vel_y

        # Check for settling (snap to target if within threshold)
        if abs(diff_x) < settling_threshold and abs(vel_x) < settling_threshold:
            pos_x = target_x
            vel_x = 0.0

        if abs(diff_y) < settling_threshold and abs(vel_y) < settling_threshold:
            pos_y = target_y
            vel_y = 0.0

        # Store updated values
        spring_data[action.uid] = [pos_x, pos_y, vel_x, vel_y]

    return spring_data


def set_multiply_transparency(hwnd, opacity: int, color_key):
    # """
    # Apply a transparency effect with a multiply blend-like approach.
    # Black (0, 0, 0) becomes transparent, with gradual alpha transparency.
    # """
    # # Constants for transparency options
    # LWA_COLORKEY = 0x1
    # LWA_ALPHA = 0x2
    # WS_EX_LAYERED = 0x80000
    #
    # # Ensure the window has the WS_EX_LAYERED style
    # current_exstyle = ctypes.windll.user32.GetWindowLongW(hwnd, -20)  # GWL_EXSTYLE = -20
    # if not (current_exstyle & WS_EX_LAYERED):
    #     ctypes.windll.user32.SetWindowLongW(hwnd, -20, current_exstyle | WS_EX_LAYERED)
    #
    # # Apply the transparency attributes
    # result = SetLayeredWindowAttributes(hwnd, COLORREF(color_key), BYTE(int(alpha)), DWORD(LWA_COLORKEY | LWA_ALPHA))
    #
    # if not result:
    #     raise ctypes.WinError(ctypes.get_last_error())
    """
    Apply a plain transparency effect to a window.
    The transparency level is controlled by the `opacity` parameter (0-255).
    No blending is applied; the opacity value determines uniform transparency.
    """
    import ctypes

    # Constants for transparency options
    LWA_ALPHA = 0x2
    WS_EX_LAYERED = 0x80000

    # Ensure the window has the WS_EX_LAYERED style
    current_exstyle = ctypes.windll.user32.GetWindowLongW(hwnd, -20)  # GWL_EXSTYLE = -20
    if not (current_exstyle & WS_EX_LAYERED):
        ctypes.windll.user32.SetWindowLongW(hwnd, -20, current_exstyle | WS_EX_LAYERED)

    # Apply the transparency attributes (only alpha, no blending)
    result = SetLayeredWindowAttributes(
        hwnd, 0, ctypes.c_ubyte(int(opacity)), LWA_ALPHA
    )

    if not result:
        raise ctypes.WinError(ctypes.get_last_error())


def load_image_with_alpha_and_gif_check(img_path: str, g_buffers: List):
    """
    Loads the image using Pillow, obtains all frames (RGBA),
    and creates GDI+ bitmaps from those frames.
    Returns (list_of_gdip_bitmaps, is_animated, frame_count, frame_delays)
    """

    if not os.path.isfile(img_path):
        raise FileNotFoundError(f"File not found: {img_path}")

    pil_img = Image.open(img_path)

    frames = []
    frame_delays = []
    idx = 0

    # Attempt to read all frames
    try:
        while True:
            frame_rgba = pil_img.convert("RGBA").copy()
            frames.append(frame_rgba)

            delay_ms = pil_img.info.get("duration", 100)
            frame_delays.append(delay_ms)

            idx += 1
            pil_img.seek(idx)
    except EOFError:
        pass

    frame_count = len(frames)
    is_animated = (frame_count > 1)

    gdi_frames = []

    for i, frame in enumerate(frames):
        width, height = frame.size
        stride = width * 4

        # Make sure the channels are in BGRA order
        raw_data = frame.tobytes("raw", "BGRA")

        # Create a buffer so we keep ownership of the bytes
        cdata = ctypes.create_string_buffer(raw_data)
        g_buffers.append(cdata)

        gdip_bitmap = c_void_p()
        status = gdiplus.GdipCreateBitmapFromScan0(
            width,
            height,
            stride,
            PixelFormat32bppARGB,
            cdata,
            byref(gdip_bitmap)
        )
        if status != 0 or not gdip_bitmap:
            raise OSError(f"Failed to create GDI+ bitmap from frame #{i} (status={status})")

        gdi_frames.append(gdip_bitmap)

    return gdi_frames, is_animated, frame_count, frame_delays


def _create_bitmaps(screen_dc, mem_dc, width, height):
    """
    Creates a memory DC compatible with screen_dc, with a DIB section
    of the specified width and height. Returns (memory_dc, hbitmap, old_bmp).
    Caller is responsible for cleaning up the DC/hbitmap later.
    """
    bmi = BITMAPINFO()
    bmi.bmiHeader.biSize = ctypes.sizeof(BITMAPINFOHEADER)
    bmi.bmiHeader.biWidth = width
    bmi.bmiHeader.biHeight = -height  # top-down
    bmi.bmiHeader.biPlanes = 1
    bmi.bmiHeader.biBitCount = 32
    bmi.bmiHeader.biCompression = win32con.BI_RGB

    bits_pointer = ctypes.c_void_p()
    hbitmap = gdi32.CreateDIBSection(
        screen_dc,
        byref(bmi),
        win32con.DIB_RGB_COLORS,
        byref(bits_pointer),
        None,
        0
    )

    old_bmp = win32gui.SelectObject(mem_dc, hbitmap)

    return hbitmap, old_bmp


def _create_frame(mem_dc, frame_index, window_data: IconWindowData):
    """
    Initializes and caches the bitmap for a frame. This is called lazily.
    Returns the created hbitmap and the old bitmap selected into the memory DC.
    """
    hbitmap, old_bmp = _create_bitmaps(
        window_data.screen_dc,
        mem_dc,
        window_data.final_width,
        window_data.final_height
    )

    # Draw the frame to the newly created hbitmap
    _draw_frame_to_dc(mem_dc, frame_index, window_data)

    return hbitmap, old_bmp


def init_draw_resources(window_data: IconWindowData):
    """
    Initialize and cache resources (graphics objects, pens, etc.)
    that do not need to be recreated each frame.
    """
    # 1. Create a single GDI+ Graphics object for the memory DC and store it.
    if not window_data.graphics:
        graphics = c_void_p()
        status = gdiplus.GdipCreateFromHDC(window_data.offscreen_mem_dc, byref(graphics))
        if status != 0:
            raise OSError(f"GdipCreateFromHDC failed with status {status}")

        window_data.graphics = graphics

        # Set quality settings only once
        gdiplus.GdipSetSmoothingMode(graphics, SmoothingModeHighQuality)
        gdiplus.GdipSetInterpolationMode(graphics, InterpolationModeHighQualityBicubic)

    # 2. Pre-create the pen for the outline if needed and store it.
    if window_data.draw_outline and not window_data.outline_pen:
        outline_color = 0xFF00FF00  # Fully opaque green (ARGB)
        pen = c_void_p()
        status = gdiplus.GdipCreatePen1(
            ctypes.c_uint32(outline_color),
            ctypes.c_float(2.0),  # Thickness: 2px
            UnitPixel,  # Unit
            byref(pen)
        )
        if status != 0:
            raise OSError(f"GdipCreatePen1 failed with status {status}")

        window_data.outline_pen = pen

    # 3. Cache the BLENDFUNCTION and other structs if they don't change.
    if not window_data.blend:
        blend = BLENDFUNCTION()
        blend.BlendOp = AC_SRC_OVER
        blend.BlendFlags = 0
        blend.SourceConstantAlpha = window_data.opacity
        blend.AlphaFormat = AC_SRC_ALPHA
        window_data.blend = blend

    # 4. Cache the POINT and SIZE if fixed.
    if not window_data.pt_dest:
        window_data.pt_dest = POINT(window_data.x, window_data.y)
    if not window_data.sz:
        window_data.sz = SIZE(window_data.final_width, window_data.final_height)

    return _create_bitmaps(window_data.screen_dc, window_data.offscreen_mem_dc, window_data.final_width, window_data.final_height)


def _draw_frame_to_dc(mem_dc, frame_index, window_data: IconWindowData):
    """
    Draws the specified frame into the provided memory DC.
    """
    # Create a temporary GDI+ Graphics object for the memory DC
    graphics = c_void_p()
    status = gdiplus.GdipCreateFromHDC(mem_dc, byref(graphics))
    if status != 0:
        raise OSError(f"GdipCreateFromHDC failed with status {status}")

    try:
        # Optional: set smoothing and interpolation
        # gdiplus.GdipSetSmoothingMode(graphics, SmoothingModeHighQuality)
        # gdiplus.GdipSetInterpolationMode(graphics, InterpolationModeHighQualityBicubic)

        # Clear the graphics to transparent
        gdiplus.GdipGraphicsClear(graphics, 0x00000000)

        # Draw the frame’s GDI+ bitmap into the memory DC
        gdi_frame = window_data.gdi_frames[frame_index]
        status = gdiplus.GdipDrawImageRectI(
            graphics,
            gdi_frame,
            0, 0,
            window_data.final_width,
            window_data.final_height
        )
        if status != 0:
            raise OSError(f"GdipDrawImageRectI failed with status {status}")

        # (Optional) Draw an outline
        if window_data.draw_outline and window_data.outline_pen:
            pen = window_data.outline_pen
            gdiplus.GdipDrawRectangle(
                graphics,
                pen,
                0, 0,
                window_data.final_width - 1,
                window_data.final_height - 1
            )
    finally:
        # Cleanup the temporary graphics
        gdiplus.GdipDeleteGraphics(graphics)


def draw_animation_frame(hwnd, frame_index, window_data: IconWindowData):
    """
    Draws the specified GIF frame using a single shared offscreen_mem_dc.
    Dynamically updates the layered window with each frame.
    """
    blend = window_data.blend
    opacity = window_data.opacity

    # Skip if fully transparent
    if blend.SourceConstantAlpha == 0 and opacity == 0:
        return

    blend.SourceConstantAlpha = opacity

    # Clear the offscreen memory DC to transparent
    fill_dib_with_color(window_data.offscreen_mem_dc, window_data.final_width, window_data.final_height, (0, 0, 0, 0))

    # Draw the frame into the offscreen_mem_dc
    _draw_frame_to_dc(window_data.offscreen_mem_dc, frame_index, window_data)

    # Update the layered window with the new frame
    pt_dest = POINT(window_data.x, window_data.y)
    sz = SIZE(window_data.final_width, window_data.final_height)
    source_pt = POINT(0, 0)

    if not window_data.offscreen_mem_dc:
        raise ValueError("offscreen_mem_dc is not initialized.")

    result = user32.UpdateLayeredWindow(
        hwnd,
        window_data.screen_dc,
        byref(pt_dest),
        byref(sz),
        window_data.offscreen_mem_dc,
        byref(source_pt),
        0,
        byref(blend),
        ULW_ALPHA
    )

    if not result:
        error_code = ctypes.get_last_error()
        raise OSError(f"UpdateLayeredWindow failed with error {error_code}")


def cleanup_draw_resources(window_data: IconWindowData):
    """
    Properly release cached GDI+ resources when you no longer need them
    (e.g., when closing the window or stopping the animation).
    """
    if window_data.outline_pen:
        gdiplus.GdipDeletePen(window_data.outline_pen)
        window_data.outline_pen = None

    if window_data.graphics:
        gdiplus.GdipDeleteGraphics(window_data.graphics)
        window_data.graphics = None

    for key in ["blend", "pt_dest", "sz"]:
        setattr(window_data, key, None)

    cleanup_cached_frames(window_data)


def cleanup_cached_frames(window_data: IconWindowData):
    """
    Properly restore/select out bitmaps, delete DCs,
    and free up GDI+ graphics objects if stored.
    """
    if window_data.gdi_graphics:
        for graphics in window_data.gdi_graphics:
            if graphics:
                gdiplus.GdipDeleteGraphics(graphics)
        window_data.gdi_graphics = []

    if window_data.hdc_screens:
        try:
            for (mem_dc, hbitmap, old_bmp) in window_data.hdc_screens:
                gdi32.SelectObject(mem_dc, old_bmp)
                gdi32.DeleteObject(hbitmap)
                gdi32.DeleteDC(mem_dc)
        except:
            DEFAULT_LOGGER.log_debug("[ERROR] Failed to delete DC/bitmaps.")
        window_data.hdc_screens = []


def init_text_draw_resources(window_data: TextWindowData):
    if not window_data.graphics:
        graphics = ctypes.c_void_p()
        status = gdiplus.GdipCreateFromHDC(window_data.mem_dc, byref(graphics))
        if status != 0:
            raise OSError(f"GdipCreateFromHDC failed with status {status}")

        window_data.graphics = graphics

        gdiplus.GdipSetSmoothingMode(graphics, SmoothingModeHighQuality)
        gdiplus.GdipSetInterpolationMode(graphics, InterpolationModeHighQualityBicubic)

    if not window_data.blend:
        blend = BLENDFUNCTION()
        blend.BlendOp = AC_SRC_OVER
        blend.BlendFlags = 0
        blend.SourceConstantAlpha = window_data.opacity
        blend.AlphaFormat = AC_SRC_ALPHA
        window_data.blend = blend

    return _create_bitmaps(window_data.screen_dc, window_data.offscreen_mem_dc, window_data.final_width, window_data.final_height)


def fill_dib_with_color(mem_dc, width, height, color):
    """
    Fills the DIB section with a specified ARGB color.
    """
    r, g, b, a = color
    brush_color = (a << 24) | (r << 16) | (g << 8) | b

    hBrush = gdi32.CreateSolidBrush(brush_color)
    rect = RECT(0, 0, width, height)
    user32.FillRect(mem_dc, byref(rect), hBrush)
    gdi32.DeleteObject(hBrush)


def draw_text_to_dib(hdc_mem, width, height, text, font_name, font_size, text_color: int, x_offset=0, y_offset=0):
    """
    Draw text into the DIB section. GDI ignores the alpha channel in text_color;
    it only uses the lower 24 bits (BGR). We handle alpha separately via the layered
    window or manual pixel manipulations if needed.

    Adds support for text offset for shadow effects or positioning.
    """
    # Split out R, G, B from ARGB
    alpha = (text_color >> 24) & 0xFF
    red = (text_color >> 16) & 0xFF
    green = (text_color >> 8) & 0xFF
    blue = text_color & 0xFF

    # GDI color is 0x00BBGGRR
    gdi_color = (blue << 16) | (green << 8) | red

    # Create the GDI font
    hFont = gdi32.CreateFontW(
        font_size,  # nHeight
        0,  # nWidth
        0,  # nEscapement
        0,  # nOrientation
        400,  # fnWeight (400 = normal)
        0,  # fdwItalic
        0,  # fdwUnderline
        0,  # fdwStrikeOut
        0,  # fdwCharSet
        0,  # fdwOutputPrecision
        0,  # fdwClipPrecision
        0,  # fdwQuality
        0,  # fdwPitchAndFamily
        font_name
    )
    old_font = gdi32.SelectObject(hdc_mem, hFont)

    # Set text color (ignores alpha, but layered window can still use the DIB's alpha)
    gdi32.SetTextColor(hdc_mem, gdi_color)

    # Transparent background for text
    gdi32.SetBkMode(hdc_mem, 1)  # TRANSPARENT = 1

    # Draw text with offset applied
    rect = RECT(x_offset, y_offset, width + x_offset, height + y_offset)
    user32.DrawTextW(
        hdc_mem,  # handle to DC
        text,  # text (Unicode string)
        -1,  # text length (-1 for auto-len)
        byref(rect),  # drawing rectangle
        0  # text format flags
    )

    # Cleanup
    gdi32.SelectObject(hdc_mem, old_font)
    gdi32.DeleteObject(hFont)


# Main rendering function with monitoring for text changes
def draw_text_line_with_animation(hwnd: int, window_data: TextWindowData):
    """
    Displays the given text word by word, each word fading in with random small delays.
    Dynamically monitors for changes to `window_data.text` and updates text.
    If no changes occur, the text follows the mouse position.
    Dynamically resizes the window to fit the text and adds a shadow effect.
    """
    text = ""
    opacity = 255
    visibility_frame_count = 0
    drawing_done = True

    while True:
        if len(window_data.text_deque) > 0 and drawing_done:
            text = window_data.text_deque.popleft()
            drawing_done = False
            DEFAULT_LOGGER.log_debug(f">>> {text}")

            visibility_frame_count = 0
            opacity = 255
            window_data.blend.SourceConstantAlpha = opacity  # Ensure blend is fully opaque for the new message

            # Initialize a 32-bit DIB section
            # hBitmap, _ = create_dib_section_bitmaps(window_data.hdc_mem, width, height)

            # Extract base RGB color
            r, g, b = window_data.color

            # Keep track of the full text that has already appeared
            accumulated_text = ""

            # Iterate through words and animate
            for word in text.split():
                if accumulated_text:
                    accumulated_text += " " + word
                else:
                    accumulated_text = word

                for alpha in range(0, 256, 16):
                    # 1) Clear the DIB to transparent
                    # fill_dib_with_rounded_rect(window_data.hdc_mem, width, height, (0, 0, 0, 0))
                    # fill_dib_with_color(window_data.mem_dc, window_data.final_width, window_data.final_height, (0, 0, 0, 0))
                    fill_dib_with_color(window_data.offscreen_mem_dc, window_data.final_width, window_data.final_height, (0, 0, 0, 0))

                    # 2) Draw the shadow with a small offset
                    # shadow_offset = 1
                    # draw_text_to_dib(
                    #     window_data.hdc_mem, width, height,
                    #     text=accumulated_text,
                    #     font_name=window_data.font,
                    #     font_size=24,
                    #     text_color=(0 << 24) | (0 << 16) | (0 << 8) | 0,
                    #     x_offset=shadow_offset,
                    #     y_offset=16 + shadow_offset
                    # )

                    # # 3) Draw the current word with partial alpha
                    draw_text_to_dib(
                        window_data.offscreen_mem_dc, window_data.final_width, window_data.final_height,
                        text=accumulated_text,
                        font_name=window_data.font,
                        font_size=window_data.font_size,
                        text_color=((alpha & 0xFF) << 24) | (r << 16) | (g << 8) | b,
                        x_offset=32,
                        y_offset=16
                    )

                    # 4) Update the layered window once for this frame
                    mouse_x, mouse_y = get_mouse_pos()
                    pt_dest = POINT(mouse_x, mouse_y)
                    sz = SIZE(window_data.final_width, window_data.final_height)
                    source_pt = POINT(0, 0)

                    result = user32.UpdateLayeredWindow(
                        hwnd,
                        window_data.screen_dc,
                        byref(pt_dest),
                        byref(sz),
                        window_data.offscreen_mem_dc,
                        byref(source_pt),
                        0,
                        byref(window_data.blend),
                        ULW_ALPHA
                    )

                    if not result:
                        error_code = ctypes.get_last_error()
                        raise OSError(f"UpdateLayeredWindow failed with error {error_code}")
                    time.sleep(0.003)

                time.sleep(random.uniform(0.005, 0.02))
        else:
            if visibility_frame_count < len(text) * 8:
                # Keep opacity at 255 for the first 300 frames
                opacity = 255
            elif opacity > 0:
                # Gradually reduce opacity after 300 frames
                opacity = max(0, opacity - 10)
            elif opacity <= 0:
                drawing_done = True

            # Ensure the opacity stays within the range [0, 255]
            window_data.blend.SourceConstantAlpha = opacity

            # Get the current mouse position
            mouse_x, mouse_y = get_mouse_pos()
            pt_dest = POINT(mouse_x, mouse_y)
            sz = SIZE(window_data.final_width, window_data.final_height)
            source_pt = POINT(0, 0)

            # Update the layered window
            result = user32.UpdateLayeredWindow(
                hwnd,
                window_data.screen_dc,
                byref(pt_dest),
                byref(sz),
                window_data.offscreen_mem_dc,
                byref(source_pt),
                0,
                byref(window_data.blend),
                ULW_ALPHA
            )

            if not result:
                error_code = ctypes.get_last_error()
                raise OSError(f"UpdateLayeredWindow failed with error {error_code}")

            # Increment the frame counter
            visibility_frame_count += 1

            # Sleep for a short duration to simulate frame delay (e.g., 16ms for ~60FPS)
            time.sleep(0.016)

        win32gui.PumpWaitingMessages()
        time.sleep(0.01)


def cleanup_text_draw_resources(window_data: TextWindowData):
    if window_data.graphics:
        gdiplus.GdipDeleteGraphics(window_data.graphics)
        window_data.graphics = None

    if window_data.blend:
        window_data.blend = None


def disable_color_keying(hwnd):
    """
    Disable any color keying on the layered window to ensure normal transparency behavior.
    """
    import ctypes

    # Constants
    WS_EX_LAYERED = 0x80000
    LWA_ALPHA = 0x2

    # Ensure the window has the WS_EX_LAYERED style
    current_exstyle = ctypes.windll.user32.GetWindowLongW(hwnd, -20)  # GWL_EXSTYLE = -20
    if not (current_exstyle & WS_EX_LAYERED):
        ctypes.windll.user32.SetWindowLongW(hwnd, -20, current_exstyle | WS_EX_LAYERED)

    # Reset Layered Window Attributes with alpha only, ignoring color key
    result = ctypes.windll.user32.SetLayeredWindowAttributes(hwnd, 0, 255, LWA_ALPHA)
    if not result:
        raise ctypes.WinError(ctypes.get_last_error())
