import ctypes
import math
import os
import time
from ctypes import byref
from ctypes.wintypes import SIZE, POINT

import win32api
import win32clipboard
import win32con
import win32gui
import win32process

from compliance.services.logging_service import DEFAULT_LOGGER
from integration.desktop.models.icon_window_data import IconWindowData
from integration.desktop.models.text_window_data import TextWindowData
from integration.desktop.windows.util.animation_util import (
    set_multiply_transparency,
    load_image_with_alpha_and_gif_check,
    init_draw_resources,
    init_text_draw_resources,
    draw_animation_frame,
)
from integration.desktop.windows.util.win_constants import (
    gdiplus,
    LASTINPUTINFO,
    user32,
    BLENDFUNCTION,
    get_mouse_pos, EnumWindowsProc,
)


def sleep_eyes_open(period: float) -> None:
    """
    Sleeps for a specified period while manage Windows messages.

    Args:
        period (float): The time in seconds to sleep.
    """
    elapsed = 0.0
    while elapsed < period:
        elapsed += 0.1
        win32gui.PumpWaitingMessages()
        time.sleep(0.1)


def register_window_class(class_name: str) -> int:
    """
    Registers a custom window class for creating windows.

    Args:
        class_name (str): The name of the window class to register.

    Returns:
        int: The atom value returned by RegisterClass.
    """
    hinst = win32api.GetModuleHandle(None)
    wndclass = win32gui.WNDCLASS()
    wndclass.hInstance = hinst
    wndclass.lpszClassName = class_name
    wndclass.lpfnWndProc = wnd_proc  # type: ignore
    wndclass.hCursor = win32gui.LoadCursor(0, win32con.IDC_ARROW)
    return win32gui.RegisterClass(wndclass)


def create_host_window(
        class_name: str,
        window_title: str,
        width: int,
        height: int,
) -> int:
    """
    Creates a borderless and always-topmost window.

    Args:
        class_name (str): The name of the registered window class.
        window_title (str): The title of the window.
        width (int): The initial width of the window.
        height (int): The initial height of the window.

    Returns:
        int: The handle to the created window.
    """
    hinst = win32api.GetModuleHandle(None)
    atom = register_window_class(class_name)

    style = win32con.WS_POPUP | win32con.WS_VISIBLE
    ex_style = (
            win32con.WS_EX_TOPMOST
            | win32con.WS_EX_TOOLWINDOW
            | win32con.WS_EX_LAYERED
    )

    hwnd = win32gui.CreateWindowEx(
        ex_style,
        atom,
        window_title,
        style,
        0,
        0,
        width,
        height,
        0,
        0,
        hinst,
        None,
    )

    win32gui.ShowWindow(hwnd, win32con.SW_SHOWNOACTIVATE)
    win32gui.UpdateWindow(hwnd)
    return hwnd


def resize_child_to_parent(parent_hwnd: int, child_hwnd: int) -> None:
    """
    Resizes a child window to fill the client area of its parent.

    Args:
        parent_hwnd (int): Handle to the parent window.
        child_hwnd (int): Handle to the child window.
    """
    left, top, right, bottom = win32gui.GetClientRect(parent_hwnd)
    width = right - left
    height = bottom - top

    win32gui.SetWindowPos(
        child_hwnd,
        None,
        0,
        0,
        width,
        height,
        win32con.SWP_NOZORDER | win32con.SWP_NOOWNERZORDER,
    )


def find_window_by_title(search_title: str) -> list[int]:
    """
    Finds visible windows by matching part of their title.

    Args:
        search_title (str): The substring to look for in window titles.

    Returns:
        list[int]: List of handles to windows that match the criteria.
    """
    results = []

    def callback(hwnd: int, results: list[int]) -> bool:
        if win32gui.IsWindowVisible(hwnd):
            title = win32gui.GetWindowText(hwnd)
            if search_title in title:
                results.append(hwnd)
        return True

    win32gui.EnumWindows(callback, results)
    return results


def rename_process_window(webui_pid, title):
    """Callback function for EnumWindows to check each window."""

    def foreach_window(hwnd, lParam):
        if win32gui.IsWindowVisible(hwnd):
            _, pid = win32process.GetWindowThreadProcessId(hwnd)
            if pid == webui_pid:
                # Found the window; set its title
                win32gui.SetWindowText(hwnd, title)
                return False  # Stop enumeration
        return True  # Continue enumeration

    win32gui.EnumWindows(EnumWindowsProc(foreach_window), 0)


def wnd_proc(hwnd: int, msg: int, wparam: int, lparam: int) -> int:
    """
    Default window procedure for handling messages.

    Args:
        hwnd (int): Handle to the window.
        msg (int): Message identifier.
        wparam (int): Additional message information.
        lparam (int): Additional message information.

    Returns:
        int: Result of DefWindowProc if message is unhandled.
    """
    if msg == win32con.WM_DESTROY:
        win32gui.PostQuitMessage(0)
    elif msg == win32con.WM_PAINT:
        return 0
    return win32gui.DefWindowProc(hwnd, msg, wparam, lparam)


def create_layered_window(
        class_name: str,
        atom: int,
        x: int,
        y: int,
        width: int,
        height: int,
) -> int:
    """
    Creates a layered window for transparency effects.

    Args:
        class_name (str): The name of the registered window class.
        atom (int): The atom value from RegisterClass.
        x (int): Initial x-coordinate of the window.
        y (int): Initial y-coordinate of the window.
        width (int): Initial width of the window.
        height (int): Initial height of the window.

    Returns:
        int: Handle to the created layered window.
    """
    style = win32con.WS_POPUP | win32con.WS_VISIBLE
    ex_style = (
            win32con.WS_EX_TOPMOST
            | win32con.WS_EX_TOOLWINDOW
            | win32con.WS_EX_LAYERED
    )

    hwnd = win32gui.CreateWindowEx(
        ex_style,
        atom,
        class_name,
        style,
        x,
        y,
        width,
        height,
        0,
        0,
        win32api.GetModuleHandle(None),
        None,
    )
    return hwnd


def initialize_layered_window(
        hwnd: int,
        hdc_mem: int,
        hdc_screen: int,
        width: int,
        height: int,
        transparency_color_key: int = 0,
) -> None:
    """
    Initializes a layered window with transparency properties.

    Args:
        hwnd (int): Handle to the window.
        hdc_mem (int): Memory device context handle.
        hdc_screen (int): Screen device context handle.
        width (int): Width of the window.
        height (int): Height of the window.
        transparency_color_key (int, optional): Color key for transparency. Defaults to 0.
    """
    blend_function = BLENDFUNCTION()
    blend_function.BlendOp = win32con.AC_SRC_OVER
    blend_function.BlendFlags = 0
    blend_function.SourceConstantAlpha = 0  # Fully transparent
    blend_function.AlphaFormat = win32con.AC_SRC_ALPHA

    size = SIZE(width, height)
    point_source = POINT(0, 0)
    point_dest = POINT(0, 0)

    result = user32.UpdateLayeredWindow(
        hwnd,
        hdc_screen,
        byref(point_dest),
        byref(size),
        hdc_mem,
        byref(point_source),
        transparency_color_key,
        byref(blend_function),
        win32con.ULW_ALPHA,
    )

    if not result:
        raise ctypes.WinError(ctypes.get_last_error())


def _create_compatible_dc() -> tuple[int, int]:
    """
    Creates a memory device context compatible with the screen.

    Returns:
        tuple[int, int]: Tuple containing screen DC and memory DC.
    """
    screen_dc = win32gui.GetDC(0)
    mem_dc = win32gui.CreateCompatibleDC(screen_dc)
    return screen_dc, mem_dc


def create_text_window(
        class_name: str,
        text: str,
        font: str = "Montserrat",
        font_size: int = 28,
        color: tuple[int, int, int] = (255, 255, 255),
        x: int = 0,
        y: int = 0,
        final_width: int = 720,
        final_height: int = 100,
        speed: float = 1.0,
        opacity: int = 255,
) -> tuple[int, TextWindowData]:
    """
    Creates a window for displaying animated text.

    Args:
        class_name (str): The name of the registered window class.
        text (str): The text to display.
        font (str, optional): Font family. Defaults to "Montserrat".
        font_size (int, optional): Font size in points. Defaults to 28.
        color (tuple[int, int, int], optional): Text color in RGB. Defaults to (255, 255, 255).
        x (int, optional): Initial x-coordinate. Defaults to 0.
        y (int, optional): Initial y-coordinate. Defaults to 0.
        final_width (int, optional): Final width of the window. Defaults to 720.
        final_height (int, optional): Final height of the window. Defaults to 100.
        speed (float, optional): Animation speed factor. Defaults to 1.0.
        opacity (int, optional): Initial opacity. Defaults to 255.

    Returns:
        tuple[int, TextWindowData]: Tuple containing window handle and text data.
    """
    atom = register_window_class(class_name)
    hwnd = create_layered_window(class_name, atom, x, y, final_width, final_height)

    screen_dc, mem_dc = _create_compatible_dc()
    offscreen_mem_dc = win32gui.CreateCompatibleDC(screen_dc)

    window_data = TextWindowData(
        mem_dc=mem_dc,
        offscreen_mem_dc=offscreen_mem_dc,
        screen_dc=screen_dc,
        x=x,
        y=y,
        final_width=final_width,
        final_height=final_height,
        font=font,
        font_size=font_size,
        color=color,
        text=text,
        speed=speed,
        opacity=opacity,
    )

    init_text_draw_resources(window_data)
    win32gui.ShowWindow(hwnd, win32con.SW_SHOWNOACTIVATE)

    return hwnd, window_data


def create_icon_window(
        class_name: str,
        img_path: str,
        x: int = -1024,
        y: int = -1024,
        draw_outline: bool = False,
        final_width: int | None = None,
        final_height: int | None = None,
        g_buffers: list | None = None,
) -> tuple[int, IconWindowData]:
    """
    Creates a window for displaying icons or images with optional animation.

    Args:
        class_name (str): The name of the registered window class.
        img_path (str): Path to the image file.
        x (int, optional): Initial x-coordinate. Defaults to -1024.
        y (int, optional): Initial y-coordinate. Defaults to -1024.
        draw_outline (bool, optional): Whether to draw an outline around the icon. Defaults to False.
        final_width (int | None, optional): Final width of the window. If None, uses image dimensions. Defaults to None.
        final_height (int | None, optional): Final height of the window. If None, uses image dimensions. Defaults to None.
        g_buffers: List for GDI+ buffers. Not used in this implementation.

    Returns:
        tuple[int, IconWindowData]: Tuple containing window handle and icon data.
    """
    if not os.path.isfile(img_path):
        raise FileNotFoundError(f"File not found: {img_path}")

    atom = register_window_class(class_name)
    gdi_frames, is_gif, frame_count_val, frame_delays_list = load_image_with_alpha_and_gif_check(
        img_path, g_buffers
    )

    native_w = ctypes.c_uint()
    native_h = ctypes.c_uint()
    gdiplus.GdipGetImageWidth(gdi_frames[0], byref(native_w))
    gdiplus.GdipGetImageHeight(gdi_frames[0], byref(native_h))

    if final_width is None:
        final_width = native_w.value
    if final_height is None:
        final_height = native_h.value

    hwnd = create_layered_window(class_name, atom, x, y, final_width, final_height)

    screen_dc, mem_dc = _create_compatible_dc()
    offscreen_mem_dc = win32gui.CreateCompatibleDC(screen_dc)

    is_animated = is_gif and frame_count_val > 1

    window_data = IconWindowData(
        screen_dc=screen_dc,
        offscreen_mem_dc=offscreen_mem_dc,
        mem_dc=mem_dc,
        x=x,
        y=y,
        gdi_frames=gdi_frames,
        final_width=final_width,
        final_height=final_height,
        draw_outline=draw_outline,
        is_animated=is_animated,
        frame_count=frame_count_val if is_animated else 1,
        frame_delays=frame_delays_list if is_animated else [],
        run_animation=is_animated,
        opacity=255,
    )

    init_draw_resources(window_data)
    draw_animation_frame(hwnd, 0, window_data)
    win32gui.ShowWindow(hwnd, win32con.SW_SHOWNOACTIVATE)

    return hwnd, window_data


def calculate_euclidean_distance(
        x1: float, y1: float, x2: float, y2: float
) -> float:
    """
    Calculates the Euclidean distance between two points.

    Args:
        x1 (float): X-coordinate of first point.
        y1 (float): Y-coordinate of first point.
        x2 (float): X-coordinate of second point.
        y2 (float): Y-coordinate of second point.

    Returns:
        float: The Euclidean distance between the two points.
    """
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)


def get_last_input_time() -> float:
    """
    Gets the number of seconds since the last user input.

    Returns:
        float: Time in seconds since the last mouse or keyboard input.
    """
    lii = LASTINPUTINFO()
    lii.cbSize = ctypes.sizeof(LASTINPUTINFO)
    if ctypes.windll.user32.GetLastInputInfo(byref(lii)):
        tick_count = ctypes.windll.kernel32.GetTickCount()
        millis_since_last_input = tick_count - lii.dwTime
        return millis_since_last_input / 1000.0
    else:
        return 0.0


def is_alt_pressed() -> bool:
    """
    Checks if the ALT key is currently pressed.

    Returns:
        bool: True if ALT is pressed, False otherwise.
    """
    VK_ALT = 0x12  # Virtual-Key code for ALT
    return (
            ctypes.windll.user32.GetAsyncKeyState(VK_ALT) & 0x8000 != 0
    )


def is_ctrl_pressed() -> bool:
    """
    Checks if the CTRL key is currently pressed.

    Returns:
        bool: True if CTRL is pressed, False otherwise.
    """
    VK_CTRL = 0x11  # Virtual-Key code for CTRL
    return (
            ctypes.windll.user32.GetAsyncKeyState(VK_CTRL) & 0x8000 != 0
    )


def is_cursor_over_window(hwnd: int) -> bool:
    """
    Checks if the cursor is currently over a window.

    Args:
        hwnd (int): Handle to the window.

    Returns:
        bool: True if cursor is over the window, False otherwise.
    """
    try:
        cursor_x, cursor_y = get_mouse_pos()
        rect = win32gui.GetWindowRect(hwnd)
        window_left, window_top, window_right, window_bottom = rect
        return (
                window_left <= cursor_x <= window_right
                and window_top <= cursor_y <= window_bottom
        )
    except Exception:
        return False


def get_monitor_info_with_retries(
        hwnd: int,
        retries: int = 3,
        delay: float = 0.1,
) -> dict:
    """
    Retrieves monitor information with retry logic.

    Args:
        hwnd (int): Handle to the window.
        retries (int, optional): Number of attempts. Defaults to 3.
        delay (float, optional): Delay between retries in seconds. Defaults to 0.1.

    Returns:
        dict: Monitor information dictionary.

    Raises:
        RuntimeError: If unable to get monitor info after all retries.
    """
    for attempt in range(retries):
        try:
            monitor_handle = win32api.MonitorFromWindow(
                hwnd, win32con.MONITOR_DEFAULTTONEAREST
            )
            if monitor_handle is not None:
                return win32api.GetMonitorInfo(monitor_handle)
        except win32api.error as e:
            DEFAULT_LOGGER.log_debug(f"Attempt {attempt + 1} failed: {e}")
            time.sleep(delay)

    raise RuntimeError(
        f"Unable to get monitor info after {retries} attempts."
    )


def set_window_to_fullscreen(hwnd: int) -> None:
    """
    Sets a window to fullscreen mode.

    Args:
        hwnd (int): Handle to the window.
    """
    monitor_info = get_monitor_info_with_retries(hwnd)
    mon_left, mon_top, mon_right, mon_bottom = monitor_info["Monitor"]
    mon_width = mon_right - mon_left
    mon_height = mon_bottom - mon_top

    win32gui.SetWindowPos(
        hwnd,
        win32con.HWND_TOP,
        0,
        0,
        int(mon_width),
        int(mon_height),
        win32con.SWP_NOACTIVATE | win32con.SWP_SHOWWINDOW,
    )

    set_multiply_transparency(hwnd, 255, 0x000000)


def reparent(hwnd: int, parent: int) -> None:
    """
    Reparents a window under a new parent.

    Args:
        hwnd (int): Handle to the child window.
        parent (int): Handle to the new parent window.
    """
    win32gui.SetParent(hwnd, parent)

    child_style = win32gui.GetWindowLong(hwnd, win32con.GWL_STYLE)
    child_style |= win32con.WS_CHILD
    child_style &= ~(
            win32con.WS_CAPTION | win32con.WS_THICKFRAME | win32con.WS_SYSMENU
    )
    win32gui.SetWindowLong(hwnd, win32con.GWL_STYLE, child_style)

    resize_child_to_parent(parent, hwnd)


def get_clipboard_text() -> str | None:
    """
    Retrieves text from the clipboard.

    Returns:
        str | None: Clipboard text or None if retrieval fails.
    """
    try:
        win32clipboard.OpenClipboard()
        text = win32clipboard.GetClipboardData(win32clipboard.CF_TEXT)
        return text.decode("utf-8")
    except Exception as e:
        DEFAULT_LOGGER.log_debug(f"Error retrieving clipboard text: {e}")
        return None
    finally:
        win32clipboard.CloseClipboard()
