import ctypes
import os
import time
from ctypes.wintypes import RECT

import win32api
import win32gui
import win32ui
import win32con
import win32process
import win32security

HIGHLIGHT_COLOR_RGB = (77, 187, 235)


def get_dpi_scaling(hwnd):
    """Get the DPI scaling factor for the given window."""
    dpi = ctypes.windll.user32.GetDpiForWindow(hwnd)
    return dpi / 96.0  # Default DPI is 96


def capture_window_screenshot(hwnd):
    """Capture a screenshot of the given window using GDI and save it."""
    try:
        # Get the window's device context (DC)
        left, top, right, bottom = win32gui.GetWindowRect(hwnd)
        width, height = right - left, bottom - top

        # Adjust dimensions for DPI scaling
        scaling_factor = get_dpi_scaling(hwnd)
        width = int(width * scaling_factor)
        height = int(height * scaling_factor)

        hwnd_dc = win32gui.GetWindowDC(hwnd)
        mfc_dc = win32ui.CreateDCFromHandle(hwnd_dc)
        save_dc = mfc_dc.CreateCompatibleDC()

        # Create a bitmap object
        bitmap = win32ui.CreateBitmap()
        bitmap.CreateCompatibleBitmap(mfc_dc, width, height)
        save_dc.SelectObject(bitmap)

        # Copy the window's content into the bitmap
        save_dc.BitBlt((0, 0), (width, height), mfc_dc, (0, 0), win32con.SRCCOPY)

        # Save the bitmap to a file
        timestamp = int(time.time())
        bitmap.SaveBitmapFile(save_dc, f"window_screenshot_{timestamp}.bmp")
        print(f"Screenshot saved as window_screenshot_{timestamp}.bmp")
    except Exception as e:
        print(f"Error capturing screenshot: {e}")
    finally:
        # Cleanup GDI objects
        try:
            if save_dc:
                save_dc.DeleteDC()
            if mfc_dc:
                mfc_dc.DeleteDC()
            if hwnd_dc:
                win32gui.ReleaseDC(hwnd, hwnd_dc)
            if bitmap:
                win32gui.DeleteObject(bitmap.GetHandle())  # Correct cleanup for bitmap
        except Exception as cleanup_error:
            print(f"Error during cleanup: {cleanup_error}")


def clear_and_destroy_highlight_window(highlight_hwnd):
    """Clear graphical buffers and destroy the highlight window."""
    if win32gui.IsWindow(highlight_hwnd):
        # Hide the window first
        win32gui.ShowWindow(highlight_hwnd, win32con.SW_HIDE)

        # Release any associated device context or resources
        hdc = win32gui.GetDC(highlight_hwnd)
        win32gui.PatBlt(hdc, 0, 0, *win32gui.GetClientRect(highlight_hwnd)[2:], win32con.BLACKNESS)
        win32gui.ReleaseDC(highlight_hwnd, hdc)

        # Destroy the window to ensure no leftover artifacts
        win32gui.DestroyWindow(highlight_hwnd)


def create_highlight_window():
    """Create a transparent layered window to act as a highlight."""
    hinst = win32api.GetModuleHandle(None)
    wc = win32gui.WNDCLASS()
    wc.hInstance = hinst
    wc.lpszClassName = "HighlightWindow"
    wc.lpfnWndProc = win32gui.DefWindowProc

    # Register the class (ignore error if already registered)
    try:
        win32gui.RegisterClass(wc)
    except Exception:
        pass

    ex_style = (
            win32con.WS_EX_LAYERED
            | win32con.WS_EX_TRANSPARENT
            | win32con.WS_EX_TOPMOST
            | win32con.WS_EX_TOOLWINDOW  # Exclude from alt-tab
    )
    style = win32con.WS_POPUP

    hwnd = win32gui.CreateWindowEx(
        ex_style,
        wc.lpszClassName,
        None,
        style,
        0,
        0,
        0,
        0,
        0,
        0,
        hinst,
        None,
    )

    # Set the transparency of the window (fully opaque highlight, no color key)
    win32gui.SetLayeredWindowAttributes(hwnd, 0, 255, win32con.LWA_ALPHA)
    # Initially hidden
    win32gui.ShowWindow(hwnd, win32con.SW_HIDE)

    return hwnd


def update_highlight_window(highlight_hwnd, target_hwnd, color=HIGHLIGHT_COLOR_RGB):
    """Update the highlight window to cover the target window."""
    # If no target or not a real window, hide the highlight
    if not target_hwnd or not win32gui.IsWindow(target_hwnd):
        win32gui.ShowWindow(highlight_hwnd, win32con.SW_HIDE)
        return

    rect = RECT()
    # Retrieve the screen coordinates of the target window
    if not ctypes.windll.user32.GetWindowRect(target_hwnd, ctypes.byref(rect)):
        win32gui.ShowWindow(highlight_hwnd, win32con.SW_HIDE)
        return

    width = rect.right - rect.left
    height = rect.bottom - rect.top

    # If invalid size, hide the highlight
    if width <= 0 or height <= 0:
        win32gui.ShowWindow(highlight_hwnd, win32con.SW_HIDE)
        return

    # Move/resize the highlight window
    # We use HWND_TOPMOST again to ensure it stays on top of other windows,
    # but we also set WS_EX_TRANSPARENT so it won't steal focus or clicks.
    win32gui.SetWindowPos(
        highlight_hwnd,
        win32con.HWND_TOPMOST,
        rect.left,
        rect.top,
        width,
        height,
        win32con.SWP_NOACTIVATE | win32con.SWP_SHOWWINDOW,
    )

    # Draw a visible rounded border
    try:
        draw_rounded_border(highlight_hwnd, color)
    except Exception as e:
        print(f"Error drawing border: {e}")


def draw_rounded_border(hwnd, color=(77, 187, 235), thickness=2):
    """Draw a rounded border using FrameRgn for precise control."""
    hdc = win32gui.GetDC(hwnd)
    # Get the client rect of our highlight window
    rect = win32gui.GetClientRect(hwnd)
    left, top, right, bottom = rect

    # Create a region with rounded corners
    # (20, 20) is the radius for the corners; adjust as desired
    hrgn = win32gui.CreateRoundRectRgn(left + 6, top, right - 6, bottom, 16, 16)

    # Create a pen for the border
    pen = win32gui.CreatePen(win32con.PS_SOLID, thickness, win32api.RGB(*color))
    old_pen = win32gui.SelectObject(hdc, pen)

    # Draw the region frame
    win32gui.FrameRgn(hdc, hrgn, pen, thickness, thickness)

    # Clean up GDI objects
    win32gui.SelectObject(hdc, old_pen)
    win32gui.DeleteObject(hrgn)
    win32gui.DeleteObject(pen)
    win32gui.ReleaseDC(hwnd, hdc)


def is_process_elevated(process_id):
    """
    Check if the process with the given process ID is running with elevated privileges (admin).
    Returns True if elevated, False otherwise.
    """
    try:
        # Attempt to open the process
        process_handle = win32api.OpenProcess(
            win32con.PROCESS_QUERY_INFORMATION, False, process_id
        )

        # Open the process token
        token_handle = win32security.OpenProcessToken(
            process_handle, win32security.TOKEN_QUERY
        )

        # Get elevation level
        elevation = win32security.GetTokenInformation(
            token_handle, win32security.TokenElevation
        )

        # Clean up handles
        win32api.CloseHandle(process_handle)
        win32api.CloseHandle(token_handle)

        # Check if elevation is non-zero (indicating admin privileges)
        return elevation != 0

    except win32api.error as e:
        if e.winerror == 5:  # Access is denied
            # If access is denied, assume the process is elevated
            return True
        else:
            # Log unexpected errors
            print(f"Unexpected error checking process elevation: {e}")
            return False
    except Exception as e:
        print(f"Error checking process elevation: {e}")
        return False


def is_valid_target_window(target_hwnd, highlight_hwnd):
    """
    Determine if the target window is valid for highlighting:
    - Must be a real, visible window
    - Not minimized or maximized
    - Not in the same process (avoid highlighting ourselves)
    - Not the highlight window itself
    - Not a window belonging to a process with elevated privileges (admin)
    """
    if not win32gui.IsWindow(target_hwnd):
        return False

    if target_hwnd == highlight_hwnd:
        return False

    # Exclude windows from the same process
    _, target_process_id = win32process.GetWindowThreadProcessId(target_hwnd)
    if target_process_id == os.getpid():
        return False

    # Exclude invisible or minimized windows
    if not win32gui.IsWindowVisible(target_hwnd) or win32gui.IsIconic(target_hwnd):
        return False

    # Check window placement
    placement = win32gui.GetWindowPlacement(target_hwnd)
    show_cmd = placement[1]
    # 2 == SW_MINIMIZE, 3 == SW_MAXIMIZE, etc.
    if show_cmd in [win32con.SW_SHOWMINIMIZED, win32con.SW_MAXIMIZE]:
        return False

    # Exclude windows belonging to processes running as admin
    if is_process_elevated(target_process_id):
        return False

    return True


def get_topmost_window_at_cursor():
    """Get the root-level topmost window under the cursor."""
    pt = win32api.GetCursorPos()
    hwnd = win32gui.WindowFromPoint(pt)
    if hwnd:
        # GetAncestor with GA_ROOT gets the top-level (root) window in that hierarchy
        hwnd = win32gui.GetAncestor(hwnd, win32con.GA_ROOT)
    return hwnd


def select_target(hwnd, highlight_hwnd, callback):
    """
    Takes the HWND of the clicked window, stops the highlighting, and destroys
    all associated resources.
    """
    print(f"Selected target window: {hwnd}")

    if callback is not None:
        callback()

    # (Optional) If you want to capture a screenshot here, do it before destruction:
    # capture_screenshot_of_window(hwnd)

    # Destroy the highlight window
    clear_and_destroy_highlight_window(highlight_hwnd)
    print("Highlighting stopped and resources destroyed.")


def highlight_hovered_window(color=HIGHLIGHT_COLOR_RGB, stop_event=None, callback=None):
    """
    Continuously highlight the topmost valid window under the mouse cursor.
    Capture a screenshot (or perform another action) when the window is clicked.
    Press Ctrl+C in the console to stop.
    """
    highlight_hwnd = create_highlight_window()
    prev_hwnd = None

    try:
        while True:
            try:
                # If an external thread signals stop, exit the loop.
                if stop_event and stop_event.is_set():
                    print("Stop event set. Exiting highlight loop.")
                    break

                if win32api.GetAsyncKeyState(win32con.VK_ESCAPE) & 0x8000:
                    print("Esc key pressed. Exiting highlight loop.")
                    break

                # Get the topmost window under the cursor
                hwnd = get_topmost_window_at_cursor()

                # Update highlight window only if the hovered window changes
                if hwnd and hwnd != prev_hwnd:
                    if is_valid_target_window(hwnd, highlight_hwnd):
                        prev_hwnd = hwnd
                        update_highlight_window(highlight_hwnd, hwnd, color)
                    else:
                        # Hide highlight if the new window is invalid
                        prev_hwnd = None
                        win32gui.ShowWindow(highlight_hwnd, win32con.SW_HIDE)

                # Detect left mouse button click
                if win32api.GetAsyncKeyState(win32con.VK_LBUTTON) & 0x8000:
                    if prev_hwnd:
                        time.sleep(0.1)  # Allow click to settle/focus
                        if win32gui.IsWindow(prev_hwnd):
                            # We have a valid, clicked window
                            select_target(prev_hwnd, highlight_hwnd, callback)
                            break  # Stop the highlight loop
                        else:
                            prev_hwnd = None

                time.sleep(0.05)  # Slight delay to reduce rapid updates
            except Exception as loop_err:
                print(f"Error in highlight loop: {loop_err}")
                prev_hwnd = None
                win32gui.ShowWindow(highlight_hwnd, win32con.SW_HIDE)
    except KeyboardInterrupt:
        print("Exiting via Ctrl+C...")
        # Clean up if user manually interrupts
        clear_and_destroy_highlight_window(highlight_hwnd)
        return False

    # If we broke out of the loop by selecting a target,
    # there's no more cleanup needed here because select_target()
    # already handled it.
    return True


if __name__ == "__main__":
    highlight_hovered_window()
