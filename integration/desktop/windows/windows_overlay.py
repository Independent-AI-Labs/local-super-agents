import os

import subprocess
import threading
import time
from pathlib import Path

import win32gui
from pynput import keyboard

from integration.data.config import CHROME_PATH, OPEN_WEBUI_BASE_URL, DESKTOP_OVERLAY_TARGET_FRAME_TIME, OPEN_WEBUI_PORT, INSTALL_PATH
from integration.desktop.models.desktop_target import DesktopTarget
from integration.desktop.windows.action_manager import ActionManager
from integration.desktop.windows.components.smart_window import SmartWindow
from integration.desktop.windows.user.actions import ACTION_PAIRS, ACTIONS
from integration.desktop.windows.util.animation_util import cleanup_draw_resources, draw_text_line_with_animation
from integration.desktop.windows.util.gui_util import create_icon_window, create_text_window, sleep_eyes_open, rename_process_window
from integration.desktop.windows.util.shell_util import is_port_open
from integration.desktop.windows.util.win_constants import get_mouse_pos

ALT_DOWN = False
CTRL_DOWN = False

OPEN_WEBUI_WINDOW: SmartWindow | None = None
ACTION_MANAGER: ActionManager | None = None
TEXT_WINDOW: int | None = None


##############################################################################
# Keyboard Hooks
##############################################################################

# TODO User should be able to remap all keys / controls.
def on_press(key):
    global ALT_DOWN, CTRL_DOWN

    try:
        if key == keyboard.Key.alt_l:
            ALT_DOWN = True
        elif key == keyboard.Key.alt_r:
            CTRL_DOWN = False

        # Only handle shortcuts if ALT is down AND the pressed key has a char
        if ALT_DOWN and hasattr(key, "char"):
            # Toggle Fullscreen Mode
            if ACTION_MANAGER.current_action is None:
                if hasattr(key, 'char') and key.char == '`':
                    print("Toggle Fullscreen Mode (ALT + `) detected!")
                    OPEN_WEBUI_WINDOW.toggle_floating_mode()

            # Toggle Minimal Mode
            if hasattr(key, 'char') and key.char == 'z':
                if OPEN_WEBUI_WINDOW.floating_state:
                    print("Toggle Minimal Mode (ALT + Z) detected!")
                    OPEN_WEBUI_WINDOW.toggle_minimal_state()

            # Toggle Docked Mode
            if hasattr(key, 'char') and key.char == 'd':
                if OPEN_WEBUI_WINDOW.floating_state:
                    print("Toggle Pinned Mode (ALT + D) detected!")
                    OPEN_WEBUI_WINDOW.toggle_pinned_state()

            # Toggle Docked Mode
            if hasattr(key, 'char') and key.char == 'q':
                if OPEN_WEBUI_WINDOW.floating_state:
                    print("Move (ALT + Q) detected!")
                    on_target_selected()

            pressed_char = key.char

            # Check if the pressed_char is in our ACTION_PAIRS
            if pressed_char in ACTION_PAIRS:
                normal_action, opposite_action = ACTION_PAIRS[pressed_char]

                # If ACTION_MANAGER.current_action is None, we default to normal
                if ACTION_MANAGER.current_action is None:
                    ACTION_MANAGER.current_action = normal_action
                    print(f"Activating {ACTION_MANAGER.current_action.name}")
                else:
                    # If we're already using one of these two, toggle to the other
                    if ACTION_MANAGER.current_action == normal_action:
                        ACTION_MANAGER.current_action = opposite_action
                        print(f"Toggling to {ACTION_MANAGER.current_action.name}")

                    elif ACTION_MANAGER.current_action == opposite_action:
                        ACTION_MANAGER.current_action = normal_action
                        print(f"Toggling to {ACTION_MANAGER.current_action.name}")
                    else:
                        # We have some *other* action active,
                        # so let's switch to the normal action
                        ACTION_MANAGER.current_action = normal_action
                        print(f"Switching from another action to {ACTION_MANAGER.current_action.name}")

                if ACTION_MANAGER.current_action.function:
                    ACTION_MANAGER.current_action.function()

        # If user presses ESC while we have a current action, clear it
        if ACTION_MANAGER.current_action is not None and key == keyboard.Key.esc:
            print(f"Deactivating {ACTION_MANAGER.current_action.name}")
            ACTION_MANAGER.current_action = None

    except AttributeError:
        # Some keys (e.g. arrow keys, function keys) don't have a .char
        pass


def on_release(key):
    global ALT_DOWN, CTRL_DOWN

    if key == keyboard.Key.alt_l:
        ALT_DOWN = False
    elif key == keyboard.Key.alt_r:
        CTRL_DOWN = False


# Example usage:
def on_target_selected():
    ACTION_MANAGER.current_action = None
    mouse_x, mouse_y = get_mouse_pos()
    new_x = mouse_x - 32
    new_y = mouse_y - 32
    new_target = DesktopTarget(
        x=new_x,
        y=new_y,
        description="ActionTarget"
    )

    # TODO Replace with add_target().
    ACTION_MANAGER.action_targets_dequeue.append(new_target)
    OPEN_WEBUI_WINDOW._update_target_window(new_target, 0.025)


##############################################################################
# Main
##############################################################################

def init():
    global OPEN_WEBUI_WINDOW, ACTION_MANAGER, TEXT_WINDOW

    listener = keyboard.Listener(on_press=on_press, on_release=on_release)
    listener.start()

    # 2) ActionManager instance.
    ACTION_MANAGER = ActionManager(
        actions=ACTIONS,
        action_pairs=ACTION_PAIRS,
        on_target_selected_callback=on_target_selected  # Replace with actual callback if needed
    )

    TEXT_WINDOW, window_data = create_text_window("TextBubble", "Hello!", x=2560, y=720)
    ACTION_MANAGER.text_hwnd = TEXT_WINDOW
    ACTION_MANAGER.text_window_data[TEXT_WINDOW] = window_data

    # Start the thread to update the action icon's position
    threading.Thread(target=ACTION_MANAGER.manage_action_windows, daemon=True).start()

    add_message("Starting services...")

    def draw_messages():
        draw_text_line_with_animation(TEXT_WINDOW, window_data)

    threading.Thread(target=draw_messages).start()

    # 3) Ait for Chrome to load and look for the title.
    # Wait for Open WebUI to start
    print("[INFO] Waiting for Open WebUI to start...")

    max_attempts = 180
    for attempt in range(max_attempts):
        if attempt > max_attempts or is_port_open(OPEN_WEBUI_PORT):
            break

        sleep_eyes_open(.5)

    if not is_port_open(OPEN_WEBUI_PORT):
        print("[ERROR] Couldn't wait for Open WebUI to start. UI is not going to be loaded.")
        # TODO
    else:
        add_message("Preparing AI Assistant...")

        STARTF_USESHOWWINDOW = 1
        SW_SHOWMINIMIZED = 2

        startupinfo = subprocess.STARTUPINFO()
        startupinfo.dwFlags |= STARTF_USESHOWWINDOW
        startupinfo.wShowWindow = SW_SHOWMINIMIZED
        # 1) Launch Chromium or Brave in "app mode"

        process = subprocess.Popen(
            [
                CHROME_PATH,
                "--disable-gpu",
                fr'--user-data-dir={INSTALL_PATH}\.tools\chromedata',
                f"--app={OPEN_WEBUI_BASE_URL}",
                "--no-first-run",
                "--no-default-browser-check",
                "--disable-notifications",
                "--disable-infobars",
                "--window-name=OPENWEBUI",
                "--kiosk"
            ],
            startupinfo=startupinfo
        )

        sleep_eyes_open(2)

        rename_process_window(process.pid, "OPENWEBUI")

        OPEN_WEBUI_WINDOW = SmartWindow("OPENWEBUI")
        OPEN_WEBUI_WINDOW.action_manager = ACTION_MANAGER
        print("[INFO] Attached to Open WebUI window!")

        # Create the action icon windows and register them in the manager
        for action in ACTION_MANAGER.actions:
            hwnd, _ = create_icon_window(action.name.replace(" ", ""),
                                         action.icon_path,
                                         final_width=32,
                                         final_height=32,
                                         g_buffers=ACTION_MANAGER.G_BUFFERS)

            ACTION_MANAGER.action_hwnds[action.uid] = hwnd

        move_target_hwnd, window_data = create_icon_window("MoveTarget",
                                                           rf"{INSTALL_PATH}\agents\res\integration\graphics\ui\target.gif",
                                                           final_width=64,
                                                           final_height=64,
                                                           g_buffers=ACTION_MANAGER.G_BUFFERS)

        ACTION_MANAGER.move_target_hwnd = move_target_hwnd
        ACTION_MANAGER.target_window_data[move_target_hwnd] = window_data

        # ACTION_MANAGER.text_window_data[text_hwnd] = window_data

        OPEN_WEBUI_WINDOW.toggle_floating_mode()

        print("[INFO] All processes started. Monitoring...")


def cleanup():
    if ACTION_MANAGER.move_target_hwnd is not None:
        cleanup_draw_resources(ACTION_MANAGER.target_window_data[ACTION_MANAGER.move_target_hwnd])
    # cleanup_text_draw_resources(ACTION_MANAGER.text_hwnd)


def add_message(message: str):
    ACTION_MANAGER.text_window_data[ACTION_MANAGER.text_hwnd].text_deque.append(message)


def main_loop():
    # 1) Initialize GUI components.
    init()

    if not os.path.exists(fr"{Path.home()}\.ollama\models\manifests\registry.ollama.ai\library\deepseek-r1"):
        add_message("Launch the AI Assistant and create your Open WebUI account!")
        add_message("Downloading default model (DeepSeek-R1:8B / 16K context length)...")
        add_message("DeepSeek-R1:8B will become available under 'Models' in a few minutes.")
        add_message("10GB+ of VRAM is recommended. Multi-GPU setups are supported!")
        add_message("Visit ollama.com/search to explore thousands of free models!")

    # add_message("Press ALT + Q to summon the Assistant.")

    # 2) Standard message loop + small idle
    #    We'll pump messages so the parent can process events properly. Ahh, Windows...
    try:
        while True:
            win32gui.PumpWaitingMessages()
            time.sleep(DESKTOP_OVERLAY_TARGET_FRAME_TIME)
    except (KeyboardInterrupt, SystemExit):
        cleanup()
        print("Exiting...")
