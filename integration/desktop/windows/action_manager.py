import threading
import time
from collections import deque
from typing import List, Callable, Optional, Dict, Deque

import win32gui

from integration.data.config import DESKTOP_OVERLAY_TARGET_FRAME_TIME
from integration.desktop.models.desktop_action import DesktopAction
from integration.desktop.models.desktop_target import DesktopTarget
from integration.desktop.models.icon_window_data import IconWindowData
from integration.desktop.models.text_window_data import TextWindowData
from integration.desktop.windows.components.window_selector import highlight_hovered_window
from integration.desktop.windows.user.actions import ACT_WATCH_TARGET, ACT_OBSCURE_TARGET, ACT_ENGAGE_TARGET, ACT_DISENGAGE_TARGET
from integration.desktop.windows.util.animation_util import calculate_next_action_animation_step, draw_animation_frame
from integration.desktop.windows.util.win_constants import get_mouse_pos


# TODO We can create a parent "Animator" class to handle common event loop and other UI-related stuff.
class ActionManager:
    """
    Manages desktop actions, their animation, and window highlights.
    Encapsulates all state previously stored in module-level variables.
    """

    # Needed to be able to keep GDI bitmaps in memory for fast animations.
    G_BUFFERS = []
    INTERPOLATOR_FRAMERATE_REDUCTION = 16

    def __init__(
            self,
            actions: List[DesktopAction],
            action_pairs: Dict[str, tuple],
            on_target_selected_callback: Optional[Callable] = None
    ):
        """
        Initialize the ActionManager.

        :param actions: A list of DesktopAction objects.
        :param action_pairs: A dictionary mapping a base shortcut char
                             to a tuple (normal_action, opposite_action).
        :param on_target_selected_callback: An optional callback that will be
                                            invoked when a target is selected.
        """
        self.active = True
        self.actions = actions
        self.action_pairs = action_pairs
        self.action_targets_dequeue: Deque[DesktopTarget] = deque()

        # Current action can be externally set/changed
        self.current_action: Optional[DesktopAction] = None

        # Store references to any created HWNDs, keyed by action.uid
        self.target_window_data: Dict[int, IconWindowData] = {}
        self.text_window_data: Dict[int, TextWindowData] = {}
        self.action_hwnds: Dict[str, int] = {}
        self.move_target_hwnd: int = -1
        self.text_hwnd: int = -1

        # Store event references for stopping highlight threads
        self.last_stop_event: Optional[threading.Event] = None

        # Called when a highlight selection is made
        self.on_target_selected_callback = on_target_selected_callback

        # Optional: If you were using a 'CURRENT_HWND' somewhere, store it here
        self.current_hwnd: Optional[int] = None

        # Example: Assign the manager's highlight thread function to each action
        ACT_WATCH_TARGET.function = self.run_window_highlight_in_thread
        ACT_OBSCURE_TARGET.function = self.run_window_highlight_in_thread
        ACT_ENGAGE_TARGET.function = self.run_window_highlight_in_thread
        ACT_DISENGAGE_TARGET.function = self.run_window_highlight_in_thread

    def run_window_highlight_in_thread(self) -> threading.Thread:
        """
        Run highlight_hovered_window in a separate thread and notify when it exits.
        Replaces the previous global function, storing thread state in this instance.

        :return: The thread that highlights the hovered window.
        """

        def thread_target():
            # If there's already a stop event running, set it before starting a new one
            if self.last_stop_event is not None:
                self.last_stop_event.set()

            self.last_stop_event = threading.Event()
            highlight_hovered_window(
                stop_event=self.last_stop_event,
                callback=self.on_target_selected_callback
            )

        highlight_thread = threading.Thread(target=thread_target)
        highlight_thread.start()

        return highlight_thread

    def update_action_windows(self, spring_data: dict, current_action: DesktopAction, max_delta: int = 32):
        """
        Update the positions and sizes of the action windows based on the animation data.

        :param spring_data: A dictionary mapping action.uid -> [pos_x, pos_y, vel_x, vel_y].
        :param current_action: The currently selected action.
        :param max_delta: The maximum size for the icon.
        """
        for action in self.actions:
            # Each entry in spring_data is [pos_x, pos_y, vel_x, vel_y].
            pos_x, pos_y, vel_x, vel_y = spring_data[action.uid]

            if vel_x > 0 and vel_y > 0 or action == current_action:
                try:
                    mouse_x, mouse_y = get_mouse_pos()
                    # Move the window to follow the cursor with the calculated offsets
                    win32gui.MoveWindow(
                        self.action_hwnds[action.uid],
                        int(mouse_x - (max_delta + (max_delta - pos_x)) / 2),
                        int(mouse_y - max_delta + (max_delta - pos_y)) - int(max_delta / 2),
                        int(pos_x),
                        int(pos_y),
                        True  # Repaint window after moving
                    )
                except:
                    # Sometimes this fails on Windows; typically if the HWND is invalid or gone.
                    # However, it also sometimes fails randomly?
                    pass

    def manage_action_windows(self):
        """
        Main loop for animating and updating the action icons.
        This function typically runs in a background thread.
        """
        # Initialize spring animation data for all actions
        animation_data = {action.uid: [0.0, 0.0, 0.0, 0.0] for action in self.actions}

        # Apply the window position/size updates
        def update():
            while self.active:
                self.update_action_windows(animation_data, self.current_action)
                time.sleep(DESKTOP_OVERLAY_TARGET_FRAME_TIME)

        threading.Thread(target=update).start()

        frame_index = 0
        interpolated_frame_index = 0

        while self.active:
            # Update the animation (spring effect) based on the current action
            # TODO This is a totally normal way to advance an animation, yes.
            animation_data = calculate_next_action_animation_step(
                self.actions,
                animation_data,
                self.current_action
            )

            if interpolated_frame_index < self.INTERPOLATOR_FRAMERATE_REDUCTION:
                interpolated_frame_index += 1
                time.sleep(DESKTOP_OVERLAY_TARGET_FRAME_TIME)
            else:
                interpolated_frame_index = 0
                if self.move_target_hwnd in self.target_window_data:
                    data = self.target_window_data[self.move_target_hwnd]

                    # Draw this frame
                    draw_animation_frame(self.move_target_hwnd, frame_index, self.target_window_data[self.move_target_hwnd])

                    # Next frame (loop around)
                    frame_index = (frame_index + 1) % data.frame_count
                    time.sleep(DESKTOP_OVERLAY_TARGET_FRAME_TIME)
