import threading
import time
from typing import Optional

import win32con
import win32gui
from pydantic import BaseModel, Field, PrivateAttr

from integration.data.config import INSTALL_PATH
from integration.desktop.models.desktop_target import DesktopTarget
from integration.desktop.windows.action_manager import ActionManager
from integration.desktop.windows.util.animation_util import spring_animation, await_next_frame_if_needed
from integration.desktop.windows.util.gui_util import (
    get_last_input_time, is_cursor_over_window, calculate_euclidean_distance,
    resize_child_to_parent, find_window_by_title, create_host_window, set_window_to_fullscreen, is_ctrl_pressed
)
from integration.desktop.windows.util.win_constants import (
    GUI_CLASS_NAME, GUI_WINDOW_TITLE,
    SMART_WINDOW_WIDTH, SMART_WINDOW_HEIGHT, get_mouse_pos
)


# TODO Experimental pydantic-based functional class definitions.
class SmartWindow(BaseModel):
    """
    A class-based refactor of the original procedural script. This class encapsulates
    the logic for creating and managing a 'smart' floating/animated window.

    Usage:
    ------
    controller = SmartWindowController(window_title="My Window Title")
    # Then call its methods as needed:
    # controller.toggle_floating_mode()
    # controller.toggle_minimal_state()
    # controller.toggle_pinned_state()
    """

    # ------------------------------------------------------------------------------
    # Configuration fields (public). These can be overridden via constructor kwargs
    # if you like (e.g., SmartWindowController(arrive_threshold=100)).
    # ------------------------------------------------------------------------------
    action_manager: ActionManager = Field(default=None)

    arrive_threshold: float = Field(256.0, description="Distance threshold to consider the target 'arrived'")
    ctrl_threshold: int = Field(120, description="Number of frames CTRL has to be pressed to move order the window")
    spring_constant: float = Field(0.04, description="Strength of the spring pull in the window-follow animation")
    damping: float = Field(0.2, description="How quickly the velocity slows (0 < damping < 1 for typical usage)")
    idle_movement_timeout: float = Field(3200.0, description="Seconds of idle before automatically moving the window closer to the mouse")
    frame_time: float = Field(1 / 145.0, description="Animation loop sleep interval in seconds")

    # Control states
    floating_state: bool = Field(False, description="Whether the window is currently in floating mode or fullscreen")
    pinned_state: bool = Field(False, description="Whether the floating window is pinned (i.e., doesn't move)")
    minimal_state: bool = Field(False, description="Whether the floating window is in minimal (half-height) mode")

    # Remember the last docked position
    last_floating_pos_x: int = Field(-1024, description="Last X position of the floating window")
    last_floating_pos_y: int = Field(-1024, description="Last Y position of the floating window")

    # ------------------------------------------------------------------------------
    # Internal (private) attributes. Pydantic does not serialize these by default.
    # We store handles, threads, etc. in private attributes to avoid polluting
    # the public model. You can also define them as non-pydantic fields if you prefer.
    # ------------------------------------------------------------------------------
    _floating_parent_hwnd: Optional[int] = PrivateAttr(default=None)
    _floating_target_hwnd: Optional[int] = PrivateAttr(default=None)
    _roaming_thread: Optional[threading.Thread] = PrivateAttr(default=None)
    _resizing: Optional[bool] = PrivateAttr(default=False)

    class Config:
        # Allow arbitrary types like Window handles, threads, etc.
        arbitrary_types_allowed = True

    # --------------------------------------------------------------------------
    # Constructor
    # --------------------------------------------------------------------------
    def __init__(self, window_title: str, **data):
        """
        :param window_title: The title of the Chromium window to find and embed
        :param data: Additional config overrides for pydantic fields (optional)
        """
        super().__init__(**data)

        matching_windows = find_window_by_title(window_title)
        if not matching_windows:
            print("Window not found!")
            return

        self._floating_target_hwnd = matching_windows[0]
        print(f"Found Open WebUI window: {self._floating_target_hwnd}")

        # Create our borderless parent window
        self._floating_parent_hwnd = create_host_window(
            GUI_CLASS_NAME, GUI_WINDOW_TITLE, SMART_WINDOW_WIDTH, SMART_WINDOW_HEIGHT
        )
        print(f"Created parent window: {self._floating_parent_hwnd}")

        # Go fullscreen to start, or you can choose to start in floating instead.
        # set_to_fullscreen(self._floating_parent_hwnd)

        # Re-parent the Chromium window into our parent
        win32gui.SetParent(self._floating_target_hwnd, self._floating_parent_hwnd)

        # Convert it to a child window style
        child_style = win32gui.GetWindowLong(self._floating_target_hwnd, win32con.GWL_STYLE)
        child_style |= win32con.WS_CHILD
        child_style &= ~(win32con.WS_CAPTION | win32con.WS_THICKFRAME | win32con.WS_SYSMENU)
        win32gui.SetWindowLong(self._floating_target_hwnd, win32con.GWL_STYLE, child_style)

        # Size the child to fill the parent
        resize_child_to_parent(self._floating_parent_hwnd, self._floating_target_hwnd)

        # Optionally start in floating mode. Comment out if you want to remain fullscreen.
        # self.toggle_floating_mode()

        # Start the background thread that handles window-follow logic
        self._roaming_thread = threading.Thread(target=self._roam)
        self._roaming_thread.daemon = True
        self._roaming_thread.start()

    # --------------------------------------------------------------------------
    # Public methods for toggling behaviors
    # --------------------------------------------------------------------------
    def toggle_floating_mode(self):
        """Toggle between floating (portrait) and fullscreen for the parent window."""
        if self._floating_parent_hwnd is None:
            return

        self._resizing = True

        if self.floating_state:
            self.floating_state = False
            print("Switching to fullscreen...")
            self._set_to_fullscreen()
        else:
            self.floating_state = True
            print("Switching to floating...")
            self._set_to_floating()

        self._resizing = False

    def toggle_minimal_state(self):
        """
        Toggle the height of the floating window to half and alpha with a smooth
        damping animation.
        """
        if not self._floating_parent_hwnd:
            return

        self._resizing = True

        rect = win32gui.GetWindowRect(self._floating_parent_hwnd)
        current_height = rect[3] - rect[1]
        current_width = rect[2] - rect[0]

        # A simplistic approach: guess alpha based on current state
        current_alpha = int(.9 * 255) if not self.minimal_state else int(.3 * 255)

        if self.minimal_state:
            # Restore
            target_height = SMART_WINDOW_HEIGHT
            target_width = SMART_WINDOW_WIDTH
            target_alpha = int(.9 * 255)
            print("Exiting minimal state...")
        else:
            # Go minimal
            target_height = SMART_WINDOW_HEIGHT // 2
            target_width = SMART_WINDOW_WIDTH
            target_alpha = int(.3 * 255)
            print("Entering minimal state...")

        spring_animation(
            self._floating_parent_hwnd,
            start_width=current_width,
            end_width=target_width,
            start_height=current_height,
            end_height=target_height,
            start_alpha=current_alpha,
            end_alpha=target_alpha
        )

        # Adjust child
        resize_child_to_parent(self._floating_parent_hwnd, self._floating_target_hwnd)

        # Flip the state
        self.minimal_state = not self.minimal_state
        self._resizing = False

    def toggle_pinned_state(self):
        """If floating, toggle pinned so that the window stops following the user."""
        if self.floating_state:
            self.pinned_state = not self.pinned_state

    # --------------------------------------------------------------------------
    # Internal helpers
    # --------------------------------------------------------------------------
    def _set_to_fullscreen(self):
        """
        Transition smoothly to a fullscreen window with an animation.
        """
        if not self._floating_parent_hwnd:
            return

        set_window_to_fullscreen(self._floating_parent_hwnd)

        # Finally, resize the embedded child (Chromium) to match
        resize_child_to_parent(self._floating_parent_hwnd, self._floating_target_hwnd)

    def _set_to_floating(self):
        """
        Transition smoothly to a smaller floating window with an animation.
        """
        if not self._floating_parent_hwnd:
            return

        self._resizing = True

        rect = win32gui.GetWindowRect(self._floating_parent_hwnd)
        current_height = rect[3] - rect[1]
        current_width = rect[2] - rect[0]
        current_alpha = 255

        target_height = SMART_WINDOW_HEIGHT
        target_width = SMART_WINDOW_WIDTH
        target_alpha = int(.9 * 255)

        spring_animation(
            self._floating_parent_hwnd,
            start_width=current_width,
            end_width=target_width,
            start_height=current_height,
            end_height=target_height,
            start_alpha=current_alpha,
            end_alpha=target_alpha,
            start_x=0,
            start_y=0,
            end_x=self.last_floating_pos_x,
            end_y=self.last_floating_pos_y,
        )

        def resize():
            time.sleep(0.1)
            resize_child_to_parent(self._floating_parent_hwnd, self._floating_target_hwnd)

        # Adjust child
        threading.Thread(target=resize).start()

        self._resizing = False

    def _roam(self):
        """
        Continuously update the window position by following a queue of DesktopTarget
        items + additional user-interactive behaviour. Runs on a background thread.
        """
        current_x, current_y, velocity_x, velocity_y = self._initialize_roaming_variables()
        ctrl_counter, target_opacity, last_idle_movement = 0, 1, time.time()
        current_target = None

        while True:
            start_time = time.time()

            if self._should_roam():
                mouse_x, mouse_y = get_mouse_pos()
                time_since_last_input, time_since_last_idle_movement = self._time_since_last_input(last_idle_movement)

                if self._should_queue_idle_target(time_since_last_idle_movement, time_since_last_input):
                    self._queue_idle_target(mouse_x, mouse_y)
                    last_idle_movement = time.time()

                ctrl_counter = self._update_ctrl_counter(ctrl_counter)

                # if self._should_move_away(ctrl_counter):
                #     new_target = self._create_user_target(mouse_x, mouse_y)
                #     self._set_target_opacity(new_target, 0.025)

                current_target = self._get_next_target(current_target)

                target_opacity = self.action_manager.target_window_data[self.action_manager.move_target_hwnd].opacity

                if current_target:
                    target_opacity = min(self._adjust_opacity(target_opacity, increase=True), 255)
                    current_x, current_y, velocity_x, velocity_y = self._apply_spring_physics(current_x, current_y, velocity_x, velocity_y, current_target)
                    if self._is_target_reached(current_x, current_y, current_target):
                        current_target = None
                else:
                    target_opacity = max(self._adjust_opacity(target_opacity, increase=False), 0)

                self.action_manager.target_window_data[self.action_manager.move_target_hwnd].opacity = target_opacity

                if not self._resizing:
                    self._update_window_position(current_x, current_y)

            win32gui.PumpWaitingMessages()
            await_next_frame_if_needed(start_time, self.frame_time)

    # Utility functions

    def _initialize_roaming_variables(self):
        return float(self.last_floating_pos_x), float(self.last_floating_pos_y), 0.0, 0.0

    def _should_roam(self):
        return self.floating_state and not self.pinned_state

    def _time_since_last_input(self, last_idle_movement):
        return get_last_input_time(), time.time() - last_idle_movement

    def _should_queue_idle_target(self, time_since_last_idle_movement, time_since_last_input):
        return (
                time_since_last_idle_movement > self.idle_movement_timeout
                and len(self.action_manager.action_targets_dequeue) < 1
                and not is_cursor_over_window(self._floating_parent_hwnd)
                and time_since_last_input >= self.idle_movement_timeout
        )

    def _queue_idle_target(self, mouse_x, mouse_y):
        left, top, right, bottom = win32gui.GetWindowRect(self._floating_parent_hwnd)
        center_x = (left + right) / 2
        center_y = (top + bottom) / 2
        midpoint_x = (center_x + mouse_x) / 2
        midpoint_y = (center_y + mouse_y) / 2

        new_target = DesktopTarget(
            x=midpoint_x,
            y=midpoint_y,
            description="IdleTarget"
        )
        self.spring_constant = 0.005
        self.action_manager.action_targets_dequeue.append(new_target)
        self._update_target_window(new_target, 0.025)

    def _update_ctrl_counter(self, ctrl_counter):
        return ctrl_counter + 1 if is_ctrl_pressed() else 0

    def _should_move_away(self, ctrl_counter):
        return (
                ctrl_counter == self.ctrl_threshold
                and not is_cursor_over_window(self._floating_parent_hwnd)
        )

    def _create_user_target(self, mouse_x, mouse_y):
        new_target = DesktopTarget(
            x=mouse_x - 32,
            y=mouse_y - 32,
            description="UserTarget",
            icon_path=rf"{INSTALL_PATH}\agents\res\integration\graphics\ui\target.gif"
        )
        self.action_manager.action_targets_dequeue.append(new_target)
        return new_target

    def _update_target_window(self, new_target, opacity):
        self._update_target_position_async(new_target)
        if self.action_manager.move_target_hwnd in self.action_manager.target_window_data:
            self.action_manager.target_window_data[self.action_manager.move_target_hwnd].opacity = int(opacity * 255)

    def _update_target_position_async(self, new_target):
        def update_position():
            time.sleep(0.1)
            if self.action_manager.move_target_hwnd in self.action_manager.target_window_data:
                self.action_manager.target_window_data[self.action_manager.move_target_hwnd].x = int(new_target.x)
                self.action_manager.target_window_data[self.action_manager.move_target_hwnd].y = int(new_target.y)

        threading.Thread(target=update_position).start()

    def _get_next_target(self, current_target):
        if not current_target and self.action_manager.action_targets_dequeue:
            current_target = self.action_manager.action_targets_dequeue.popleft()
            if current_target.description != "IdleTarget":
                self.spring_constant = 0.04
        return current_target

    def _adjust_opacity(self, target_opacity: int, increase: bool, delta: int = 5):
        return target_opacity + delta if increase else target_opacity - delta

    def _apply_spring_physics(self, current_x, current_y, velocity_x, velocity_y, current_target):
        dx = current_target.x - current_x
        dy = current_target.y - current_y
        velocity_x += dx * self.spring_constant
        velocity_y += dy * self.spring_constant
        velocity_x *= self.damping
        velocity_y *= self.damping
        return current_x + velocity_x, current_y + velocity_y, velocity_x, velocity_y

    def _is_target_reached(self, current_x, current_y, current_target):
        return calculate_euclidean_distance(current_x, current_y, current_target.x, current_target.y) <= self.arrive_threshold

    def _update_window_position(self, current_x, current_y):
        rect = win32gui.GetWindowRect(self._floating_parent_hwnd)
        current_width = rect[2] - rect[0]
        current_height = rect[3] - rect[1]
        win32gui.MoveWindow(
            self._floating_parent_hwnd,
            int(current_x - (current_width / 2)),
            int(current_y - (current_height / 4)),
            current_width,
            current_height,
            True
        )
        self.last_floating_pos_x, self.last_floating_pos_y = int(current_x), int(current_y)
