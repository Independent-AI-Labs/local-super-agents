#
# Example usage
#
from integration.desktop.models.desktop_action import DesktopAction

# Create the DesktopAction instances
ACT_WATCH_TARGET = DesktopAction(
    uid="watch_target",
    name="Watch Target",
    icon_path="..\\..\\res\\img\\eye_color.png",
    color=(187, 77, 235),
    shortcut_key="w",
    function=None  # Will be set to the manager method, or used externally
)

ACT_OBSCURE_TARGET = DesktopAction(
    uid="obscure_target",
    name="Obscure Target",
    icon_path="..\\..\\res\\img\\eye_closed.png",
    color=(187, 77, 235),
    shortcut_key="!w",
    function=None
)

ACT_ENGAGE_TARGET = DesktopAction(
    uid="engage_target",
    name="Engage Target",
    icon_path="..\\..\\res\\img\\reach.png",
    color=(235, 77, 77),
    shortcut_key="e",
    function=None
)

ACT_DISENGAGE_TARGET = DesktopAction(
    uid="disengage_target",
    name="Disengage Target",
    icon_path="..\\..\\res\\img\\reach_over.png",
    color=(235, 77, 77),
    shortcut_key="!e",
    function=None
)

ACTIONS = [
    ACT_WATCH_TARGET,
    ACT_OBSCURE_TARGET,
    ACT_ENGAGE_TARGET,
    ACT_DISENGAGE_TARGET
]

ACTION_PAIRS = {
    "w": (ACT_WATCH_TARGET, ACT_OBSCURE_TARGET),
    "e": (ACT_ENGAGE_TARGET, ACT_DISENGAGE_TARGET)
}

# Now you can control the current action externally by setting:
# manager.current_action = ACT_WATCH_TARGET
# Or you can pass a callback that references manager.current_action:
#   get_current_action_callback=lambda: manager.current_action

# A typical usage pattern might involve:
# thread = threading.Thread(
#     target=manager.update_action_target_position,
#     args=(lambda: manager.current_action, )
# )
# thread.start()
