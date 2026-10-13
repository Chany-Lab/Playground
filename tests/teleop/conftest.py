"""Let `isaaclab.devices.device_base` import without the Kit app.

`isaaclab.devices/__init__` imports every device and with them `carb`, which only
exists inside Isaac Sim. When `carb` is missing, register a bare package for
`isaaclab.devices` so its plain-Python submodules can still be imported.
"""

import importlib.util
import os
import sys
import types

if "isaaclab.devices" not in sys.modules and importlib.util.find_spec("carb") is None:
    import isaaclab

    package = types.ModuleType("isaaclab.devices")
    package.__path__ = [os.path.join(os.path.dirname(isaaclab.__file__), "devices")]
    sys.modules["isaaclab.devices"] = package
