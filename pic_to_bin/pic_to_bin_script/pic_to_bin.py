"""Fusion 360 Script — Gridfinity Bin Generator (script entry point).

Thin entry point that picks a bin_config.json and delegates to
_bin_builder.build_bin(). Build logic, sketch consolidation, and timeline
grouping live in _bin_builder.py and are shared with the add-in form.

Usage:
    1. In Fusion 360: Shift+S → Scripts and Add-Ins → Scripts tab
    2. Click "+" next to "My Scripts" if not already listed
    3. Select "pic_to_bin" → Run
    4. Select your bin_config.json in the file dialog (or skip if a default
       generated/bin_config.json exists at the project root)
"""

import os
import sys
import traceback

import adsk.core


def _user_desktop() -> str:
    """Return the user's Desktop path (Windows %USERPROFILE%\\Desktop or
    macOS ~/Desktop), falling back to the home directory if it's missing
    (e.g., redirected to a cloud-sync provider)."""
    home = os.path.expanduser("~")
    desktop = os.path.join(home, "Desktop")
    return desktop if os.path.isdir(desktop) else home


def _import_builder():
    """Import _bin_builder from this script's directory, forcing a fresh
    read from disk so edits land on the next Run without restarting Fusion.

    Fusion's Python interpreter caches modules in sys.modules across script
    invocations; without an explicit reload, code changes never load.
    """
    import importlib
    script_dir = os.path.dirname(os.path.abspath(__file__))
    if script_dir not in sys.path:
        sys.path.insert(0, script_dir)
    import _bin_builder
    importlib.reload(_bin_builder)
    return _bin_builder


def run(context):
    app = adsk.core.Application.get()
    ui = app.userInterface

    try:
        builder = _import_builder()

        # Default: bin_config.json sitting in generated/ at the project root.
        script_dir = os.path.dirname(os.path.abspath(__file__))
        src_dir = os.path.dirname(script_dir)
        project_dir = os.path.dirname(src_dir)
        default_path = os.path.join(project_dir, "generated", "bin_config.json")

        if os.path.exists(default_path):
            config_path = default_path
        else:
            dialog = ui.createFileDialog()
            dialog.title = "Select Gridfinity Bin Config (bin_config.json)"
            dialog.filter = "JSON files (*.json);;All files (*.*)"
            dialog.initialDirectory = _user_desktop()
            dialog.isMultiSelectEnabled = False
            if dialog.showOpen() != adsk.core.DialogResults.DialogOK:
                return
            config_path = dialog.filename

        result = builder.build_bin(config_path, ui=ui)

        ui.messageBox(
            "Bin created successfully!\n\n"
            f"STL: {result['stl_path']}\n"
            f"STEP: {result['step_path']}\n"
            f"Preview: {result['preview_path']}",
            "Gridfinity Bin Generator")

    except Exception:
        ui.messageBox(
            f"Error:\n{traceback.format_exc()}",
            "Gridfinity Bin Generator — Error")
