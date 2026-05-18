"""Fusion 360 Add-In — Gridfinity Bin Generator.

Registers a "Gridfinity Pic-to-Bin" command in the Solid > Create panel
of the Design workspace. Clicking the button picks a bin_config.json and
delegates the build to _bin_builder.build_bin().
"""

import os
import sys
import traceback

import adsk.core


CMD_ID = "picToBinCreateBinCmd"
CMD_NAME = "Gridfinity Pic-to-Bin"
CMD_TOOLTIP = "Generate a gridfinity bin from a bin_config.json"
WORKSPACE_ID = "FusionSolidEnvironment"
PANEL_ID = "SolidCreatePanel"

# Command input ids for the pre-build options dialog.
INPUT_NOTE_ID = "picToBinNote"
INPUT_STL_ID = "picToBinExportStl"
INPUT_STEP_ID = "picToBinExportStep"
INPUT_PREVIEW_ID = "picToBinExportPreview"

# Resource folder for the toolbar icon (relative to this file).
RESOURCES = "./resources/pic_to_bin"

# Keep handler refs alive — Fusion drops listeners that get garbage-collected.
_handlers = []


def _addin_dir():
    return os.path.dirname(os.path.abspath(__file__))


def _user_desktop() -> str:
    """Return the user's Desktop path (Windows %USERPROFILE%\\Desktop or
    macOS ~/Desktop), falling back to the home directory if it's missing
    (e.g., redirected to a cloud-sync provider)."""
    home = os.path.expanduser("~")
    desktop = os.path.join(home, "Desktop")
    return desktop if os.path.isdir(desktop) else home


def _import_builder():
    """Import _bin_builder, forcing a fresh read from disk on each click so
    edits land on the next button press without restarting Fusion.

    Without the reload, sys.modules retains the first-loaded version even
    after the file on disk has changed (e.g. after re-running install).
    """
    import importlib
    addin_dir = _addin_dir()
    if addin_dir not in sys.path:
        sys.path.insert(0, addin_dir)
    import _bin_builder
    importlib.reload(_bin_builder)
    return _bin_builder


def _pick_config(ui):
    """Default to <project>/generated/bin_config.json; else open a file dialog."""
    addin_dir = _addin_dir()
    src_dir = os.path.dirname(addin_dir)
    project_dir = os.path.dirname(src_dir)
    default_path = os.path.join(project_dir, "generated", "bin_config.json")
    if os.path.exists(default_path):
        return default_path

    dialog = ui.createFileDialog()
    dialog.title = "Select Gridfinity Bin Config (bin_config.json)"
    dialog.filter = "JSON files (*.json);;All files (*.*)"
    dialog.initialDirectory = _user_desktop()
    dialog.isMultiSelectEnabled = False
    if dialog.showOpen() != adsk.core.DialogResults.DialogOK:
        return None
    return dialog.filename


class _ExecuteHandler(adsk.core.CommandEventHandler):
    def __init__(self):
        super().__init__()

    def notify(self, args):
        ui = adsk.core.Application.get().userInterface
        try:
            cmd_args = adsk.core.CommandEventArgs.cast(args)
            inputs = cmd_args.command.commandInputs
            write_stl = inputs.itemById(INPUT_STL_ID).value
            write_step = inputs.itemById(INPUT_STEP_ID).value
            write_preview = inputs.itemById(INPUT_PREVIEW_ID).value

            config_path = _pick_config(ui)
            if not config_path:
                return
            builder = _import_builder()
            result = builder.build_bin(
                config_path,
                ui=ui,
                write_stl=write_stl,
                write_step=write_step,
                write_preview=write_preview,
                export_dir=_user_desktop(),
            )

            lines = ["Bin created successfully!"]
            saved = [
                ("STL", result.get("stl_path")),
                ("STEP", result.get("step_path")),
                ("Preview", result.get("preview_path")),
            ]
            saved = [(label, p) for label, p in saved if p]
            if saved:
                lines.append("")
                lines.append("Saved to your desktop:")
                for label, path in saved:
                    lines.append(f"  {label}: {path}")
            ui.messageBox("\n".join(lines), CMD_NAME)
        except Exception:
            ui.messageBox(
                f"Error:\n{traceback.format_exc()}",
                f"{CMD_NAME} — Error")


class _CommandCreatedHandler(adsk.core.CommandCreatedEventHandler):
    def __init__(self):
        super().__init__()

    def notify(self, args):
        cmd = adsk.core.Command.cast(args.command)
        cmd.okButtonText = "Build bin"

        inputs = cmd.commandInputs
        # Plain explainer above the checkboxes — TextBoxCommandInput is the
        # idiomatic Fusion way to drop static help text into a command dialog.
        # Keep this short — Fusion's command palette is narrow and the box
        # truncates with "..." when text overflows the visible width.
        inputs.addTextBoxCommandInput(
            INPUT_NOTE_ID, "",
            "Optionally save copies to your Desktop:",
            1, True)
        # Long labels get truncated and push the checkbox glyph itself off
        # the right edge of the narrow palette, making the input look like
        # plain text. Keep labels short; put the filename detail in the
        # tooltipDescription (shown on hover) so users still see what file
        # each option writes.
        stl_in = inputs.addBoolValueInput(
            INPUT_STL_ID, "Save STL to Desktop", True, "", False)
        stl_in.tooltip = "Save STL to Desktop"
        stl_in.tooltipDescription = (
            "Writes gridfinity_bin.stl to your desktop. "
            "Slicer-ready mesh export."
        )
        step_in = inputs.addBoolValueInput(
            INPUT_STEP_ID, "Save STEP to Desktop", True, "", False)
        step_in.tooltip = "Save STEP to Desktop"
        step_in.tooltipDescription = (
            "Writes gridfinity_bin.step to your desktop. "
            "Re-importable parametric format."
        )
        preview_in = inputs.addBoolValueInput(
            INPUT_PREVIEW_ID, "Save Preview PNG to Desktop", True, "", False)
        preview_in.tooltip = "Save Preview PNG to Desktop"
        preview_in.tooltipDescription = (
            "Writes gridfinity_bin_preview.png to your desktop. "
            "1920×1080 viewport screenshot of the finished bin."
        )

        on_execute = _ExecuteHandler()
        cmd.execute.add(on_execute)
        _handlers.append(on_execute)


def run(context):
    ui = adsk.core.Application.get().userInterface
    try:
        cmd_def = ui.commandDefinitions.itemById(CMD_ID)
        if cmd_def is None:
            resources_path = os.path.join(_addin_dir(),
                                          "resources", "pic_to_bin")
            cmd_def = ui.commandDefinitions.addButtonDefinition(
                CMD_ID, CMD_NAME, CMD_TOOLTIP, resources_path)

        on_created = _CommandCreatedHandler()
        cmd_def.commandCreated.add(on_created)
        _handlers.append(on_created)

        workspace = ui.workspaces.itemById(WORKSPACE_ID)
        panel = workspace.toolbarPanels.itemById(PANEL_ID)
        if panel.controls.itemById(CMD_ID) is None:
            panel.controls.addCommand(cmd_def)

    except Exception:
        if ui:
            ui.messageBox(
                f"Failed to load Gridfinity Pic-to-Bin add-in:\n"
                f"{traceback.format_exc()}",
                f"{CMD_NAME} — Error")


def stopped(context):
    ui = adsk.core.Application.get().userInterface
    try:
        workspace = ui.workspaces.itemById(WORKSPACE_ID)
        if workspace:
            panel = workspace.toolbarPanels.itemById(PANEL_ID)
            if panel:
                ctrl = panel.controls.itemById(CMD_ID)
                if ctrl:
                    ctrl.deleteMe()
        cmd_def = ui.commandDefinitions.itemById(CMD_ID)
        if cmd_def:
            cmd_def.deleteMe()
    except Exception:
        if ui:
            ui.messageBox(
                f"Failed to unload Gridfinity Pic-to-Bin add-in:\n"
                f"{traceback.format_exc()}",
                f"{CMD_NAME} — Error")
    finally:
        _handlers.clear()
