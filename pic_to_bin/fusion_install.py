"""Install or uninstall the Fusion 360 pic-to-bin add-in."""

import argparse
import os
import platform
import shutil
import sys
from importlib import resources
from pathlib import Path

NAME = "pic_to_bin"


def _fusion_api_dir() -> Path:
    """Return the Fusion 360 API directory for the current OS.

    Add-ins live in <api>/AddIns.
    """
    system = platform.system()
    if system == "Windows":
        appdata = os.environ.get("APPDATA", "")
        if not appdata:
            raise RuntimeError("APPDATA environment variable not set")
        return Path(appdata) / "Autodesk" / "Autodesk Fusion 360" / "API"
    elif system == "Darwin":
        return (
            Path.home()
            / "Library"
            / "Application Support"
            / "Autodesk"
            / "Autodesk Fusion 360"
            / "API"
        )
    else:
        raise RuntimeError(
            f"Unsupported OS: {system}. Fusion 360 runs on Windows and macOS only.")


def _bundled_addin_dir() -> Path:
    """Return the path to the bundled add-in source inside the package."""
    pkg = resources.files("pic_to_bin") / "pic_to_bin_addin"
    path = Path(str(pkg))
    if not path.is_dir():
        raise RuntimeError(
            f"Could not locate bundled pic_to_bin_addin at {path}. "
            "Is the package installed correctly?")
    return path


def _copy_tree(src: Path, dest: Path) -> None:
    """Copy src into dest (replacing dest), excluding __pycache__."""
    if dest.exists():
        shutil.rmtree(dest)
    shutil.copytree(
        src, dest,
        ignore=shutil.ignore_patterns("__pycache__", ".vscode", ".*"))


def install(target_api_dir: Path | None = None) -> dict:
    """Install the add-in. Returns the destination path."""
    api_dir = target_api_dir or _fusion_api_dir()
    addins_root = api_dir / "AddIns"
    addins_root.mkdir(parents=True, exist_ok=True)

    src = _bundled_addin_dir()
    dest = addins_root / NAME
    _copy_tree(src, dest)
    print(f"Installed add-in: {dest}")

    return {"addin": dest}


def uninstall(target_api_dir: Path | None = None) -> dict:
    api_dir = target_api_dir or _fusion_api_dir()
    addin_dest = api_dir / "AddIns" / NAME

    # Also clean up the legacy Scripts/pic_to_bin folder from earlier
    # versions of this installer. Users who installed before the
    # add-in-only consolidation still have it sitting in Fusion's
    # Scripts list; remove it on uninstall so they don't see two
    # entries.
    legacy_script_dest = api_dir / "Scripts" / NAME

    found = {"addin": False, "legacy_script": False}
    if addin_dest.exists():
        shutil.rmtree(addin_dest)
        print(f"Removed: {addin_dest}")
        found["addin"] = True
    else:
        print(f"Not found: {addin_dest}")
    if legacy_script_dest.exists():
        shutil.rmtree(legacy_script_dest)
        print(f"Removed legacy script: {legacy_script_dest}")
        found["legacy_script"] = True

    return found


def main():
    parser = argparse.ArgumentParser(
        prog="pic-to-bin-fusion",
        description="Install or uninstall the Fusion 360 pic-to-bin add-in.",
    )
    parser.add_argument(
        "action",
        choices=["install", "uninstall"],
        help="Install or uninstall the add-in",
    )
    parser.add_argument(
        "--target-dir",
        type=Path,
        default=None,
        help="Override the Fusion 360 API directory (advanced)",
    )
    args = parser.parse_args()

    try:
        if args.action == "install":
            install(args.target_dir)
            print()
            print("Success! To use in Fusion 360:")
            print()
            print("  1. Open Fusion 360")
            print("  2. Press Shift+S → Add-Ins tab")
            print("  3. Select 'pic_to_bin' → Run")
            print("     (tick 'Run on Startup' so the button appears every session)")
            print("  4. In a Design workspace, the new button appears under")
            print("     Solid → Create → 'Gridfinity Pic-to-Bin'")
            print()
            print("Clicking the button picks up bin_config.json from")
            print("<project>/generated/ automatically, falling back to a file")
            print("dialog if no project config is found.")
        else:
            found = uninstall(args.target_dir)
            if not (found["addin"] or found["legacy_script"]):
                print("Nothing was installed; nothing to uninstall.")
                sys.exit(1)
            print("Uninstall complete.")
    except RuntimeError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
