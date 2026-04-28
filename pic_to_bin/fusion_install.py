"""Install or uninstall the Fusion 360 pic-to-bin script and add-in."""

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

    Scripts live in <api>/Scripts and add-ins live in <api>/AddIns.
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


def _bundled_dir(name: str) -> Path:
    """Return the path to a bundled folder inside the package."""
    pkg = resources.files("pic_to_bin") / name
    path = Path(str(pkg))
    if not path.is_dir():
        raise RuntimeError(
            f"Could not locate bundled {name} at {path}. "
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
    """Install both the script and the add-in. Returns the destination paths."""
    api_dir = target_api_dir or _fusion_api_dir()
    scripts_root = api_dir / "Scripts"
    addins_root = api_dir / "AddIns"
    scripts_root.mkdir(parents=True, exist_ok=True)
    addins_root.mkdir(parents=True, exist_ok=True)

    src_script = _bundled_dir("pic_to_bin_script")
    src_addin = _bundled_dir("pic_to_bin_addin")

    script_dest = scripts_root / NAME
    addin_dest = addins_root / NAME

    # Script: copy the whole script folder (includes _bin_builder.py).
    _copy_tree(src_script, script_dest)
    print(f"Installed script: {script_dest}")

    # Add-in: copy the addin folder, then copy _bin_builder.py from the
    # script source so the add-in entry can import it locally.
    _copy_tree(src_addin, addin_dest)
    builder_src = src_script / "_bin_builder.py"
    if builder_src.exists():
        shutil.copy2(builder_src, addin_dest / "_bin_builder.py")
    print(f"Installed add-in:  {addin_dest}")

    return {"script": script_dest, "addin": addin_dest}


def uninstall(target_api_dir: Path | None = None) -> dict:
    api_dir = target_api_dir or _fusion_api_dir()
    script_dest = api_dir / "Scripts" / NAME
    addin_dest = api_dir / "AddIns" / NAME

    found = {"script": False, "addin": False}
    if script_dest.exists():
        shutil.rmtree(script_dest)
        print(f"Removed: {script_dest}")
        found["script"] = True
    else:
        print(f"Not found: {script_dest}")
    if addin_dest.exists():
        shutil.rmtree(addin_dest)
        print(f"Removed: {addin_dest}")
        found["addin"] = True
    else:
        print(f"Not found: {addin_dest}")

    return found


def main():
    parser = argparse.ArgumentParser(
        prog="pic-to-bin-fusion",
        description="Install or uninstall the Fusion 360 pic-to-bin "
                    "script and add-in.",
    )
    parser.add_argument(
        "action",
        choices=["install", "uninstall"],
        help="Install or uninstall both the script and the add-in",
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
            paths = install(args.target_dir)
            print()
            print("Success! To use in Fusion 360:")
            print()
            print("  Add-in (toolbar button — recommended):")
            print("    1. Open Fusion 360")
            print("    2. Press Shift+S → Add-Ins tab")
            print(f"    3. Select 'pic_to_bin' → Run")
            print("       (toggle 'Run on Startup' so the button appears every session)")
            print("    4. In a Design workspace, look in Solid > Create → "
                  "'Gridfinity Pic-to-Bin'")
            print()
            print("  Script (alternate):")
            print("    1. Press Shift+S → Scripts tab")
            print("    2. Select 'pic_to_bin' → Run")
            print()
            print("Both pick up bin_config.json from <project>/generated/ "
                  "automatically, falling back to a file dialog.")
        else:
            found = uninstall(args.target_dir)
            if not (found["script"] or found["addin"]):
                print("Nothing was installed; nothing to uninstall.")
                sys.exit(1)
            print("Uninstall complete.")
    except RuntimeError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
