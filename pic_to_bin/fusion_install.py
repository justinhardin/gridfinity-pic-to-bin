"""Install or uninstall the Fusion 360 pic-to-bin script."""

import argparse
import os
import platform
import shutil
import sys
from importlib import resources
from pathlib import Path

SCRIPT_NAME = "pic_to_bin"


def _fusion_scripts_dir() -> Path:
    """Return the Fusion 360 Scripts directory for the current OS."""
    system = platform.system()
    if system == "Windows":
        appdata = os.environ.get("APPDATA", "")
        if not appdata:
            raise RuntimeError("APPDATA environment variable not set")
        return Path(appdata) / "Autodesk" / "Autodesk Fusion 360" / "API" / "Scripts"
    elif system == "Darwin":
        return (
            Path.home()
            / "Library"
            / "Application Support"
            / "Autodesk"
            / "Autodesk Fusion 360"
            / "API"
            / "Scripts"
        )
    else:
        raise RuntimeError(f"Unsupported OS: {system}. Fusion 360 runs on Windows and macOS only.")


def _bundled_script_dir() -> Path:
    """Return the path to the bundled pic_to_bin_script directory."""
    pkg = resources.files("pic_to_bin") / "pic_to_bin_script"
    # resources.files returns a Traversable; for installed packages this is
    # typically a real filesystem path.
    path = Path(str(pkg))
    if not path.is_dir():
        raise RuntimeError(
            f"Could not locate bundled pic_to_bin_script at {path}. "
            "Is the package installed correctly?"
        )
    return path


def install(target_dir: Path | None = None) -> Path:
    """Copy the Fusion 360 script to the Fusion Scripts directory.

    Returns the destination directory.
    """
    source = _bundled_script_dir()
    dest_parent = target_dir or _fusion_scripts_dir()
    dest = dest_parent / SCRIPT_NAME

    if not dest_parent.exists():
        print(f"Creating Fusion Scripts directory: {dest_parent}")
        dest_parent.mkdir(parents=True, exist_ok=True)

    if dest.exists():
        shutil.rmtree(dest)
        print(f"Replaced existing script at: {dest}")
    else:
        print(f"Installing script to: {dest}")

    # Copy only the script files (not __init__.py or __pycache__)
    dest.mkdir(parents=True, exist_ok=True)
    for filename in ("pic_to_bin.py", "pic_to_bin.manifest"):
        src_file = source / filename
        if src_file.exists():
            shutil.copy2(src_file, dest / filename)

    return dest


def uninstall(target_dir: Path | None = None) -> bool:
    """Remove the Fusion 360 script from the Fusion Scripts directory.

    Returns True if the script was found and removed.
    """
    dest_parent = target_dir or _fusion_scripts_dir()
    dest = dest_parent / SCRIPT_NAME

    if dest.exists():
        shutil.rmtree(dest)
        print(f"Removed: {dest}")
        return True
    else:
        print(f"Script not found at: {dest}")
        return False


def main():
    parser = argparse.ArgumentParser(
        prog="pic-to-bin-fusion",
        description="Install or uninstall the Fusion 360 pic-to-bin script.",
    )
    parser.add_argument(
        "action",
        choices=["install", "uninstall"],
        help="Install or uninstall the Fusion 360 script",
    )
    parser.add_argument(
        "--target-dir",
        type=Path,
        default=None,
        help="Override the Fusion 360 Scripts directory (advanced)",
    )
    args = parser.parse_args()

    try:
        if args.action == "install":
            dest = install(args.target_dir)
            print()
            print("Success! To use the script in Fusion 360:")
            print("  1. Open Fusion 360")
            print("  2. Press Shift+S (Scripts and Add-Ins)")
            print(f'  3. If not listed, click "+" and browse to:\n     {dest}')
            print('  4. Select "pic_to_bin" and click Run')
            print()
            print("The script will prompt you to select bin_config.json via a file dialog.")
            print("Tip: Your generated configs are typically in ./generated/bin_config.json")
        else:
            found = uninstall(args.target_dir)
            if found:
                print("Fusion 360 script uninstalled successfully.")
            else:
                print("Nothing to uninstall.")
                sys.exit(1)
    except RuntimeError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
