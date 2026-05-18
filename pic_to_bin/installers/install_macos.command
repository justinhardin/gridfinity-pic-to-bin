#!/bin/bash
# Pic-to-Bin Fusion 360 Add-In Installer (macOS)
#
# Double-click in Finder to run. Detects Fusion's user-API folder,
# copies AddIns/pic_to_bin into it, and prints the next steps. Aborts
# with an error if Fusion's API folder can't be found (rather than
# creating a stray empty folder when Fusion isn't installed yet).

set -e

cd "$(dirname "$0")"

echo "==========================================================="
echo " Pic-to-Bin Fusion 360 Add-In Installer (macOS)"
echo "==========================================================="
echo

BASE="$HOME/Library/Application Support/Autodesk"
TARGET=""
for D in "Autodesk Fusion 360" "Autodesk Fusion"; do
    if [ -d "$BASE/$D/API" ]; then
        TARGET="$BASE/$D/API/AddIns"
        break
    fi
done

if [ -z "$TARGET" ]; then
    echo "Could not find a Fusion 360 user-API folder. Looked in:"
    echo "  $BASE/Autodesk Fusion 360/API"
    echo "  $BASE/Autodesk Fusion/API"
    echo
    echo "If Fusion 360 isn't installed yet, install it from"
    echo "https://www.autodesk.com/products/fusion/personal and open it"
    echo "at least once, then re-run this installer."
    echo
    read -n 1 -s -r -p "Press any key to close..."
    echo
    exit 1
fi

echo "Installing to:"
echo "  $TARGET/pic_to_bin"
echo

mkdir -p "$TARGET"
rm -rf "$TARGET/pic_to_bin"
cp -R "AddIns/pic_to_bin" "$TARGET/pic_to_bin"

echo
echo "==========================================================="
echo " Success! Finish setup inside Fusion 360:"
echo "==========================================================="
echo "  1. Open Fusion 360"
echo "  2. Press Shift+S, switch to the Add-Ins tab"
echo "  3. Select 'pic_to_bin' and click Run"
echo '  4. Tick "Run on Startup" so it loads every session'
echo
echo "The new button appears under:"
echo "  Solid > Create > Gridfinity Pic-to-Bin"
echo
read -n 1 -s -r -p "Press any key to close..."
echo
