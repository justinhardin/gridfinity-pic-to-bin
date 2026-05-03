# `pic_to_bin/pic_to_bin_addin/resources/` — Fusion add-in resources

Fusion looks for command icons in a `resources/<command-id>/` folder
next to the add-in entry script. The `pic_to_bin/` subfolder here
matches the command id and contains the toolbar icons:

- `16x16.png` — small button (toolbar)
- `32x32.png` — medium (panel)
- `64x64.png` — large (Solid → Create dropdown)

Replace these PNGs to change the icon — Fusion picks them up the next
time the add-in starts.
