# Don't run this tape directly - use ./record.sh, which seeds the data dir, starts
# the mocks, records, and post-processes to 10fps. VHS ignores `Set Framerate` for
# GIF output. The raw GIF comes out at 25fps, and 1432 frames made the GitHub iOS
# app's inline renderer show "?" (~570 frames is known-good). The ffmpeg pass in
# record.sh is mandatory.
# The PEARLARR_DATA_DIR below is rendered in the boot header - keep it in sync with
# record.sh and keep it presentable.
Output demo.gif

Set FontSize 15
Set Width 1130
Set Height 660
Set Padding 14
Set TypingSpeed 55ms
Set Framerate 15

Hide
Type 'export PATH="$PWD/../../.venv/bin:$PATH" PEARLARR_DATA_DIR="$HOME/.local/share/pearlarr" && clear'
Enter
Show

Sleep 600ms
Type "pearlarr run single --sonarr"
Sleep 700ms
Enter
Sleep 50s
Sleep 6s
