"""Print the pointer position ("x y") whenever it changes. Arena runs under XWayland, and X11 reports
the pointer to any client, which Wayland does not. Read by overlay/draft.qml."""

import ctypes
import ctypes.util
import glob
import sys
import time

x = None
for lib in filter(None, [ctypes.util.find_library("X11"), *glob.glob("/nix/store/*-libx11-*/lib/libX11.so.6")]):
    try:
        x = ctypes.CDLL(lib)  # the store also holds 32-bit builds (Steam); skip those
        break
    except OSError:
        pass
if not x:
    sys.exit("libX11 not found")
x.XOpenDisplay.restype = ctypes.c_void_p
display = x.XOpenDisplay(None)
if not display:
    sys.exit("no X display")
display = ctypes.c_void_p(display)
root = ctypes.c_ulong(x.XDefaultRootWindow(display))

last = None
while True:
    r, c = ctypes.c_ulong(), ctypes.c_ulong()
    rx, ry, wx, wy = (ctypes.c_int() for _ in range(4))
    mask = ctypes.c_uint()
    x.XQueryPointer(display, root, ctypes.byref(r), ctypes.byref(c), ctypes.byref(rx), ctypes.byref(ry),
                    ctypes.byref(wx), ctypes.byref(wy), ctypes.byref(mask))
    if (rx.value, ry.value) != last:
        last = (rx.value, ry.value)
        print(*last, flush=True)
    time.sleep(0.05)
