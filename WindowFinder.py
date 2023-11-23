import win32gui
import win32con


def getWindowMonitor(window_text):
    window_text = find_main_window(window_text)
    hw = win32gui.FindWindow(None, window_text)
    # win32gui.SetForegroundWindow(hw)
    win32gui.ShowWindow(hw, win32con.SW_NORMAL)
    clientRect = win32gui.GetClientRect(hw)

    left, top = win32gui.ClientToScreen(hw, (clientRect[0], clientRect[1]))
    width, height = int(clientRect[2]), int(clientRect[3])

    monitor = {"top": top, "left": left, "width": width, "height": height}

    return monitor


def find_main_window(starttxt):
    global window_text
    win32gui.EnumChildWindows(0, is_win_ok, starttxt)
    return window_text


def is_win_ok(hwnd, starttext):
    s = win32gui.GetWindowText(hwnd)
    if starttext in s:
        global window_text
        window_text = s
        return None
    return 1