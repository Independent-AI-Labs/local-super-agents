import ctypes
from ctypes import wintypes

import win32gui
from _ctypes import byref

GUI_CLASS_NAME = "LocalSuperAgentsGUI"
GUI_WINDOW_TITLE = "Super-Agents"

SMART_WINDOW_WIDTH = 420
SMART_WINDOW_HEIGHT = 615

try:
    # Structure definitions:
    class GdiplusStartupInput(ctypes.Structure):
        _fields_ = [
            ("GdiplusVersion", ctypes.c_uint32),
            ("DebugEventCallback", ctypes.c_void_p),
            ("SuppressBackgroundThread", wintypes.BOOL),
            ("SuppressExternalCodecs", wintypes.BOOL)
        ]


    gdiplus_token = ctypes.c_ulong()
    gdiplus_startup_input = GdiplusStartupInput()
    gdiplus_startup_input.GdiplusVersion = 1  # GDI+ version 1

    gdiplus = ctypes.WinDLL("gdiplus.dll")

    # somewhere early in your code, before GDI+ calls:
    gdiplus.GdiplusStartup(
        ctypes.byref(gdiplus_token),
        ctypes.byref(gdiplus_startup_input),
        None
    )
except:
    raise OSError

# GDI+ constants / function pointers
gdiplus = ctypes.windll.gdiplus
user32 = ctypes.windll.user32
gdi32 = ctypes.windll.gdi32

gdiplus.GdipGetImageWidth.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint)]
gdiplus.GdipGetImageWidth.restype = ctypes.c_int

# Uhh, this case because Windows naming I guess?
SmoothingModeAntiAlias = 2
SmoothingModeHighQuality = 4
InterpolationModeHighQualityBicubic = 7

AC_SRC_OVER = 0x00
AC_SRC_ALPHA = 0x01
ULW_ALPHA = 0x00000002

# For brevity, some function prototypes are assumed declared.
# E.g.:
# GdipCreateFromHDC, GdipCreateBitmapFromFile, GdipDrawImageRectI, GdipSetSmoothingMode, etc.
# Also define any needed constants:
UnitPixel = 2

# Frame dimension for time-based frames in GIF
FrameDimensionTime = ctypes.c_byte * 16  # In GDI+ C headers, this is a GUID
# In practice, you'd fill this with the actual GUID for time-based frames:
# {6AEDBD6D-3FB5-418A-83A6-7F45229DC872}
FrameDimensionTimeValue = FrameDimensionTime(
    0x6A, 0xED, 0xBD, 0x6D,
    0x3F, 0xB5,
    0x41, 0x8A,
    0x83, 0xA6,
    0x7F, 0x45,
    0x22, 0x9D,
    0xC8, 0x72
)

# ------------------------------------------------------------------------
# Declare the GdipCreateBitmapFromScan0 prototype
#    We'll create GDI+ bitmaps from raw RGBA data in memory.
# ------------------------------------------------------------------------
# C prototype:
# GpStatus WINGDIPAPI GdipCreateBitmapFromScan0(
#     INT width,
#     INT height,
#     INT stride,
#     PixelFormat format,
#     BYTE* scan0,
#     GpBitmap** bitmap
# );
#
# We'll define it with ctypes:
gdiplus.GdipCreateBitmapFromScan0.argtypes = [
    ctypes.c_int,  # width
    ctypes.c_int,  # height
    ctypes.c_int,  # stride (bytes per row)
    ctypes.c_int,  # PixelFormat
    ctypes.c_void_p,  # raw pointer to pixel data (can be None)
    ctypes.POINTER(ctypes.c_void_p),  # out param: GpBitmap**
]
gdiplus.GdipCreateBitmapFromScan0.restype = ctypes.c_int

# GDI+ PixelFormat constants. For 32-bit RGBA, it's usually PixelFormat32bppARGB.
# According to GDI+ docs, that's 0x26200A.
PixelFormat32bppARGB = 0x26200A


class GdiplusPropertyItem(ctypes.Structure):
    _fields_ = [
        ("id", ctypes.c_ulong),
        ("length", ctypes.c_ulong),
        ("type", ctypes.c_ushort),
        ("value", ctypes.c_void_p),
    ]


class RGBColor(ctypes.Structure):
    _fields_ = [("r", ctypes.c_ubyte),
                ("g", ctypes.c_ubyte),
                ("b", ctypes.c_ubyte)]


class BITMAPINFOHEADER(ctypes.Structure):
    _fields_ = [
        ("biSize", ctypes.c_uint32),
        ("biWidth", ctypes.c_int32),
        ("biHeight", ctypes.c_int32),
        ("biPlanes", ctypes.c_uint16),
        ("biBitCount", ctypes.c_uint16),
        ("biCompression", ctypes.c_uint32),
        ("biSizeImage", ctypes.c_uint32),
        ("biXPelsPerMeter", ctypes.c_int32),
        ("biYPelsPerMeter", ctypes.c_int32),
        ("biClrUsed", ctypes.c_uint32),
        ("biClrImportant", ctypes.c_uint32),
    ]


class BITMAPINFO(ctypes.Structure):
    _fields_ = [
        ("bmiHeader", BITMAPINFOHEADER),
        ("bmiColors", ctypes.c_uint32 * 1),  # This can be extended if needed
    ]


class BLENDFUNCTION(ctypes.Structure):
    _fields_ = [
        ("BlendOp", ctypes.c_byte),
        ("BlendFlags", ctypes.c_byte),
        ("SourceConstantAlpha", ctypes.c_ubyte),
        ("AlphaFormat", ctypes.c_byte)
    ]


# Define the LASTINPUTINFO structure
class LASTINPUTINFO(ctypes.Structure):
    _fields_ = [("cbSize", wintypes.UINT), ("dwTime", wintypes.DWORD)]


class Rect(ctypes.Structure):
    _fields_ = [
        ("X", ctypes.c_int),
        ("Y", ctypes.c_int),
        ("Width", ctypes.c_int),
        ("Height", ctypes.c_int)
    ]


# Define the function prototypes
SetLayeredWindowAttributes = ctypes.windll.user32.SetLayeredWindowAttributes

# Define Font Style Constants (if not already defined)
FontStyleRegular = 0  # Regular font style
FontStyleBold = 1  # Bold font style
FontStyleItalic = 2  # Italic font style
FontStyleBoldItalic = 3
FontStyleUnderline = 4
FontStyleStrikeout = 8

# Define constants
FR_PRIVATE = 0x10  # Load the font as private
FR_NOT_ENUM = 0x20  # Do not add the font to the system enumeration

# Define the callback function type
EnumWindowsProc = ctypes.WINFUNCTYPE(
    wintypes.BOOL,
    wintypes.HWND,
    wintypes.LPARAM
)


def list_available_fonts():
    installed_fonts = []
    font_collection = ctypes.c_void_p()

    # Create a font collection
    status = gdiplus.GdipNewInstalledFontCollection(byref(font_collection))
    if status != 0:
        raise OSError(f"GdipNewInstalledFontCollection failed with status {status}")

    # Get the number of font families
    num_families = ctypes.c_int()
    status = gdiplus.GdipGetFontCollectionFamilyCount(font_collection, byref(num_families))
    if status != 0:
        raise OSError(f"GdipGetFontCollectionFamilyCount failed with status {status}")

    # Create an array to store font families
    families = (ctypes.c_void_p * num_families.value)()
    count = ctypes.c_int()
    status = gdiplus.GdipGetFontCollectionFamilyList(
        font_collection, num_families.value, families, byref(count)
    )
    if status != 0:
        raise OSError(f"GdipGetFontCollectionFamilyList failed with status {status}")

    # Retrieve names of each font family
    for i in range(count.value):
        font_family = ctypes.cast(families[i], ctypes.POINTER(ctypes.c_void_p))  # Cast to FontFamily type
        name = ctypes.create_unicode_buffer(32)
        status = gdiplus.GdipGetFamilyName(font_family, name, 0)
        if status == 0:
            installed_fonts.append(name.value)

    return installed_fonts


# print(list_available_fonts())


def load_font_from_file(font_path, font_name, font_size):
    # Add the font resource to the system
    font_count = gdi32.AddFontResourceExW(font_path, FR_PRIVATE | FR_NOT_ENUM, None)
    if font_count == 0:
        raise OSError("Failed to add font resource")

    # Create the font
    font_family = ctypes.c_void_p()
    status = gdiplus.GdipCreateFontFamilyFromName(
        font_name,
        None,
        byref(font_family)
    )
    if status != 0:
        # Remove the font if it fails
        gdi32.RemoveFontResourceExW(font_path, FR_PRIVATE | FR_NOT_ENUM, None)
        raise OSError(f"GdipCreateFontFamilyFromName failed with status {status}")

    font = ctypes.c_void_p()
    status = gdiplus.GdipCreateFont(font_family, font_size, FontStyleRegular, 0, byref(font))
    if status != 0:
        # Remove the font if it fails
        gdi32.RemoveFontResourceExW(font_path, FR_PRIVATE | FR_NOT_ENUM, None)
        raise OSError(f"GdipCreateFont failed with status {status}")

    return font


def get_mouse_pos():
    try:
        return win32gui.GetCursorPos()
    except:
        return 0, 0
