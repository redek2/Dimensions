# Window
SCREEN_WIDTH = 1280
SCREEN_HEIGHT = 720
FPS = 60
CAPTION = "Dimensions"

# Font
FONT_NAME = "arial"
FONT_SIZE = 16

# Colors
BACKGROUND_COLOR = "black"
LINE_COLOR = "white"
FPS_COLOR = "yellow"

class AppState():
    def __init__(self):
        self.dimensions = 4
        self.rotation_speed = 0.01
        self.scale = 1000
        self.distance = 3.0
        self.active_planes = []
        self.angles = {}