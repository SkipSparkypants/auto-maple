"""A collection of variables shared across multiple modules."""
import queue
from src.detection import detection

#########################
#       Constants       #
#########################
RESOURCES_DIR = 'resources'


#################################
#       Global Variables        #
#################################
# The player's position relative to the minimap
player_pos = (0, 0)

# Describes whether the main bot loop is currently running or not
enabled = False
auto_pot_enabled = True

# If there is another player in the map, Auto Maple will purposely make random human-like mistakes
stage_fright = False

# Represents the current shortest path that the bot is taking
path = []


#############################
#       Shared Modules      #
#############################
# A Routine object that manages the 'machine code' of the current routine
routine = None

# Stores the Layout object associated with the current routine
layout = None

# Shares the main bot loop
bot = None

# Shares the video capture loop
capture = None

# Shares the keyboard listener
listener = None

# Shares the auto pot loop
auto_pot = None

# Shares the gui to all modules
gui = None

queue_max_size = 50
# Detection Infra
model = detection.load_model()
frame_queue = queue.Queue(maxsize=queue_max_size)
detection_inferences = {}
detection_result = None
solve_rune_attempt = 0
