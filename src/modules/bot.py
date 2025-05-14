"""An interpreter that reads and executes user-created routines."""

import threading
import time
import git
import cv2
import traceback
from src.modules.notifier import send_to_webhook
from src.common import config, utils
from src.routine.routine import Routine
from src.command_book.command_book import CommandBook
from src.routine.components import Point
from src.common.vkeys import press
from src.common.interfaces import Configurable
from src.detection import detection
import queue

from random import randint


# The rune's buff icon
RUNE_BUFF_TEMPLATE = cv2.imread('assets/rune_buff_template.jpg', 0)


class Bot(Configurable):
    """A class that interprets and executes user-defined routines."""

    DEFAULT_CONFIG = {
        'Interact': 'space',
        'Feed Pet': '6',
        'Change Channel': ','
    }

    def __init__(self):
        """Loads a user-defined routine on start up and initializes this Bot's main thread."""

        super().__init__('keybindings')
        config.bot = self

        self.NUM_DETECTION_WORKERS = 2
        self.NUM_FRAMES_TO_PROCESS = self.NUM_DETECTION_WORKERS * 10
        self.TIME_BETWEEN_FRAMES = 0.2
        self.TIME_TO_SOLVE = 10

        self.cc_flag = False
        self.command_book = None            # CommandBook instance
        self.rune_active = False
        self.rune_pos = (0, 0)
        self.rune_closest_pos = (0, 0)      # Location of the Point closest to rune
        self.submodules = []
        self.time_in_map = time.time()

        # self.module_name = None
        # self.buff = components.Buff()

        # self.command_book = {}
        # for c in (components.Wait, components.Walk, components.Fall,
        #           components.Move, components.Adjust, components.Buff):
        #     self.command_book[c.__name__.lower()] = c

        config.routine = Routine()

        self.ready = False
        self.thread = threading.Thread(target=self._main)
        self.thread.daemon = True

    def start(self):
        """
        Starts this Bot object's thread.
        :return:    None
        """

        self.update_submodules()
        print('\n[~] Started main bot loop')
        self.thread.start()

    def _main(self):
        """
        The main body of Bot that executes the user's routine.
        :return:    None
        """
        self.ready = True
        config.listener.enabled = True
        last_fed = time.time()
        while True:
            if config.enabled and len(config.routine) > 0:
                if self.cc_flag:
                    self.change_channel()
                    self.cc_flag = False

                # Buff and feed pets
                self.command_book.buff.main()
                pet_settings = config.gui.settings.pets
                auto_feed = pet_settings.auto_feed.get()
                num_pets = pet_settings.num_pets.get()
                now = time.time()
                if auto_feed and now - last_fed > 1200 / num_pets:
                    press(self.config['Feed Pet'], 1)
                    last_fed = now

                # Settings
                settings = config.gui.settings.bot_settings
                self.use_tensorflow = settings.use_tensorflow.get()

                # Highlight the current Point
                config.gui.view.routine.select(config.routine.index)
                config.gui.view.details.display_info(config.routine.index)

                # Execute next Point in the routine
                element = config.routine[config.routine.index]
                if self.rune_active and isinstance(element, Point) \
                        and element.location == self.rune_closest_pos:
                    self._solve_rune()
                element.execute()
                config.routine.step()
            else:
                time.sleep(0.01)

    @utils.run_if_enabled
    def _solve_rune(self):
        """
        Moves to the position of the rune and adds frames to the frame queue.
        :param sct:     The mss instance object with which to take screenshots.
        :return:        None
        """
        config.auto_pot_enabled = False
        move = self.command_book['move']
        move(*self.rune_pos).execute()
        adjust = self.command_book['adjust']
        self.command_book.deff.main()
        adjust(*self.rune_pos).execute()
        time.sleep(0.3)

        print('\nSolving rune:')
        config.detection_result = None
        press(self.config['Interact'], 1, down_time=0.2)  # Inherited from Configurable
        time.sleep(1.0)

        for _ in range(self.NUM_FRAMES_TO_PROCESS):
            try:
                config.frame_queue.put_nowait(config.capture.frame)
            except queue.Full:
                #clear the queue
                config.frame_queue.queue.clear()
            time.sleep(self.TIME_BETWEEN_FRAMES)

        now = time.time()
        while (time.time() - now < self.TIME_TO_SOLVE and
               config.frame_queue.unfinished_tasks and
               config.detection_result is not None):
            time.sleep(0.1)

        if config.detection_result is not None:
            print('Solution found, entering result')
            for arrow in config.detection_result:
                press(arrow, 1, down_time=0.15)
            self.rune_active = False
            config.solve_rune_attempt = 0
            send_to_webhook('solved rune')
        if config.solve_rune_attempt > 3:
            config.auto_pot_enabled = False
            self.change_channel()
            config.solve_rune_attempt = 0

        config.auto_pot_enabled = False
        config.detection_result = None
        #print(config.detection_inferences)
        config.detection_inferences.clear()
        config.solve_rune_attempt += 1
        return

    def change_channel(self):
        """
        Changes to a random channel
        :return:        None
        """
        print('\nChanging Channel')
        self.command_book.deff.main()
        time.sleep(5.00)
        num_steps = randint(1, 10)
        press(self.config['Change Channel'], 1)
        for i in range(num_steps):
            key = randint(1, 4)
            if key == 1:
                press('left', 1)
            elif key == 2:
                press('right', 1)
            elif key == 3:
                press('down', 1)
            else:
                press('up', 1)
        press('enter', 1)

        self.rune_active = False
        send_to_webhook('change channel')


    def load_commands(self, file):
        try:
            self.command_book = CommandBook(file)
            config.gui.settings.update_class_bindings()
        except ValueError:
            pass    # TODO: UI warning popup, say check cmd for errors
        #
        # utils.print_separator()
        # print(f"[~] Loading command book '{basename(file)}':")
        #
        # ext = splitext(file)[1]
        # if ext != '.py':
        #     print(f" !  '{ext}' is not a supported file extension.")
        #     return False
        #
        # new_step = components.step
        # new_cb = {}
        # for c in (components.Wait, components.Walk, components.Fall):
        #     new_cb[c.__name__.lower()] = c
        #
        # # Import the desired command book file
        # module_name = splitext(basename(file))[0]
        # target = '.'.join(['resources', 'command_books', module_name])
        # try:
        #     module = importlib.import_module(target)
        #     module = importlib.reload(module)
        # except ImportError:     # Display errors in the target Command Book
        #     print(' !  Errors during compilation:\n')
        #     for line in traceback.format_exc().split('\n'):
        #         line = line.rstrip()
        #         if line:
        #             print(' ' * 4 + line)
        #     print(f"\n !  Command book '{module_name}' was not loaded")
        #     return
        #
        # # Check if the 'step' function has been implemented
        # step_found = False
        # for name, func in inspect.getmembers(module, inspect.isfunction):
        #     if name.lower() == 'step':
        #         step_found = True
        #         new_step = func
        #
        # # Populate the new command book
        # for name, command in inspect.getmembers(module, inspect.isclass):
        #     new_cb[name.lower()] = command
        #
        # # Check if required commands have been implemented and overridden
        # required_found = True
        # for command in [components.Buff]:
        #     name = command.__name__.lower()
        #     if name not in new_cb:
        #         required_found = False
        #         new_cb[name] = command
        #         print(f" !  Error: Must implement required command '{name}'.")
        #
        # # Look for overridden movement commands
        # movement_found = True
        # for command in (components.Move, components.Adjust):
        #     name = command.__name__.lower()
        #     if name not in new_cb:
        #         movement_found = False
        #         new_cb[name] = command
        #
        # if not step_found and not movement_found:
        #     print(f" !  Error: Must either implement both 'Move' and 'Adjust' commands, "
        #           f"or the function 'step'")
        # if required_found and (step_found or movement_found):
        #     self.module_name = module_name
        #     self.command_book = new_cb
        #     self.buff = new_cb['buff']()
        #     components.step = new_step
        #     config.gui.menu.file.enable_routine_state()
        #     config.gui.view.status.set_cb(basename(file))
        #     config.routine.clear()
        #     print(f" ~  Successfully loaded command book '{module_name}'")
        # else:
        #     print(f" !  Command book '{module_name}' was not loaded")

    def update_submodules(self, force=False):
        """
        Pulls updates from the submodule repositories. If FORCE is True,
        rebuilds submodules by overwriting all local changes.
        """

        utils.print_separator()
        print('[~] Retrieving latest submodules:')
        self.submodules = []
        repo = git.Repo.init()
        with open('.gitmodules', 'r') as file:
            lines = file.readlines()
            i = 0
            while i < len(lines):
                if lines[i].startswith('[') and i < len(lines) - 2:
                    path = lines[i + 1].split('=')[1].strip()
                    url = lines[i + 2].split('=')[1].strip()
                    self.submodules.append(path)
                    try:
                        repo.git.clone(url, path)       # First time loading submodule
                        print(f" -  Initialized submodule '{path}'")
                    except git.exc.GitCommandError:
                        sub_repo = git.Repo(path)
                        if not force:
                            sub_repo.git.stash()        # Save modified content
                        sub_repo.git.fetch('origin', 'main')
                        sub_repo.git.reset('--hard', 'FETCH_HEAD')
                        if not force:
                            try:                # Restore modified content
                                sub_repo.git.checkout('stash', '--', '.')
                                print(f" -  Updated submodule '{path}', restored local changes")
                            except git.exc.GitCommandError:
                                print(f" -  Updated submodule '{path}'")
                        else:
                            print(f" -  Rebuilt submodule '{path}'")
                        sub_repo.git.stash('clear')
                    i += 3
                else:
                    i += 1
