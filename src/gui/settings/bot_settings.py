import tkinter as tk
from src.gui.interfaces import LabelFrame, Frame
from src.common.interfaces import Configurable

USE_TENSORFLOW_KEY = 'Use TensorFlow'

class BotSettings(LabelFrame):
    def __init__(self, parent, **kwargs):
        super().__init__(parent, 'Settings', **kwargs)

        self.bot_settings = BotParams('settings')
        self.use_tensorflow = tk.BooleanVar(value=self.bot_settings.get(USE_TENSORFLOW_KEY))

        row = Frame(self)
        row.pack(side=tk.TOP, fill='x', expand=True, pady=5, padx=5)
        check = tk.Checkbutton(
            row,
            variable=self.use_tensorflow,
            text=USE_TENSORFLOW_KEY,
            command=self._on_change
        )
        check.pack()

    def _on_change(self):
        self.bot_settings.set(USE_TENSORFLOW_KEY, self.use_tensorflow.get())
        self.bot_settings.save_config()


class BotParams(Configurable):
    DEFAULT_CONFIG = {
        USE_TENSORFLOW_KEY: False
    }

    def get(self, key):
        return self.config[key]

    def set(self, key, value):
        assert key in self.config
        self.config[key] = value