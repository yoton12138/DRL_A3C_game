import pygame
import threading

class EnvManager:
    _instance = None
    _lock = threading.Lock()
    _initialized = False
    
    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(EnvManager, cls).__new__(cls)
            return cls._instance
    
    def __init__(self):
        if not EnvManager._initialized:
            pygame.init()
            self._display_created = False
            EnvManager._initialized = True
    
    def create_display(self, width, height):
        with self._lock:
            if not self._display_created:
                pygame.display.set_mode((width, height))
                self._display_created = True
    
    def remove_display(self):
        with self._lock:
            if self._display_created:
                pygame.display.quit()
                self._display_created = False
    
    def quit(self):
        with self._lock:
            if EnvManager._initialized:
                pygame.quit()
                EnvManager._initialized = False
                self._display_created = False 