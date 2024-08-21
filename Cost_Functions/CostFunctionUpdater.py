import os
import atexit
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from yaml import safe_load


class CostFunctionUpdater:
    # Class-level dictionary to keep track of active observers
    active_watchers = {}

    def __init__(self, cost_function, environment_name, cost_function_name):

        self.config_path = os.path.abspath(cost_function.config_path)
        # Verify if the configuration file exists
        if not os.path.isfile(self.config_path):
            raise FileNotFoundError(f"Configuration file not found at path: {self.config_path}")

        # If the path is already being watched, stop the previous watcher
        if self.config_path in CostFunctionUpdater.active_watchers:
            print(f"Path {self.config_path} is already being watched. Stopping the previous watcher.")
            CostFunctionUpdater.active_watchers[self.config_path].stop()

        self.observer = Observer()
        self.handler = ConfigChangeHandler(cost_function, environment_name, cost_function_name)  # Your custom event handler
        self.cost_config_watch = self.observer.schedule(self.handler, self.config_path, recursive=True)
        atexit.register(self.stop)
        self.observer.start()

        # Update the active_watchers dictionary with the new observer
        CostFunctionUpdater.active_watchers[self.config_path] = self

    def stop(self):
        if self.observer is not None and self.observer.is_alive():
            self.observer.stop()
            self.observer.join()
            self.observer = None

        # Remove the path from the active_watchers dictionary
        if self.config_path in CostFunctionUpdater.active_watchers:
            del CostFunctionUpdater.active_watchers[self.config_path]

    def __del__(self):
        self.stop()

    @classmethod
    def stop_all_watchers(cls):
        """Class method to stop all active watchers."""
        for watcher in list(cls.active_watchers.values()):
            watcher.stop()
        cls.active_watchers.clear()


# Handler for configuration changes
class ConfigChangeHandler(FileSystemEventHandler):
    def __init__(self, cost_function, environment_name, cost_function_name):
        self.cost_function = cost_function
        self.config_path = os.path.abspath(cost_function.config_path)

        self.environment_name = environment_name
        self.cost_function_name = cost_function_name

    def on_modified(self, event):
        if event.src_path == self.config_path:
            self.cost_function.config = safe_load(open(self.config_path, 'r'))[self.environment_name][self.cost_function_name]
            self.cost_function.reload_cost_parameters_from_config_flag = True
