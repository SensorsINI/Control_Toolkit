import os
import atexit
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from yaml import safe_load


class CostFunctionUpdater:
    def __init__(self, cost_function, environment_name, cost_function_name):
        self.observer = Observer()
        self.handler = ConfigChangeHandler(cost_function, environment_name, cost_function_name)  # Your custom event handler
        self.observer.schedule(self.handler, cost_function.config_path, recursive=True)
        atexit.register(self.stop)
        self.observer.start()

    def stop(self):
        self.observer.stop()
        self.observer.join()


# Handler for configuration changes
class ConfigChangeHandler(FileSystemEventHandler):
    def __init__(self, cost_function, environment_name, cost_function_name):
        self.cost_function = cost_function
        self.config_path = cost_function.config_path

        self.environment_name = environment_name
        self.cost_function_name = cost_function_name

    def on_modified(self, event):
        if event.src_path == self.config_path:
            self.cost_function.config = safe_load(open(self.config_path, 'r'))[self.environment_name][self.cost_function_name]
            self.cost_function.reload_cost_parameters_from_config_flag = True
