import glob
import logging
import os
from datetime import datetime
from importlib import import_module
from importlib.util import find_spec

import numpy as np

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "0"  # all TF messages

import tensorflow as tf
from numpy.random import SFC64, Generator
from SI_Toolkit.Functions.TF.Compile import Compile

LOGGING_LEVEL = logging.INFO


class CustomFormatter(logging.Formatter):
    """Logging Formatter to add colors and count warning / errors"""

    grey = "\x1b[38;21m"
    yellow = "\x1b[33;21m"
    red = "\x1b[31;21m"
    bold_red = "\x1b[31;1m"
    reset = "\x1b[0m"
    format = (
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s (%(filename)s:%(lineno)d)"
    )

    FORMATS = {
        logging.DEBUG: grey + format + reset,
        logging.INFO: grey + format + reset,
        logging.WARNING: yellow + format + reset,
        logging.ERROR: red + format + reset,
        logging.CRITICAL: bold_red + format + reset,
    }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)


def get_logger(name):
    # logging.basicConfig(stream=sys.stdout, level=logging.INFO)
    logger = logging.getLogger(name)
    logger.setLevel(LOGGING_LEVEL)
    # create console handler
    ch = logging.StreamHandler()
    ch.setFormatter(CustomFormatter())
    logger.addHandler(ch)
    return logger


log = get_logger(__name__)


def create_rng(id: str, seed: str, use_tf: bool = False):
    if seed == None:
        log.info(f"{id}: No random seed specified. Seeding with datetime.")
        seed = int(
            (datetime.now() - datetime(1970, 1, 1)).total_seconds() * 1000.0
        )  # Fully random

    if use_tf:
        return tf.random.Generator.from_seed(seed=seed)
    else:
        return Generator(SFC64(seed=seed))


def import_controller_by_name(controller_full_name: str) -> type:
    """Search for the specified controller name in the following order:
    1) Control_Toolkit_ASF/Controllers/
    2) Control_Toolkit/Controllers/

    :param controller_full_name: The controller to import by full name
    :type controller_full_name: str
    :return: The controller class
    :rtype: type[template_controller]
    """
    controller_relative_paths = (
        glob.glob(f"{os.path.join('Control_Toolkit_ASF', 'Controllers', controller_full_name)}.py")
        + glob.glob(f"{os.path.join('**', 'Control_Toolkit', 'Controllers', controller_full_name)}.py", recursive=True)
    )
    assert len(controller_relative_paths) == 1, f"Controller {controller_full_name} must be in a unique location. {len(controller_relative_paths)} found."
    controller_relative_path = controller_relative_paths[0]

    log.info(f"Importing controller from {controller_relative_path}")
    
    return getattr(import_module(controller_relative_path.replace(".py", "").replace("/", ".").replace("\\\\", ".").replace("\\", ".")), controller_full_name)
    
def get_available_controller_names() -> "list[str]":
    """
    Method returns the list of controllers available in the Control Toolkit or Application Specific Files
    """
    controller_files = (
        glob.glob(f"Control_Toolkit_ASF/Controllers/controller_*.py")
        + glob.glob(f"**/Control_Toolkit/Controllers/controller_*.py", recursive=True)
    )
    controller_names = ['manual-stabilization']
    controller_names.extend(np.sort(
        [os.path.basename(item)[len('controller_'):-len('.py')].replace('_', '-') for item in controller_files]
    ))

    return controller_names


def get_controller(controller_names=None, controller_name=None, controller_idx=None) -> type:
    """
    The method sets a new controller as the current controller.
    The controller may be indicated either by its name
    or by the index on the controller list (see get_available_controller_names method).
    """

    # Check if the proper information was provided: either controller_name or controller_idx
    if (controller_name is None) and (controller_idx is None):
        raise ValueError('You have to specify either controller_name or controller_idx to set a new controller.'
                            'You have specified none of the two.')
    elif (controller_name is not None) and (controller_idx is not None):
        raise ValueError('You have to specify either controller_name or controller_idx to set a new controller.'
                            'You have specified both.')
    else:
        pass
        
    if controller_names is None:
        controller_names = get_available_controller_names()

    # If controller name provided get controller index and vice versa
    if (controller_name is not None):
        try:
            controller_idx = controller_names.index(controller_name)
        except ValueError:
            print('{} is not in list. \n In list are: {}'.format(controller_name, controller_names))
            return None
    else:
        controller_name = controller_names[controller_idx]

    # Load controller
    if controller_name == 'manual-stabilization':
        Controller = None
    else:
        controller_full_name = 'controller_' + controller_name.replace('-', '_')
        Controller = import_controller_by_name(controller_full_name)

    return Controller, controller_name, controller_idx