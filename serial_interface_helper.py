import getpass
import platform
import subprocess

import serial

SUDO_PASSWORD = None  # Required to set FTDI latency timer on Linux systems, can be set to a hardcoded password for convenience or left as None to prompt the user via terminal.

def get_serial_port(chip_type="STM", serial_port_number=None):
    """
    Finds the cartpole serial port, or throws exception if not present
    :param chip_type: "ZYNQ" or "STM" depending on which one you use
    :param serial_port_number: Only used if serial port not found using chip type, can be left None, for normal operation
    :returns:  the string name of the COM port
    """

    import sys
    from serial.tools import list_ports
    ports = list(serial.tools.list_ports.comports())

    # Linux-only refinement: hide legacy ttyS* placeholders and list only real USB CDC/serial endpoints.
    if sys.platform.startswith("linux"):
        visible_ports = [p for p in ports if (getattr(p, "device", "") or "").startswith(("/dev/ttyUSB", "/dev/ttyACM"))]
    else:
        visible_ports = ports

    serial_ports_names = []
    print('\nAvailable serial ports:')
    for index, port in enumerate(visible_ports):
        serial_ports_names.append(port.device)
        print(f'{index}: port={port.device}; description={port.description}')
    print()

    if chip_type == "STM":
        expected_descriptions = ['USB Serial']
    elif chip_type == "ZYNQ":
        expected_descriptions = ['Digilent Adept USB Device - Digilent Adept USB Device', 'Digilent Adept USB Device']
    else:
        raise ValueError(f'Unknown chip type: {chip_type}')

    possible_ports = []
    for port in visible_ports:
        if port.description in expected_descriptions:
            possible_ports.append(port.device)

    SERIAL_PORT = None
    if not possible_ports:
        message = f"Searching serial port by its expected descriptions - {expected_descriptions} - not successful."
        if serial_port_number is not None:
            print(message)
        else:
            raise Exception(message)
    else:
        if serial_port_number is None:
            SERIAL_PORT = possible_ports[0]
        elif 0 <= serial_port_number < len(possible_ports):
            SERIAL_PORT = possible_ports[serial_port_number]
        else:
            print(
                f"Requested serial port number {serial_port_number} is out of range. Available ports: {len(possible_ports)}")
            print(f"Using the first available port: {possible_ports[0]}")
            SERIAL_PORT = possible_ports[0]

    if SERIAL_PORT is None and serial_port_number is not None:
        if len(serial_ports_names) == 0:
            print('No serial ports')
        elif 0 <= serial_port_number < len(serial_ports_names):
            print(f"Setting serial port with requested number ({serial_port_number})\n")
            SERIAL_PORT = serial_ports_names[serial_port_number]

    return SERIAL_PORT


def set_ftdi_latency_timer(serial_port_name):
    print('\nSetting FTDI latency timer')
    requested_value = 1  # in ms

    if platform.system() == 'Linux':
        # check for hardcoded sudo password or prompt the user
        if SUDO_PASSWORD:
            password = SUDO_PASSWORD
        else:
            password = getpass.getpass('Enter sudo password: ')

        serial_port = serial_port_name.split('/')[-1]
        ftdi_timer_latency_requested_value = 1
        command_ftdi_timer_latency_set = f"sh -c 'echo {ftdi_timer_latency_requested_value} > /sys/bus/usb-serial/devices/{serial_port}/latency_timer'"
        command_ftdi_timer_latency_check = f'cat /sys/bus/usb-serial/devices/{serial_port}/latency_timer'
        try:
            subprocess.run(command_ftdi_timer_latency_set, shell=True, check=True, capture_output=True, text=True)
        except subprocess.CalledProcessError as e:
            print(e.stderr)
            if "Permission denied" in e.stderr:
                print("Trying with sudo...")
                command_ftdi_timer_latency_set = f"echo {password} | sudo -S {command_ftdi_timer_latency_set}"
                try:
                    subprocess.run(command_ftdi_timer_latency_set, shell=True, check=True, capture_output=True,
                                   text=True)
                except subprocess.CalledProcessError as e:
                    print(e.stderr)

        ftdi_latency_timer_value = subprocess.run(command_ftdi_timer_latency_check, shell=True, capture_output=True,
                                                  text=True).stdout.rstrip()
        print(
            f'FTDI latency timer value (tested only for FTDI with Zybo and with Linux on PC side): {ftdi_latency_timer_value} ms  \n')
