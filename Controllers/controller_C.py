import os
import subprocess
import tempfile
import ctypes
import numpy as np

from SI_Toolkit.computation_library import NumpyLibrary
from Control_Toolkit.Controllers import template_controller

try:
    from SI_Toolkit_ASF.ToolkitCustomization.predictors_customization import STATE_INDICES
except (ModuleNotFoundError, ImportError):
    # Fallback STATE_INDICES if the module is not available
    STATE_INDICES = {
        'position': 0,
        'positionD': 1, 
        'angle': 2,
        'angleD': 3
    }


class controller_C(template_controller):
    _computation_library = NumpyLibrary()
    
    @property
    def controller_name(self):
        return "c"  # Single C controller type

    def configure(self):
        """
        Configure the C controller by compiling the specified C controller code.
        """
        # Get controller configuration
        controller_file = self.config_controller.get("controller_file", "lqr.c")
        firmware_path = self.config_controller.get("firmware_path", "Firmware/Src/General")
        
        # Create temporary directory for compilation
        self.temp_dir = tempfile.mkdtemp(prefix="c_controller_")
        
        # Compile the C controller
        self._compile_c_controller(controller_file, firmware_path)
        
        # Load the compiled library
        self._load_compiled_library()
        
        # Get controller specification
        self._get_controller_spec()
        
        # Initialize controller
        if hasattr(self, 'init_func') and self.init_func:
            self.init_func()
        
        print(f'Configured C controller: {controller_file}')

    def _compile_c_controller(self, controller_file, firmware_path):
        """
        Compile the C controller code into a shared library.
        """
        # Copy controller_api.h
        api_src = os.path.join(firmware_path, "controller_api.h")
        api_dst = os.path.join(self.temp_dir, "controller_api.h")
        if os.path.exists(api_src):
            with open(api_src, 'r') as f:
                content = f.read()
            with open(api_dst, 'w') as f:
                f.write(content)
        
        # Copy the controller files
        controller_src = os.path.join(firmware_path, controller_file)
        controller_dst = os.path.join(self.temp_dir, controller_file)
        
        if not os.path.exists(controller_src):
            raise FileNotFoundError(f"Controller file not found: {controller_src}")
        
        with open(controller_src, 'r') as f:
            content = f.read()
        with open(controller_dst, 'w') as f:
            f.write(content)
        
        # Copy header file if it exists
        header_file = controller_file.replace('.c', '.h')
        header_src = os.path.join(firmware_path, header_file)
        header_dst = os.path.join(self.temp_dir, header_file)
        
        if os.path.exists(header_src):
            with open(header_src, 'r') as f:
                content = f.read()
            with open(header_dst, 'w') as f:
                f.write(content)
        
        # Create hardware bridge stub for PID controller
        if "pid" in controller_file.lower():
            hw_bridge_dst = os.path.join(self.temp_dir, "hardware_bridge.h")
            minimal_hw_bridge = '''
#ifndef HARDWARE_BRIDGE_H
#define HARDWARE_BRIDGE_H

#include <stdint.h>

// Minimal hardware bridge for PC compilation
static inline void enable_irq(void) { /* No-op for PC */ }
static inline void disable_irq(void) { /* No-op for PC */ }
static inline void Message_SendToPC(const unsigned char* data, unsigned int length) { /* No-op for PC */ }
static inline void Message_SendToPC_blocking(const unsigned char* data, unsigned int length) { /* No-op for PC */ }
static inline int Message_GetFromPC(unsigned char* data) { return 0; }

#endif /* HARDWARE_BRIDGE_H */
'''
            with open(hw_bridge_dst, 'w') as f:
                f.write(minimal_hw_bridge)
            
            # Create communication header stub
            comm_dst = os.path.join(self.temp_dir, "communication_with_PC_general.h")
            minimal_comm = '''
#ifndef COMMUNICATION_WITH_PC_GENERAL_H
#define COMMUNICATION_WITH_PC_GENERAL_H

#include <stdint.h>
#include <stdbool.h>

// Minimal communication header for PC compilation
unsigned char crc(const unsigned char * message, unsigned int len);
bool crcIsValid(const unsigned char * buff, unsigned int len, unsigned char crcVal);
void prepare_message_to_PC_config_PID(unsigned char * txBuffer, float position_KP, float position_KI, float position_KD, float angle_KP, float angle_KI, float angle_KD);

#endif /* COMMUNICATION_WITH_PC_GENERAL_H */
'''
            with open(comm_dst, 'w') as f:
                f.write(minimal_comm)
        
        # Create simple wrapper
        wrapper_c = self._create_simple_wrapper(controller_file)
        wrapper_path = os.path.join(self.temp_dir, "wrapper.c")
        with open(wrapper_path, 'w') as f:
            f.write(wrapper_c)
        
        # Compile the shared library
        self._compile_shared_library(wrapper_path, controller_file)

    def _create_simple_wrapper(self, controller_file):
        """
        Create a simple C wrapper that exposes the controller functions.
        """
        # Map common controller files to their ops names
        ops_mapping = {
            "lqr.c": "LQR_Ops",
            "hardware_pid.c": "PID_Ops", 
            "neural_controller_C.c": "NNC_Ops"
        }
        
        controller_name = controller_file.replace('.c', '').upper()
        ops_name = ops_mapping.get(controller_file, f"{controller_name}_Ops")
        
        wrapper_c = f'''
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

// Include controller API
#include "controller_api.h"

// Include the controller
#include "{controller_file.replace('.c', '.h')}"

// Wrapper functions for Python ctypes
#ifdef __cplusplus
extern "C" {{
#endif
    // Initialize the controller
    void controller_init() {{
        if ({ops_name}.init) {{
            {ops_name}.init();
        }}
    }}
    
    // Evaluate the controller
    void controller_evaluate(const float* inputs, float* outputs) {{
        if ({ops_name}.evaluate) {{
            {ops_name}.evaluate(inputs, outputs);
        }}
    }}
    
    // Get controller specification
    void controller_get_spec(int* version, int* n_inputs, int* n_outputs) {{
        if ({ops_name}.spec) {{
            const ControllerSpec* spec = {ops_name}.spec();
            *version = spec->version;
            *n_inputs = spec->n_inputs;
            *n_outputs = spec->n_outputs;
        }}
    }}
    
    // Get input names (returns concatenated string)
    void controller_get_input_names(char* buffer, int buffer_size) {{
        if ({ops_name}.spec) {{
            const ControllerSpec* spec = {ops_name}.spec();
            int pos = 0;
            for (int i = 0; i < spec->n_inputs && pos < buffer_size - 1; i++) {{
                int len = strlen(spec->names[i]);
                if (pos + len < buffer_size - 1) {{
                    strcpy(buffer + pos, spec->names[i]);
                    pos += len;
                    if (i < spec->n_inputs - 1) {{
                        buffer[pos++] = ',';
                    }}
                }}
            }}
            buffer[pos] = '\\0';
        }}
    }}
    
    // Release the controller
    void controller_release() {{
        if ({ops_name}.release) {{
            {ops_name}.release();
        }}
    }}
#ifdef __cplusplus
}}
#endif
'''
        return wrapper_c

    def _compile_shared_library(self, wrapper_path, controller_file):
        """
        Compile the C code into a shared library.
        """
        # Build the compilation command
        cmd = ["gcc", "-shared", "-fPIC", "-o", os.path.join(self.temp_dir, "controller.so")]
        
        # Add wrapper file
        cmd.append(wrapper_path)
        
        # Add controller file
        controller_path = os.path.join(self.temp_dir, controller_file)
        cmd.append(controller_path)
        
        # Add include directories
        cmd.extend(["-I", self.temp_dir])
        
        # Add math library
        cmd.append("-lm")
        
        # Compile
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, cwd=self.temp_dir)
            if result.returncode != 0:
                raise RuntimeError(f"Compilation failed: {result.stderr}")
        except FileNotFoundError:
            raise RuntimeError("gcc compiler not found. Please install gcc.")

    def _load_compiled_library(self):
        """
        Load the compiled shared library using ctypes.
        """
        lib_path = os.path.join(self.temp_dir, "controller.so")
        if not os.path.exists(lib_path):
            raise RuntimeError(f"Compiled library not found: {lib_path}")
        
        self.lib_ctypes = ctypes.CDLL(lib_path)
        
        # Define function signatures
        self.lib_ctypes.controller_init.argtypes = []
        self.lib_ctypes.controller_init.restype = None
        
        self.lib_ctypes.controller_evaluate.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float)]
        self.lib_ctypes.controller_evaluate.restype = None
        
        self.lib_ctypes.controller_get_spec.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int)]
        self.lib_ctypes.controller_get_spec.restype = None
        
        self.lib_ctypes.controller_get_input_names.argtypes = [ctypes.c_char_p, ctypes.c_int]
        self.lib_ctypes.controller_get_input_names.restype = None
        
        self.lib_ctypes.controller_release.argtypes = []
        self.lib_ctypes.controller_release.restype = None

    def _get_controller_spec(self):
        """
        Get the controller specification from the compiled library.
        """
        # Get spec
        version = ctypes.c_int()
        n_inputs = ctypes.c_int()
        n_outputs = ctypes.c_int()
        
        self.lib_ctypes.controller_get_spec(ctypes.byref(version), ctypes.byref(n_inputs), ctypes.byref(n_outputs))
        
        self.spec_version = version.value
        self.n_inputs = n_inputs.value
        self.n_outputs = n_outputs.value
        
        # Get input names
        buffer_size = 1024
        buffer = ctypes.create_string_buffer(buffer_size)
        self.lib_ctypes.controller_get_input_names(buffer, buffer_size)
        
        input_names_str = buffer.value.decode('utf-8')
        self.input_names = input_names_str.split(',') if input_names_str else []
        
        # Create state index mapping
        self._state_idx = dict(STATE_INDICES)

    def step(self, s: np.ndarray, time=None, updated_attributes: "dict[str, TensorType]" = {}):
        """
        Execute one step of the C controller.
        """
        if updated_attributes is None:
            updated_attributes = {}
        
        # Build inputs in the order expected by the C controller
        arr = np.empty(self.n_inputs, dtype=np.float32)
        for i, name in enumerate(self.input_names):
            if name == "time":
                if time is None:
                    raise Exception("Controller input 'time' is required but not provided.")
                else:
                    val = float(time)
                arr[i] = val
                continue

            if name in updated_attributes:
                val = float(updated_attributes[name])
            elif name in self._state_idx:
                val = float(s[..., self._state_idx[name]])
            elif hasattr(self, 'variable_parameters') and hasattr(self.variable_parameters, name):
                val = float(getattr(self.variable_parameters, name))
            else:
                val = 0.0
            arr[i] = val

        # Call the C controller
        inputs_array = (ctypes.c_float * self.n_inputs)(*arr)
        outputs_array = (ctypes.c_float * self.n_outputs)()
        
        self.lib_ctypes.controller_evaluate(inputs_array, outputs_array)
        
        # Convert output to numpy array
        controller_output = np.array([outputs_array[i] for i in range(self.n_outputs)], dtype=np.float32)
        controller_output = controller_output[np.newaxis, np.newaxis, :]
        
        return controller_output

    def controller_reset(self):
        """
        Reset the controller by reinitializing it.
        """
        if hasattr(self, 'lib_ctypes') and self.lib_ctypes:
            self.lib_ctypes.controller_init()

    def __del__(self):
        """
        Cleanup when the controller is destroyed.
        """
        if hasattr(self, 'lib_ctypes') and self.lib_ctypes:
            try:
                self.lib_ctypes.controller_release()
            except:
                pass  # Ignore errors during cleanup
