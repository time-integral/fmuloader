# fmuloader ⚙️

A lightweight, **zero-dependency** Python library for loading and calling FMI 2.0
and 3.0 shared-library binaries via ctypes.

[![PyPI version](https://badge.fury.io/py/fmuloader.svg)](https://badge.fury.io/py/fmuloader)

## Installation 📦

Add `fmuloader` to your project with `uv`:

```bash
uv add fmuloader
```

> To install `uv`, see <https://docs.astral.sh/uv/getting-started/installation/>

## How to use 🚀

### FMI 2.0 Co-Simulation

```python
from fmuloader.fmi2 import Fmi2Slave, Fmi2Type

slave = Fmi2Slave("model.fmu", model_identifier="MyModel")

slave.instantiate("instance1", Fmi2Type.CO_SIMULATION, guid="{...}")
slave.setup_experiment(start_time=0.0, stop_time=10.0)
slave.enter_initialization_mode()
slave.exit_initialization_mode()

t, dt = 0.0, 0.01
while t < 10.0:
    slave.do_step(t, dt)
    t += dt

values = slave.get_real([1, 2])
slave.terminate()
slave.free_instance()
```

### FMI 3.0 Co-Simulation

```python
from fmuloader.fmi3 import Fmi3Slave

slave = Fmi3Slave("model.fmu", model_identifier="MyModel")

slave.instantiate_co_simulation("instance1", instantiation_token="{...}")
slave.enter_initialization_mode(start_time=0.0, stop_time=10.0)
slave.exit_initialization_mode()

t, dt = 0.0, 0.01
while t < 10.0:
    result = slave.do_step(t, dt)
    t += dt

values = slave.get_float64([1, 2])
slave.terminate()
slave.free_instance()
```

### FMI 3.0 Model Exchange

```python
from fmuloader.fmi3 import Fmi3Slave

slave = Fmi3Slave("model.fmu", model_identifier="MyModel")

slave.instantiate_model_exchange("instance1", instantiation_token="{...}")
slave.enter_initialization_mode(start_time=0.0)
slave.exit_initialization_mode()

result = slave.update_discrete_states()
while result.discrete_states_need_update:
    result = slave.update_discrete_states()
slave.enter_continuous_time_mode()

nx = slave.get_number_of_continuous_states()
slave.set_time(0.0)
derivs = slave.get_continuous_state_derivatives(nx)

slave.terminate()
slave.free_instance()
```

### FMI 3.0 Scheduled Execution

```python
from fmuloader.fmi3 import Fmi3Slave

slave = Fmi3Slave("model.fmu", model_identifier="MyModel")

slave.instantiate_scheduled_execution("instance1", instantiation_token="{...}")
slave.enter_initialization_mode(start_time=0.0)
slave.exit_initialization_mode()

slave.activate_model_partition(clock_reference=1001, activation_time=0.0)

slave.terminate()
slave.free_instance()
```

## Features ✨

- Load and call FMI **2.0** and **3.0** shared-library binaries via ctypes
- Full Co-Simulation, Model Exchange, and Scheduled Execution support
- All FMI 3.0 data types: `Float32`, `Float64`, `Int8`–`Int64`, `UInt8`–`UInt64`, `Boolean`, `String`, `Binary`, `Clock`
- FMU state management: get, set, serialize, and deserialize
- Directional and adjoint derivatives
- Clock interval and shift functions
- Context manager support for automatic cleanup
- Automatic platform detection (macOS, Linux, Windows; x86, x86_64, aarch64)

## Design philosophy 💡

**fmuloader intentionally separates binary loading from modelDescription.xml parsing.**

Most FMI libraries tightly couple XML parsing with binary invocation, pulling in
heavy dependencies and making it hard to use one without the other. fmuloader
takes a different approach:

- **fmuloader** handles only the **binary loading** — extracting the shared
  library from an `.fmu` archive, binding every C function via ctypes, and
  exposing thin Python wrappers.
- **modelDescription.xml parsing** is left to the user or to a dedicated library
  like [fmureader](https://github.com/time-integral/fmureader).

This means:

- **Zero runtime dependencies** — fmuloader uses only the Python standard library.
- **Bring your own parser** — use fmureader, FMPy, lxml, or anything else to
  read GUIDs, value references, and variable metadata. Then pass them straight
  to fmuloader.
- **Minimal surface area** — each module (`fmi2`, `fmi3`) is a single file you
  can vendor into your own project if needed.

```python
# Example: combine fmureader (parsing) with fmuloader (execution)
import fmureader.fmi3 as reader
from fmuloader.fmi3 import Fmi3Slave

md = reader.read_model_description("model.fmu")

slave = Fmi3Slave("model.fmu", model_identifier=md.co_simulation.model_identifier)
slave.instantiate_co_simulation("inst", instantiation_token=md.instantiation_token)
# ... use md.model_variables to look up value references, then call slave.get_float64() etc.
```

## Related projects 🔗

- [fmureader](https://github.com/time-integral/fmureader) — Lightweight Pydantic-based modelDescription.xml parser for FMI 2.0 and 3.0
- [FMPy](https://github.com/CATIA-Systems/FMPy) — Full-featured FMI library with simulation, GUI, and more

## Licensing 📄

The code in this project is licensed under MIT license.
See the [LICENSE](LICENSE) file for details.
