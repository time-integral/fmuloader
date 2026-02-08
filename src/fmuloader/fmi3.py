"""
Python ctypes bindings for the FMI 3.0 standard.

This module provides complete Python bindings for loading and interacting with
FMI 3.0 Functional Mock-up Units (FMUs), supporting Co-Simulation, Model
Exchange, and Scheduled Execution interfaces.

Reference: FMI Specification 3.0.2
"""

from __future__ import annotations

import ctypes
import platform
import struct
import tempfile
import zipfile
from ctypes import (
    CDLL,
    CFUNCTYPE,
    POINTER,
    byref,
    c_bool,
    c_char,
    c_char_p,
    c_double,
    c_float,
    c_int,
    c_int8,
    c_int16,
    c_int32,
    c_int64,
    c_size_t,
    c_uint8,
    c_uint16,
    c_uint32,
    c_uint64,
    c_void_p,
)
from enum import IntEnum
from pathlib import Path
from typing import Any, Sequence

# ---------------------------------------------------------------------------
# FMI 3.0 primitive types (mirrors fmi3PlatformTypes.h)
# ---------------------------------------------------------------------------
fmi3Instance = c_void_p
fmi3InstanceEnvironment = c_void_p
fmi3FMUState = c_void_p
fmi3ValueReference = c_uint32

fmi3Float32 = c_float
fmi3Float64 = c_double
fmi3Int8 = c_int8
fmi3UInt8 = c_uint8
fmi3Int16 = c_int16
fmi3UInt16 = c_uint16
fmi3Int32 = c_int32
fmi3UInt32 = c_uint32
fmi3Int64 = c_int64
fmi3UInt64 = c_uint64
fmi3Boolean = c_bool
fmi3Char = c_char
fmi3String = c_char_p
fmi3Byte = c_uint8
fmi3Binary = POINTER(c_uint8)
fmi3Clock = c_bool

fmi3True: bool = True
fmi3False: bool = False
fmi3ClockActive: bool = True
fmi3ClockInactive: bool = False


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------
class Fmi3Status(IntEnum):
    OK = 0
    WARNING = 1
    DISCARD = 2
    ERROR = 3
    FATAL = 4


class Fmi3Type(IntEnum):
    MODEL_EXCHANGE = 0
    CO_SIMULATION = 1
    SCHEDULED_EXECUTION = 2


class Fmi3DependencyKind(IntEnum):
    INDEPENDENT = 0
    CONSTANT = 1
    FIXED = 2
    TUNABLE = 3
    DISCRETE = 4
    DEPENDENT = 5


class Fmi3IntervalQualifier(IntEnum):
    INTERVAL_NOT_YET_KNOWN = 0
    INTERVAL_UNCHANGED = 1
    INTERVAL_CHANGED = 2


# ---------------------------------------------------------------------------
# Callback function types
# ---------------------------------------------------------------------------
# void fmi3LogMessageCallback(fmi3InstanceEnvironment, fmi3Status, fmi3String,
#                              fmi3String)
_fmi3LogMessageCallback = CFUNCTYPE(
    None,
    fmi3InstanceEnvironment,
    c_int,  # fmi3Status
    fmi3String,
    fmi3String,
)

# void fmi3ClockUpdateCallback(fmi3InstanceEnvironment)
_fmi3ClockUpdateCallback = CFUNCTYPE(None, fmi3InstanceEnvironment)

# void fmi3IntermediateUpdateCallback(
#     fmi3InstanceEnvironment, fmi3Float64, fmi3Boolean, fmi3Boolean,
#     fmi3Boolean, fmi3Boolean, fmi3Boolean*, fmi3Float64*)
_fmi3IntermediateUpdateCallback = CFUNCTYPE(
    None,
    fmi3InstanceEnvironment,
    fmi3Float64,
    fmi3Boolean,
    fmi3Boolean,
    fmi3Boolean,
    fmi3Boolean,
    POINTER(fmi3Boolean),
    POINTER(fmi3Float64),
)

# void fmi3LockPreemptionCallback(void)
_fmi3LockPreemptionCallback = CFUNCTYPE(None)

# void fmi3UnlockPreemptionCallback(void)
_fmi3UnlockPreemptionCallback = CFUNCTYPE(None)


# ---------------------------------------------------------------------------
# Default callbacks
# ---------------------------------------------------------------------------
def _default_logger(
    _env: object,
    status: int,
    category: bytes | None,
    message: bytes | None,
) -> None:
    cat = category.decode() if category else ""
    msg = message.decode() if message else ""
    status_str = Fmi3Status(status).name
    print(f"[{status_str}] [{cat}] {msg}")


_LOGGER_FUNC = _fmi3LogMessageCallback(_default_logger)


# ---------------------------------------------------------------------------
# Shared library helpers
# ---------------------------------------------------------------------------
def _shared_lib_extension() -> str:
    s = platform.system()
    if s == "Windows":
        return ".dll"
    if s == "Darwin":
        return ".dylib"
    return ".so"


def _platform_folder() -> str:
    """Return the FMI 3.0 platform tuple, e.g. ``'x86_64-darwin'``.

    The FMI 3.0 standard defines platform tuples of the form
    ``<arch>-<sys>``.  Common examples:

    * ``x86_64-darwin``, ``aarch64-darwin``
    * ``x86_64-linux``, ``x86-linux``, ``aarch64-linux``
    * ``x86_64-windows``, ``x86-windows``
    """
    machine = platform.machine().lower()
    s = platform.system()

    # Map Python machine names → FMI 3.0 architecture names
    arch_map: dict[str, str] = {
        "x86_64": "x86_64",
        "amd64": "x86_64",
        "i386": "x86",
        "i686": "x86",
        "x86": "x86",
        "aarch64": "aarch64",
        "arm64": "aarch64",
        "armv7l": "aarch32",
        "armv6l": "aarch32",
    }
    arch = arch_map.get(machine)
    if arch is None:
        # Fallback: use pointer size to guess x86 vs x86_64
        if struct.calcsize("P") * 8 == 64:
            arch = "x86_64"
        else:
            arch = "x86"

    if s == "Darwin":
        return f"{arch}-darwin"
    if s == "Linux":
        return f"{arch}-linux"
    if s == "Windows":
        return f"{arch}-windows"
    raise RuntimeError(f"Unsupported platform: {s} ({machine})")


def _find_binary(
    binaries_dir: Path,
    model_identifier: str,
    binary_dir: str | None = None,
) -> Path:
    """Locate the shared library in the binaries/ directory.

    Args:
        binaries_dir: The ``binaries/`` directory inside an extracted FMU.
        model_identifier: The model identifier (shared lib name without
            extension).
        binary_dir: Optional override for the platform subfolder name.
            When *None*, the standard FMI 3.0 platform tuple is used.
    """
    ext = _shared_lib_extension()
    lib_name = model_identifier + ext

    # Try the user-provided or standard folder
    folder = binary_dir if binary_dir is not None else _platform_folder()
    candidate = binaries_dir / folder / lib_name
    if candidate.exists():
        return candidate

    # Fallback: scan all sub-directories
    for found in binaries_dir.rglob(lib_name):
        return found

    raise FileNotFoundError(
        f"Cannot find shared library {lib_name!r} under {binaries_dir}"
    )


# ---------------------------------------------------------------------------
# FMI 3.0 error checking
# ---------------------------------------------------------------------------
class Fmi3Error(Exception):
    """Raised when an FMI 3.0 function returns an error status."""

    def __init__(self, func_name: str, status: Fmi3Status) -> None:
        self.func_name = func_name
        self.status = status
        super().__init__(f"{func_name} returned {status.name} ({status.value})")


def _check_status(func_name: str, status: int) -> Fmi3Status:
    s = Fmi3Status(status)
    if s in (Fmi3Status.ERROR, Fmi3Status.FATAL):
        raise Fmi3Error(func_name, s)
    return s


# ---------------------------------------------------------------------------
# FMI3 Instance wrapper
# ---------------------------------------------------------------------------
class Fmi3Slave:
    """Low-level wrapper around an FMI 3.0 shared library instance.

    This class binds all FMI 3.0 C functions via ctypes and provides
    thin Python methods that handle type conversions automatically.

    Args:
        path: Path to an ``.fmu`` archive **or** an already-extracted FMU
            directory that contains ``binaries/``.
        model_identifier: The model identifier -- i.e. the shared-library
            file name without extension (e.g. ``"BouncingBall"``).
        binary_dir: Override for the platform subfolder inside
            ``binaries/``.  The FMI 3.0 standard uses platform tuples like
            ``x86_64-darwin``, ``aarch64-linux``, ``x86_64-windows``.
            When *None*, the standard tuple for the current platform is
            tried first, then all subdirectories are scanned as fallback.
        unpack_dir: Where to extract the ``.fmu`` archive.  If *None* a
            temporary directory is used (cleaned up on context-manager
            exit or garbage collection).

    Typical Co-Simulation usage::

        slave = Fmi3Slave("BouncingBall.fmu",
                          model_identifier="BouncingBall")

        slave.instantiate_co_simulation(
            "inst1",
            instantiation_token="{...}",
        )
        slave.enter_initialization_mode(start_time=0.0, stop_time=10.0)
        slave.exit_initialization_mode()

        for t in ...:
            slave.do_step(t, step_size)
            values = slave.get_float64([vr1, vr2])

        slave.terminate()
        slave.free_instance()

    Typical Model Exchange usage::

        slave = Fmi3Slave("BouncingBall.fmu",
                          model_identifier="BouncingBall")

        slave.instantiate_model_exchange(
            "inst1",
            instantiation_token="{...}",
        )
        slave.enter_initialization_mode(start_time=0.0)
        slave.exit_initialization_mode()

        # Initial event iteration
        while True:
            result = slave.update_discrete_states()
            if not result.discrete_states_need_update:
                break

        slave.enter_continuous_time_mode()

        # Integration loop
        slave.set_time(t)
        derivs = slave.get_continuous_state_derivatives(nx)
        ...

        slave.terminate()
        slave.free_instance()
    """

    def __init__(
        self,
        path: str | Path,
        *,
        model_identifier: str,
        binary_dir: str | None = None,
        unpack_dir: str | Path | None = None,
    ) -> None:
        self._path = Path(path)
        self._tmpdir: tempfile.TemporaryDirectory[str] | None = None
        self._instance: c_void_p | None = None
        self._dll: CDLL | None = None

        # Determine whether we got an .fmu archive or an extracted directory
        if self._path.suffix == ".fmu":
            if unpack_dir is not None:
                self._extract_dir = Path(unpack_dir)
                self._extract_dir.mkdir(parents=True, exist_ok=True)
            else:
                self._tmpdir = tempfile.TemporaryDirectory(prefix="fmuloader_")
                self._extract_dir = Path(self._tmpdir.name)
            with zipfile.ZipFile(self._path) as zf:
                zf.extractall(self._extract_dir)
        else:
            # Assume it's already an extracted directory
            self._extract_dir = self._path

        self._model_identifier = model_identifier

        # Load shared library
        binaries_dir = self._extract_dir / "binaries"
        lib_path = _find_binary(binaries_dir, model_identifier, binary_dir)
        self._dll = CDLL(str(lib_path))

        # Bind all FMI 3.0 functions
        self._bind_functions()

    # ------------------------------------------------------------------
    # Function binding
    # ------------------------------------------------------------------
    def _bind_functions(self) -> None:
        """Bind all FMI 3.0 C functions from the shared library."""
        dll = self._dll
        assert dll is not None

        # ---- Common functions ----
        self._fmi3GetVersion = dll.fmi3GetVersion
        self._fmi3GetVersion.restype = c_char_p
        self._fmi3GetVersion.argtypes = []

        self._fmi3SetDebugLogging = dll.fmi3SetDebugLogging
        self._fmi3SetDebugLogging.restype = c_int
        self._fmi3SetDebugLogging.argtypes = [
            fmi3Instance,
            fmi3Boolean,
            c_size_t,
            POINTER(fmi3String),
        ]

        # ---- Instantiation functions ----
        self._fmi3InstantiateModelExchange = dll.fmi3InstantiateModelExchange
        self._fmi3InstantiateModelExchange.restype = fmi3Instance
        self._fmi3InstantiateModelExchange.argtypes = [
            fmi3String,  # instanceName
            fmi3String,  # instantiationToken
            fmi3String,  # resourcePath
            fmi3Boolean,  # visible
            fmi3Boolean,  # loggingOn
            fmi3InstanceEnvironment,  # instanceEnvironment
            _fmi3LogMessageCallback,  # logMessage
        ]

        self._fmi3InstantiateCoSimulation = dll.fmi3InstantiateCoSimulation
        self._fmi3InstantiateCoSimulation.restype = fmi3Instance
        self._fmi3InstantiateCoSimulation.argtypes = [
            fmi3String,  # instanceName
            fmi3String,  # instantiationToken
            fmi3String,  # resourcePath
            fmi3Boolean,  # visible
            fmi3Boolean,  # loggingOn
            fmi3Boolean,  # eventModeUsed
            fmi3Boolean,  # earlyReturnAllowed
            POINTER(fmi3ValueReference),  # requiredIntermediateVariables
            c_size_t,  # nRequiredIntermediateVariables
            fmi3InstanceEnvironment,  # instanceEnvironment
            _fmi3LogMessageCallback,  # logMessage
            _fmi3IntermediateUpdateCallback,  # intermediateUpdate
        ]

        self._fmi3InstantiateScheduledExecution = dll.fmi3InstantiateScheduledExecution
        self._fmi3InstantiateScheduledExecution.restype = fmi3Instance
        self._fmi3InstantiateScheduledExecution.argtypes = [
            fmi3String,  # instanceName
            fmi3String,  # instantiationToken
            fmi3String,  # resourcePath
            fmi3Boolean,  # visible
            fmi3Boolean,  # loggingOn
            fmi3InstanceEnvironment,  # instanceEnvironment
            _fmi3LogMessageCallback,  # logMessage
            _fmi3ClockUpdateCallback,  # clockUpdate
            _fmi3LockPreemptionCallback,  # lockPreemption
            _fmi3UnlockPreemptionCallback,  # unlockPreemption
        ]

        self._fmi3FreeInstance = dll.fmi3FreeInstance
        self._fmi3FreeInstance.restype = None
        self._fmi3FreeInstance.argtypes = [fmi3Instance]

        # ---- Initialization / lifecycle ----
        self._fmi3EnterInitializationMode = dll.fmi3EnterInitializationMode
        self._fmi3EnterInitializationMode.restype = c_int
        self._fmi3EnterInitializationMode.argtypes = [
            fmi3Instance,
            fmi3Boolean,  # toleranceDefined
            fmi3Float64,  # tolerance
            fmi3Float64,  # startTime
            fmi3Boolean,  # stopTimeDefined
            fmi3Float64,  # stopTime
        ]

        self._fmi3ExitInitializationMode = dll.fmi3ExitInitializationMode
        self._fmi3ExitInitializationMode.restype = c_int
        self._fmi3ExitInitializationMode.argtypes = [fmi3Instance]

        self._fmi3EnterEventMode = dll.fmi3EnterEventMode
        self._fmi3EnterEventMode.restype = c_int
        self._fmi3EnterEventMode.argtypes = [fmi3Instance]

        self._fmi3Terminate = dll.fmi3Terminate
        self._fmi3Terminate.restype = c_int
        self._fmi3Terminate.argtypes = [fmi3Instance]

        self._fmi3Reset = dll.fmi3Reset
        self._fmi3Reset.restype = c_int
        self._fmi3Reset.argtypes = [fmi3Instance]

        # ---- Getters ----
        # fmi3GetFloat32
        self._fmi3GetFloat32 = dll.fmi3GetFloat32
        self._fmi3GetFloat32.restype = c_int
        self._fmi3GetFloat32.argtypes = [
            fmi3Instance,
            POINTER(fmi3ValueReference),
            c_size_t,
            POINTER(fmi3Float32),
            c_size_t,
        ]

        # fmi3GetFloat64
        self._fmi3GetFloat64 = dll.fmi3GetFloat64
        self._fmi3GetFloat64.restype = c_int
        self._fmi3GetFloat64.argtypes = [
            fmi3Instance,
            POINTER(fmi3ValueReference),
            c_size_t,
            POINTER(fmi3Float64),
            c_size_t,
        ]

        # fmi3GetInt8
        self._fmi3GetInt8 = dll.fmi3GetInt8
        self._fmi3GetInt8.restype = c_int
        self._fmi3GetInt8.argtypes = [
            fmi3Instance,
            POINTER(fmi3ValueReference),
            c_size_t,
            POINTER(fmi3Int8),
            c_size_t,
        ]

        # fmi3GetUInt8
        self._fmi3GetUInt8 = dll.fmi3GetUInt8
        self._fmi3GetUInt8.restype = c_int
        self._fmi3GetUInt8.argtypes = [
            fmi3Instance,
            POINTER(fmi3ValueReference),
            c_size_t,
            POINTER(fmi3UInt8),
            c_size_t,
        ]

        # fmi3GetInt16
        self._fmi3GetInt16 = dll.fmi3GetInt16
        self._fmi3GetInt16.restype = c_int
        self._fmi3GetInt16.argtypes = [
            fmi3Instance,
            POINTER(fmi3ValueReference),
            c_size_t,
            POINTER(fmi3Int16),
            c_size_t,
        ]

        # fmi3GetUInt16
        self._fmi3GetUInt16 = dll.fmi3GetUInt16
        self._fmi3GetUInt16.restype = c_int
        self._fmi3GetUInt16.argtypes = [
            fmi3Instance,
            POINTER(fmi3ValueReference),
            c_size_t,
            POINTER(fmi3UInt16),
            c_size_t,
        ]

        # fmi3GetInt32
        self._fmi3GetInt32 = dll.fmi3GetInt32
        self._fmi3GetInt32.restype = c_int
        self._fmi3GetInt32.argtypes = [
            fmi3Instance,
            POINTER(fmi3ValueReference),
            c_size_t,
            POINTER(fmi3Int32),
            c_size_t,
        ]

        # fmi3GetUInt32
        self._fmi3GetUInt32 = dll.fmi3GetUInt32
        self._fmi3GetUInt32.restype = c_int
        self._fmi3GetUInt32.argtypes = [
            fmi3Instance,
            POINTER(fmi3ValueReference),
            c_size_t,
            POINTER(fmi3UInt32),
            c_size_t,
        ]

        # fmi3GetInt64
        self._fmi3GetInt64 = dll.fmi3GetInt64
        self._fmi3GetInt64.restype = c_int
        self._fmi3GetInt64.argtypes = [
            fmi3Instance,
            POINTER(fmi3ValueReference),
            c_size_t,
            POINTER(fmi3Int64),
            c_size_t,
        ]

        # fmi3GetUInt64
        self._fmi3GetUInt64 = dll.fmi3GetUInt64
        self._fmi3GetUInt64.restype = c_int
        self._fmi3GetUInt64.argtypes = [
            fmi3Instance,
            POINTER(fmi3ValueReference),
            c_size_t,
            POINTER(fmi3UInt64),
            c_size_t,
        ]

        # fmi3GetBoolean
        self._fmi3GetBoolean = dll.fmi3GetBoolean
        self._fmi3GetBoolean.restype = c_int
        self._fmi3GetBoolean.argtypes = [
            fmi3Instance,
            POINTER(fmi3ValueReference),
            c_size_t,
            POINTER(fmi3Boolean),
            c_size_t,
        ]

        # fmi3GetString
        self._fmi3GetString = dll.fmi3GetString
        self._fmi3GetString.restype = c_int
        self._fmi3GetString.argtypes = [
            fmi3Instance,
            POINTER(fmi3ValueReference),
            c_size_t,
            POINTER(fmi3String),
            c_size_t,
        ]

        # fmi3GetBinary
        self._fmi3GetBinary = dll.fmi3GetBinary
        self._fmi3GetBinary.restype = c_int
        self._fmi3GetBinary.argtypes = [
            fmi3Instance,
            POINTER(fmi3ValueReference),
            c_size_t,
            POINTER(c_size_t),  # valueSizes
            POINTER(fmi3Binary),  # values
            c_size_t,
        ]

        # fmi3GetClock
        self._fmi3GetClock = dll.fmi3GetClock
        self._fmi3GetClock.restype = c_int
        self._fmi3GetClock.argtypes = [
            fmi3Instance,
            POINTER(fmi3ValueReference),
            c_size_t,
            POINTER(fmi3Clock),
        ]

        # ---- Setters ----
        # fmi3SetFloat32
        self._fmi3SetFloat32 = dll.fmi3SetFloat32
        self._fmi3SetFloat32.restype = c_int
        self._fmi3SetFloat32.argtypes = [
            fmi3Instance,
            POINTER(fmi3ValueReference),
            c_size_t,
            POINTER(fmi3Float32),
            c_size_t,
        ]

        # fmi3SetFloat64
        self._fmi3SetFloat64 = dll.fmi3SetFloat64
        self._fmi3SetFloat64.restype = c_int
        self._fmi3SetFloat64.argtypes = [
            fmi3Instance,
            POINTER(fmi3ValueReference),
            c_size_t,
            POINTER(fmi3Float64),
            c_size_t,
        ]

        # fmi3SetInt8
        self._fmi3SetInt8 = dll.fmi3SetInt8
        self._fmi3SetInt8.restype = c_int
        self._fmi3SetInt8.argtypes = [
            fmi3Instance,
            POINTER(fmi3ValueReference),
            c_size_t,
            POINTER(fmi3Int8),
            c_size_t,
        ]

        # fmi3SetUInt8
        self._fmi3SetUInt8 = dll.fmi3SetUInt8
        self._fmi3SetUInt8.restype = c_int
        self._fmi3SetUInt8.argtypes = [
            fmi3Instance,
            POINTER(fmi3ValueReference),
            c_size_t,
            POINTER(fmi3UInt8),
            c_size_t,
        ]

        # fmi3SetInt16
        self._fmi3SetInt16 = dll.fmi3SetInt16
        self._fmi3SetInt16.restype = c_int
        self._fmi3SetInt16.argtypes = [
            fmi3Instance,
            POINTER(fmi3ValueReference),
            c_size_t,
            POINTER(fmi3Int16),
            c_size_t,
        ]

        # fmi3SetUInt16
        self._fmi3SetUInt16 = dll.fmi3SetUInt16
        self._fmi3SetUInt16.restype = c_int
        self._fmi3SetUInt16.argtypes = [
            fmi3Instance,
            POINTER(fmi3ValueReference),
            c_size_t,
            POINTER(fmi3UInt16),
            c_size_t,
        ]

        # fmi3SetInt32
        self._fmi3SetInt32 = dll.fmi3SetInt32
        self._fmi3SetInt32.restype = c_int
        self._fmi3SetInt32.argtypes = [
            fmi3Instance,
            POINTER(fmi3ValueReference),
            c_size_t,
            POINTER(fmi3Int32),
            c_size_t,
        ]

        # fmi3SetUInt32
        self._fmi3SetUInt32 = dll.fmi3SetUInt32
        self._fmi3SetUInt32.restype = c_int
        self._fmi3SetUInt32.argtypes = [
            fmi3Instance,
            POINTER(fmi3ValueReference),
            c_size_t,
            POINTER(fmi3UInt32),
            c_size_t,
        ]

        # fmi3SetInt64
        self._fmi3SetInt64 = dll.fmi3SetInt64
        self._fmi3SetInt64.restype = c_int
        self._fmi3SetInt64.argtypes = [
            fmi3Instance,
            POINTER(fmi3ValueReference),
            c_size_t,
            POINTER(fmi3Int64),
            c_size_t,
        ]

        # fmi3SetUInt64
        self._fmi3SetUInt64 = dll.fmi3SetUInt64
        self._fmi3SetUInt64.restype = c_int
        self._fmi3SetUInt64.argtypes = [
            fmi3Instance,
            POINTER(fmi3ValueReference),
            c_size_t,
            POINTER(fmi3UInt64),
            c_size_t,
        ]

        # fmi3SetBoolean
        self._fmi3SetBoolean = dll.fmi3SetBoolean
        self._fmi3SetBoolean.restype = c_int
        self._fmi3SetBoolean.argtypes = [
            fmi3Instance,
            POINTER(fmi3ValueReference),
            c_size_t,
            POINTER(fmi3Boolean),
            c_size_t,
        ]

        # fmi3SetString
        self._fmi3SetString = dll.fmi3SetString
        self._fmi3SetString.restype = c_int
        self._fmi3SetString.argtypes = [
            fmi3Instance,
            POINTER(fmi3ValueReference),
            c_size_t,
            POINTER(fmi3String),
            c_size_t,
        ]

        # fmi3SetBinary
        self._fmi3SetBinary = dll.fmi3SetBinary
        self._fmi3SetBinary.restype = c_int
        self._fmi3SetBinary.argtypes = [
            fmi3Instance,
            POINTER(fmi3ValueReference),
            c_size_t,
            POINTER(c_size_t),  # valueSizes
            POINTER(fmi3Binary),  # values
            c_size_t,
        ]

        # fmi3SetClock
        self._fmi3SetClock = dll.fmi3SetClock
        self._fmi3SetClock.restype = c_int
        self._fmi3SetClock.argtypes = [
            fmi3Instance,
            POINTER(fmi3ValueReference),
            c_size_t,
            POINTER(fmi3Clock),
        ]

        # ---- Variable dependencies ----
        self._fmi3GetNumberOfVariableDependencies = (
            dll.fmi3GetNumberOfVariableDependencies
        )
        self._fmi3GetNumberOfVariableDependencies.restype = c_int
        self._fmi3GetNumberOfVariableDependencies.argtypes = [
            fmi3Instance,
            fmi3ValueReference,
            POINTER(c_size_t),
        ]

        self._fmi3GetVariableDependencies = dll.fmi3GetVariableDependencies
        self._fmi3GetVariableDependencies.restype = c_int
        self._fmi3GetVariableDependencies.argtypes = [
            fmi3Instance,
            fmi3ValueReference,
            POINTER(c_size_t),  # elementIndicesOfDependent
            POINTER(fmi3ValueReference),  # independents
            POINTER(c_size_t),  # elementIndicesOfIndependents
            POINTER(c_int),  # dependencyKinds
            c_size_t,
        ]

        # ---- FMU state ----
        self._fmi3GetFMUState = dll.fmi3GetFMUState
        self._fmi3GetFMUState.restype = c_int
        self._fmi3GetFMUState.argtypes = [
            fmi3Instance,
            POINTER(fmi3FMUState),
        ]

        self._fmi3SetFMUState = dll.fmi3SetFMUState
        self._fmi3SetFMUState.restype = c_int
        self._fmi3SetFMUState.argtypes = [fmi3Instance, fmi3FMUState]

        self._fmi3FreeFMUState = dll.fmi3FreeFMUState
        self._fmi3FreeFMUState.restype = c_int
        self._fmi3FreeFMUState.argtypes = [
            fmi3Instance,
            POINTER(fmi3FMUState),
        ]

        self._fmi3SerializedFMUStateSize = dll.fmi3SerializedFMUStateSize
        self._fmi3SerializedFMUStateSize.restype = c_int
        self._fmi3SerializedFMUStateSize.argtypes = [
            fmi3Instance,
            fmi3FMUState,
            POINTER(c_size_t),
        ]

        self._fmi3SerializeFMUState = dll.fmi3SerializeFMUState
        self._fmi3SerializeFMUState.restype = c_int
        self._fmi3SerializeFMUState.argtypes = [
            fmi3Instance,
            fmi3FMUState,
            POINTER(fmi3Byte),
            c_size_t,
        ]

        self._fmi3DeserializeFMUState = dll.fmi3DeserializeFMUState
        self._fmi3DeserializeFMUState.restype = c_int
        self._fmi3DeserializeFMUState.argtypes = [
            fmi3Instance,
            POINTER(fmi3Byte),
            c_size_t,
            POINTER(fmi3FMUState),
        ]

        # ---- Directional / adjoint derivatives ----
        self._fmi3GetDirectionalDerivative = dll.fmi3GetDirectionalDerivative
        self._fmi3GetDirectionalDerivative.restype = c_int
        self._fmi3GetDirectionalDerivative.argtypes = [
            fmi3Instance,
            POINTER(fmi3ValueReference),  # unknowns
            c_size_t,
            POINTER(fmi3ValueReference),  # knowns
            c_size_t,
            POINTER(fmi3Float64),  # seed
            c_size_t,
            POINTER(fmi3Float64),  # sensitivity
            c_size_t,
        ]

        self._fmi3GetAdjointDerivative = dll.fmi3GetAdjointDerivative
        self._fmi3GetAdjointDerivative.restype = c_int
        self._fmi3GetAdjointDerivative.argtypes = [
            fmi3Instance,
            POINTER(fmi3ValueReference),  # unknowns
            c_size_t,
            POINTER(fmi3ValueReference),  # knowns
            c_size_t,
            POINTER(fmi3Float64),  # seed
            c_size_t,
            POINTER(fmi3Float64),  # sensitivity
            c_size_t,
        ]

        # ---- Configuration mode ----
        self._fmi3EnterConfigurationMode = dll.fmi3EnterConfigurationMode
        self._fmi3EnterConfigurationMode.restype = c_int
        self._fmi3EnterConfigurationMode.argtypes = [fmi3Instance]

        self._fmi3ExitConfigurationMode = dll.fmi3ExitConfigurationMode
        self._fmi3ExitConfigurationMode.restype = c_int
        self._fmi3ExitConfigurationMode.argtypes = [fmi3Instance]

        # ---- Clock functions ----
        self._fmi3GetIntervalDecimal = dll.fmi3GetIntervalDecimal
        self._fmi3GetIntervalDecimal.restype = c_int
        self._fmi3GetIntervalDecimal.argtypes = [
            fmi3Instance,
            POINTER(fmi3ValueReference),
            c_size_t,
            POINTER(fmi3Float64),
            POINTER(c_int),  # fmi3IntervalQualifier
        ]

        self._fmi3GetIntervalFraction = dll.fmi3GetIntervalFraction
        self._fmi3GetIntervalFraction.restype = c_int
        self._fmi3GetIntervalFraction.argtypes = [
            fmi3Instance,
            POINTER(fmi3ValueReference),
            c_size_t,
            POINTER(fmi3UInt64),
            POINTER(fmi3UInt64),
            POINTER(c_int),  # fmi3IntervalQualifier
        ]

        self._fmi3GetShiftDecimal = dll.fmi3GetShiftDecimal
        self._fmi3GetShiftDecimal.restype = c_int
        self._fmi3GetShiftDecimal.argtypes = [
            fmi3Instance,
            POINTER(fmi3ValueReference),
            c_size_t,
            POINTER(fmi3Float64),
        ]

        self._fmi3GetShiftFraction = dll.fmi3GetShiftFraction
        self._fmi3GetShiftFraction.restype = c_int
        self._fmi3GetShiftFraction.argtypes = [
            fmi3Instance,
            POINTER(fmi3ValueReference),
            c_size_t,
            POINTER(fmi3UInt64),
            POINTER(fmi3UInt64),
        ]

        self._fmi3SetIntervalDecimal = dll.fmi3SetIntervalDecimal
        self._fmi3SetIntervalDecimal.restype = c_int
        self._fmi3SetIntervalDecimal.argtypes = [
            fmi3Instance,
            POINTER(fmi3ValueReference),
            c_size_t,
            POINTER(fmi3Float64),
        ]

        self._fmi3SetIntervalFraction = dll.fmi3SetIntervalFraction
        self._fmi3SetIntervalFraction.restype = c_int
        self._fmi3SetIntervalFraction.argtypes = [
            fmi3Instance,
            POINTER(fmi3ValueReference),
            c_size_t,
            POINTER(fmi3UInt64),
            POINTER(fmi3UInt64),
        ]

        self._fmi3SetShiftDecimal = dll.fmi3SetShiftDecimal
        self._fmi3SetShiftDecimal.restype = c_int
        self._fmi3SetShiftDecimal.argtypes = [
            fmi3Instance,
            POINTER(fmi3ValueReference),
            c_size_t,
            POINTER(fmi3Float64),
        ]

        self._fmi3SetShiftFraction = dll.fmi3SetShiftFraction
        self._fmi3SetShiftFraction.restype = c_int
        self._fmi3SetShiftFraction.argtypes = [
            fmi3Instance,
            POINTER(fmi3ValueReference),
            c_size_t,
            POINTER(fmi3UInt64),
            POINTER(fmi3UInt64),
        ]

        self._fmi3EvaluateDiscreteStates = dll.fmi3EvaluateDiscreteStates
        self._fmi3EvaluateDiscreteStates.restype = c_int
        self._fmi3EvaluateDiscreteStates.argtypes = [fmi3Instance]

        self._fmi3UpdateDiscreteStates = dll.fmi3UpdateDiscreteStates
        self._fmi3UpdateDiscreteStates.restype = c_int
        self._fmi3UpdateDiscreteStates.argtypes = [
            fmi3Instance,
            POINTER(fmi3Boolean),  # discreteStatesNeedUpdate
            POINTER(fmi3Boolean),  # terminateSimulation
            POINTER(fmi3Boolean),  # nominalsOfContinuousStatesChanged
            POINTER(fmi3Boolean),  # valuesOfContinuousStatesChanged
            POINTER(fmi3Boolean),  # nextEventTimeDefined
            POINTER(fmi3Float64),  # nextEventTime
        ]

        # ---- Model Exchange functions ----
        self._fmi3EnterContinuousTimeMode = dll.fmi3EnterContinuousTimeMode
        self._fmi3EnterContinuousTimeMode.restype = c_int
        self._fmi3EnterContinuousTimeMode.argtypes = [fmi3Instance]

        self._fmi3CompletedIntegratorStep = dll.fmi3CompletedIntegratorStep
        self._fmi3CompletedIntegratorStep.restype = c_int
        self._fmi3CompletedIntegratorStep.argtypes = [
            fmi3Instance,
            fmi3Boolean,  # noSetFMUStatePriorToCurrentPoint
            POINTER(fmi3Boolean),  # enterEventMode
            POINTER(fmi3Boolean),  # terminateSimulation
        ]

        self._fmi3SetTime = dll.fmi3SetTime
        self._fmi3SetTime.restype = c_int
        self._fmi3SetTime.argtypes = [fmi3Instance, fmi3Float64]

        self._fmi3SetContinuousStates = dll.fmi3SetContinuousStates
        self._fmi3SetContinuousStates.restype = c_int
        self._fmi3SetContinuousStates.argtypes = [
            fmi3Instance,
            POINTER(fmi3Float64),
            c_size_t,
        ]

        self._fmi3GetContinuousStateDerivatives = dll.fmi3GetContinuousStateDerivatives
        self._fmi3GetContinuousStateDerivatives.restype = c_int
        self._fmi3GetContinuousStateDerivatives.argtypes = [
            fmi3Instance,
            POINTER(fmi3Float64),
            c_size_t,
        ]

        self._fmi3GetEventIndicators = dll.fmi3GetEventIndicators
        self._fmi3GetEventIndicators.restype = c_int
        self._fmi3GetEventIndicators.argtypes = [
            fmi3Instance,
            POINTER(fmi3Float64),
            c_size_t,
        ]

        self._fmi3GetContinuousStates = dll.fmi3GetContinuousStates
        self._fmi3GetContinuousStates.restype = c_int
        self._fmi3GetContinuousStates.argtypes = [
            fmi3Instance,
            POINTER(fmi3Float64),
            c_size_t,
        ]

        self._fmi3GetNominalsOfContinuousStates = dll.fmi3GetNominalsOfContinuousStates
        self._fmi3GetNominalsOfContinuousStates.restype = c_int
        self._fmi3GetNominalsOfContinuousStates.argtypes = [
            fmi3Instance,
            POINTER(fmi3Float64),
            c_size_t,
        ]

        self._fmi3GetNumberOfEventIndicators = dll.fmi3GetNumberOfEventIndicators
        self._fmi3GetNumberOfEventIndicators.restype = c_int
        self._fmi3GetNumberOfEventIndicators.argtypes = [
            fmi3Instance,
            POINTER(c_size_t),
        ]

        self._fmi3GetNumberOfContinuousStates = dll.fmi3GetNumberOfContinuousStates
        self._fmi3GetNumberOfContinuousStates.restype = c_int
        self._fmi3GetNumberOfContinuousStates.argtypes = [
            fmi3Instance,
            POINTER(c_size_t),
        ]

        # ---- Co-Simulation functions ----
        self._fmi3EnterStepMode = dll.fmi3EnterStepMode
        self._fmi3EnterStepMode.restype = c_int
        self._fmi3EnterStepMode.argtypes = [fmi3Instance]

        self._fmi3GetOutputDerivatives = dll.fmi3GetOutputDerivatives
        self._fmi3GetOutputDerivatives.restype = c_int
        self._fmi3GetOutputDerivatives.argtypes = [
            fmi3Instance,
            POINTER(fmi3ValueReference),
            c_size_t,
            POINTER(fmi3Int32),
            POINTER(fmi3Float64),
            c_size_t,
        ]

        self._fmi3DoStep = dll.fmi3DoStep
        self._fmi3DoStep.restype = c_int
        self._fmi3DoStep.argtypes = [
            fmi3Instance,
            fmi3Float64,  # currentCommunicationPoint
            fmi3Float64,  # communicationStepSize
            fmi3Boolean,  # noSetFMUStatePriorToCurrentPoint
            POINTER(fmi3Boolean),  # eventHandlingNeeded
            POINTER(fmi3Boolean),  # terminateSimulation
            POINTER(fmi3Boolean),  # earlyReturn
            POINTER(fmi3Float64),  # lastSuccessfulTime
        ]

        # ---- Scheduled Execution functions ----
        self._fmi3ActivateModelPartition = dll.fmi3ActivateModelPartition
        self._fmi3ActivateModelPartition.restype = c_int
        self._fmi3ActivateModelPartition.argtypes = [
            fmi3Instance,
            fmi3ValueReference,  # clockReference
            fmi3Float64,  # activationTime
        ]

    # ------------------------------------------------------------------
    # Helper to convert Python lists → ctypes arrays
    # ------------------------------------------------------------------
    @staticmethod
    def _vr_array(vrs: Sequence[int]) -> ctypes.Array[c_uint32]:
        arr_type = fmi3ValueReference * len(vrs)
        return arr_type(*vrs)

    @staticmethod
    def _float64_array(vals: Sequence[float]) -> ctypes.Array[c_double]:
        arr_type = fmi3Float64 * len(vals)
        return arr_type(*vals)

    @staticmethod
    def _float32_array(vals: Sequence[float]) -> ctypes.Array[c_float]:
        arr_type = fmi3Float32 * len(vals)
        return arr_type(*vals)

    @staticmethod
    def _int32_array(vals: Sequence[int]) -> ctypes.Array[c_int32]:
        arr_type = fmi3Int32 * len(vals)
        return arr_type(*vals)

    @staticmethod
    def _uint32_array(vals: Sequence[int]) -> ctypes.Array[c_uint32]:
        arr_type = fmi3UInt32 * len(vals)
        return arr_type(*vals)

    @staticmethod
    def _bool_array(vals: Sequence[bool]) -> ctypes.Array[c_bool]:
        arr_type = fmi3Boolean * len(vals)
        return arr_type(*vals)

    @staticmethod
    def _string_array(vals: Sequence[str]) -> ctypes.Array[c_char_p]:
        arr_type = fmi3String * len(vals)
        return arr_type(*(v.encode("utf-8") for v in vals))

    # ------------------------------------------------------------------
    # Common functions
    # ------------------------------------------------------------------
    def get_version(self) -> str:
        """Return the FMI version string (e.g. ``'3.0'``)."""
        return self._fmi3GetVersion().decode()

    def set_debug_logging(
        self,
        logging_on: bool,
        categories: Sequence[str] | None = None,
    ) -> Fmi3Status:
        cats: Sequence[str] = categories or []
        n = len(cats)
        if n > 0:
            arr = self._string_array(cats)
            status = self._fmi3SetDebugLogging(
                self._instance,
                logging_on,
                n,
                arr,
            )
        else:
            status = self._fmi3SetDebugLogging(
                self._instance,
                logging_on,
                0,
                None,
            )
        return _check_status("fmi3SetDebugLogging", status)

    # ------------------------------------------------------------------
    # Instantiation
    # ------------------------------------------------------------------
    def instantiate_model_exchange(
        self,
        instance_name: str,
        *,
        instantiation_token: str,
        resource_path: str | None = None,
        visible: bool = False,
        logging_on: bool = False,
    ) -> None:
        """Instantiate a Model Exchange FMU.

        Args:
            instance_name: Name for this FMU instance.
            instantiation_token: The instantiationToken from
                modelDescription.xml.
            resource_path: Absolute file path to the ``resources/``
                directory (with trailing separator).  Derived
                automatically when *None*.
            visible: Whether a simulator UI should be shown.
            logging_on: Whether debug logging is initially enabled.
        """
        rp = self._resolve_resource_path(resource_path)
        instance = self._fmi3InstantiateModelExchange(
            instance_name.encode("utf-8"),
            instantiation_token.encode("utf-8"),
            rp,
            visible,
            logging_on,
            None,  # instanceEnvironment
            _LOGGER_FUNC,
        )
        if not instance:
            raise RuntimeError(
                f"fmi3InstantiateModelExchange returned NULL for {instance_name!r}"
            )
        self._instance = instance

    def instantiate_co_simulation(
        self,
        instance_name: str,
        *,
        instantiation_token: str,
        resource_path: str | None = None,
        visible: bool = False,
        logging_on: bool = False,
        event_mode_used: bool = False,
        early_return_allowed: bool = False,
        required_intermediate_variables: Sequence[int] | None = None,
        intermediate_update_callback: Any | None = None,
    ) -> None:
        """Instantiate a Co-Simulation FMU.

        Args:
            instance_name: Name for this FMU instance.
            instantiation_token: The instantiationToken from
                modelDescription.xml.
            resource_path: Absolute file path to the ``resources/``
                directory.  Derived automatically when *None*.
            visible: Whether a simulator UI should be shown.
            logging_on: Whether debug logging is initially enabled.
            event_mode_used: Whether the importer will use Event Mode.
            early_return_allowed: Whether early return from
                ``fmi3DoStep`` is allowed.
            required_intermediate_variables: Value references of
                variables that need intermediate access.
            intermediate_update_callback: Optional callback.  When
                *None*, a NULL pointer is passed.
        """
        rp = self._resolve_resource_path(resource_path)

        if required_intermediate_variables:
            n_riv = len(required_intermediate_variables)
            riv_arr = self._vr_array(required_intermediate_variables)
        else:
            n_riv = 0
            riv_arr = None  # type: ignore[assignment]

        iu_cb = (
            _fmi3IntermediateUpdateCallback(intermediate_update_callback)
            if intermediate_update_callback is not None
            else _fmi3IntermediateUpdateCallback(0)
        )

        instance = self._fmi3InstantiateCoSimulation(
            instance_name.encode("utf-8"),
            instantiation_token.encode("utf-8"),
            rp,
            visible,
            logging_on,
            event_mode_used,
            early_return_allowed,
            riv_arr,
            n_riv,
            None,  # instanceEnvironment
            _LOGGER_FUNC,
            iu_cb,
        )
        if not instance:
            raise RuntimeError(
                f"fmi3InstantiateCoSimulation returned NULL for {instance_name!r}"
            )
        self._instance = instance

    def instantiate_scheduled_execution(
        self,
        instance_name: str,
        *,
        instantiation_token: str,
        resource_path: str | None = None,
        visible: bool = False,
        logging_on: bool = False,
        clock_update_callback: Any | None = None,
        lock_preemption_callback: Any | None = None,
        unlock_preemption_callback: Any | None = None,
    ) -> None:
        """Instantiate a Scheduled Execution FMU.

        Args:
            instance_name: Name for this FMU instance.
            instantiation_token: The instantiationToken from
                modelDescription.xml.
            resource_path: Absolute file path to the ``resources/``
                directory.  Derived automatically when *None*.
            visible: Whether a simulator UI should be shown.
            logging_on: Whether debug logging is initially enabled.
            clock_update_callback: Callback for clock updates.
            lock_preemption_callback: Callback to lock preemption.
            unlock_preemption_callback: Callback to unlock preemption.
        """
        rp = self._resolve_resource_path(resource_path)

        cu_cb = (
            _fmi3ClockUpdateCallback(clock_update_callback)
            if clock_update_callback is not None
            else _fmi3ClockUpdateCallback(0)
        )
        lp_cb = (
            _fmi3LockPreemptionCallback(lock_preemption_callback)
            if lock_preemption_callback is not None
            else _fmi3LockPreemptionCallback(0)
        )
        up_cb = (
            _fmi3UnlockPreemptionCallback(unlock_preemption_callback)
            if unlock_preemption_callback is not None
            else _fmi3UnlockPreemptionCallback(0)
        )

        instance = self._fmi3InstantiateScheduledExecution(
            instance_name.encode("utf-8"),
            instantiation_token.encode("utf-8"),
            rp,
            visible,
            logging_on,
            None,  # instanceEnvironment
            _LOGGER_FUNC,
            cu_cb,
            lp_cb,
            up_cb,
        )
        if not instance:
            raise RuntimeError(
                f"fmi3InstantiateScheduledExecution returned NULL for {instance_name!r}"
            )
        self._instance = instance

    def free_instance(self) -> None:
        """Free the FMU instance and release resources."""
        if self._instance is not None:
            self._fmi3FreeInstance(self._instance)
            self._instance = None

    def _resolve_resource_path(self, resource_path: str | None) -> bytes | None:
        """Resolve resource path to an encoded bytes string or None."""
        if resource_path is not None:
            return resource_path.encode("utf-8")
        resources_dir = self._extract_dir / "resources"
        if resources_dir.exists():
            # FMI 3.0 uses absolute file path (not URI), with trailing sep
            return (str(resources_dir.resolve()) + "/").encode("utf-8")
        return None

    # ------------------------------------------------------------------
    # Initialization / lifecycle
    # ------------------------------------------------------------------
    def enter_initialization_mode(
        self,
        start_time: float = 0.0,
        stop_time: float | None = None,
        tolerance: float | None = None,
    ) -> Fmi3Status:
        tolerance_defined = tolerance is not None
        tol_val = tolerance if tolerance is not None else 0.0
        stop_defined = stop_time is not None
        stop_val = stop_time if stop_time is not None else 0.0

        status = self._fmi3EnterInitializationMode(
            self._instance,
            tolerance_defined,
            tol_val,
            start_time,
            stop_defined,
            stop_val,
        )
        return _check_status("fmi3EnterInitializationMode", status)

    def exit_initialization_mode(self) -> Fmi3Status:
        status = self._fmi3ExitInitializationMode(self._instance)
        return _check_status("fmi3ExitInitializationMode", status)

    def enter_event_mode(self) -> Fmi3Status:
        status = self._fmi3EnterEventMode(self._instance)
        return _check_status("fmi3EnterEventMode", status)

    def terminate(self) -> Fmi3Status:
        status = self._fmi3Terminate(self._instance)
        return _check_status("fmi3Terminate", status)

    def reset(self) -> Fmi3Status:
        status = self._fmi3Reset(self._instance)
        return _check_status("fmi3Reset", status)

    # ------------------------------------------------------------------
    # Getting variable values
    # ------------------------------------------------------------------
    def get_float32(self, vrs: Sequence[int]) -> list[float]:
        n = len(vrs)
        values = (fmi3Float32 * n)()
        status = self._fmi3GetFloat32(
            self._instance,
            self._vr_array(vrs),
            n,
            values,
            n,
        )
        _check_status("fmi3GetFloat32", status)
        return list(values)

    def get_float64(self, vrs: Sequence[int]) -> list[float]:
        n = len(vrs)
        values = (fmi3Float64 * n)()
        status = self._fmi3GetFloat64(
            self._instance,
            self._vr_array(vrs),
            n,
            values,
            n,
        )
        _check_status("fmi3GetFloat64", status)
        return list(values)

    def get_int8(self, vrs: Sequence[int]) -> list[int]:
        n = len(vrs)
        values = (fmi3Int8 * n)()
        status = self._fmi3GetInt8(
            self._instance,
            self._vr_array(vrs),
            n,
            values,
            n,
        )
        _check_status("fmi3GetInt8", status)
        return list(values)

    def get_uint8(self, vrs: Sequence[int]) -> list[int]:
        n = len(vrs)
        values = (fmi3UInt8 * n)()
        status = self._fmi3GetUInt8(
            self._instance,
            self._vr_array(vrs),
            n,
            values,
            n,
        )
        _check_status("fmi3GetUInt8", status)
        return list(values)

    def get_int16(self, vrs: Sequence[int]) -> list[int]:
        n = len(vrs)
        values = (fmi3Int16 * n)()
        status = self._fmi3GetInt16(
            self._instance,
            self._vr_array(vrs),
            n,
            values,
            n,
        )
        _check_status("fmi3GetInt16", status)
        return list(values)

    def get_uint16(self, vrs: Sequence[int]) -> list[int]:
        n = len(vrs)
        values = (fmi3UInt16 * n)()
        status = self._fmi3GetUInt16(
            self._instance,
            self._vr_array(vrs),
            n,
            values,
            n,
        )
        _check_status("fmi3GetUInt16", status)
        return list(values)

    def get_int32(self, vrs: Sequence[int]) -> list[int]:
        n = len(vrs)
        values = (fmi3Int32 * n)()
        status = self._fmi3GetInt32(
            self._instance,
            self._vr_array(vrs),
            n,
            values,
            n,
        )
        _check_status("fmi3GetInt32", status)
        return list(values)

    def get_uint32(self, vrs: Sequence[int]) -> list[int]:
        n = len(vrs)
        values = (fmi3UInt32 * n)()
        status = self._fmi3GetUInt32(
            self._instance,
            self._vr_array(vrs),
            n,
            values,
            n,
        )
        _check_status("fmi3GetUInt32", status)
        return list(values)

    def get_int64(self, vrs: Sequence[int]) -> list[int]:
        n = len(vrs)
        values = (fmi3Int64 * n)()
        status = self._fmi3GetInt64(
            self._instance,
            self._vr_array(vrs),
            n,
            values,
            n,
        )
        _check_status("fmi3GetInt64", status)
        return list(values)

    def get_uint64(self, vrs: Sequence[int]) -> list[int]:
        n = len(vrs)
        values = (fmi3UInt64 * n)()
        status = self._fmi3GetUInt64(
            self._instance,
            self._vr_array(vrs),
            n,
            values,
            n,
        )
        _check_status("fmi3GetUInt64", status)
        return list(values)

    def get_boolean(self, vrs: Sequence[int]) -> list[bool]:
        n = len(vrs)
        values = (fmi3Boolean * n)()
        status = self._fmi3GetBoolean(
            self._instance,
            self._vr_array(vrs),
            n,
            values,
            n,
        )
        _check_status("fmi3GetBoolean", status)
        return [bool(v) for v in values]

    def get_string(self, vrs: Sequence[int]) -> list[str]:
        n = len(vrs)
        values = (fmi3String * n)()
        status = self._fmi3GetString(
            self._instance,
            self._vr_array(vrs),
            n,
            values,
            n,
        )
        _check_status("fmi3GetString", status)
        return [v.decode("utf-8") if v else "" for v in values]

    def get_binary(self, vrs: Sequence[int]) -> list[bytes]:
        """Get binary variable values.

        Returns:
            A list of ``bytes`` objects, one per value reference.
        """
        n = len(vrs)
        sizes = (c_size_t * n)()
        values = (fmi3Binary * n)()
        status = self._fmi3GetBinary(
            self._instance,
            self._vr_array(vrs),
            n,
            sizes,
            values,
            n,
        )
        _check_status("fmi3GetBinary", status)
        result = []
        for i in range(n):
            if values[i] and sizes[i]:
                result.append(bytes(values[i][: sizes[i]]))
            else:
                result.append(b"")
        return result

    def get_clock(self, vrs: Sequence[int]) -> list[bool]:
        n = len(vrs)
        values = (fmi3Clock * n)()
        status = self._fmi3GetClock(
            self._instance,
            self._vr_array(vrs),
            n,
            values,
        )
        _check_status("fmi3GetClock", status)
        return [bool(v) for v in values]

    # ------------------------------------------------------------------
    # Setting variable values
    # ------------------------------------------------------------------
    def set_float32(
        self,
        vrs: Sequence[int],
        values: Sequence[float],
    ) -> Fmi3Status:
        n = len(vrs)
        status = self._fmi3SetFloat32(
            self._instance,
            self._vr_array(vrs),
            n,
            self._float32_array(values),
            len(values),
        )
        return _check_status("fmi3SetFloat32", status)

    def set_float64(
        self,
        vrs: Sequence[int],
        values: Sequence[float],
    ) -> Fmi3Status:
        n = len(vrs)
        status = self._fmi3SetFloat64(
            self._instance,
            self._vr_array(vrs),
            n,
            self._float64_array(values),
            len(values),
        )
        return _check_status("fmi3SetFloat64", status)

    def set_int8(
        self,
        vrs: Sequence[int],
        values: Sequence[int],
    ) -> Fmi3Status:
        n = len(vrs)
        arr = (fmi3Int8 * len(values))(*values)
        status = self._fmi3SetInt8(
            self._instance,
            self._vr_array(vrs),
            n,
            arr,
            len(values),
        )
        return _check_status("fmi3SetInt8", status)

    def set_uint8(
        self,
        vrs: Sequence[int],
        values: Sequence[int],
    ) -> Fmi3Status:
        n = len(vrs)
        arr = (fmi3UInt8 * len(values))(*values)
        status = self._fmi3SetUInt8(
            self._instance,
            self._vr_array(vrs),
            n,
            arr,
            len(values),
        )
        return _check_status("fmi3SetUInt8", status)

    def set_int16(
        self,
        vrs: Sequence[int],
        values: Sequence[int],
    ) -> Fmi3Status:
        n = len(vrs)
        arr = (fmi3Int16 * len(values))(*values)
        status = self._fmi3SetInt16(
            self._instance,
            self._vr_array(vrs),
            n,
            arr,
            len(values),
        )
        return _check_status("fmi3SetInt16", status)

    def set_uint16(
        self,
        vrs: Sequence[int],
        values: Sequence[int],
    ) -> Fmi3Status:
        n = len(vrs)
        arr = (fmi3UInt16 * len(values))(*values)
        status = self._fmi3SetUInt16(
            self._instance,
            self._vr_array(vrs),
            n,
            arr,
            len(values),
        )
        return _check_status("fmi3SetUInt16", status)

    def set_int32(
        self,
        vrs: Sequence[int],
        values: Sequence[int],
    ) -> Fmi3Status:
        n = len(vrs)
        status = self._fmi3SetInt32(
            self._instance,
            self._vr_array(vrs),
            n,
            self._int32_array(values),
            len(values),
        )
        return _check_status("fmi3SetInt32", status)

    def set_uint32(
        self,
        vrs: Sequence[int],
        values: Sequence[int],
    ) -> Fmi3Status:
        n = len(vrs)
        status = self._fmi3SetUInt32(
            self._instance,
            self._vr_array(vrs),
            n,
            self._uint32_array(values),
            len(values),
        )
        return _check_status("fmi3SetUInt32", status)

    def set_int64(
        self,
        vrs: Sequence[int],
        values: Sequence[int],
    ) -> Fmi3Status:
        n = len(vrs)
        arr = (fmi3Int64 * len(values))(*values)
        status = self._fmi3SetInt64(
            self._instance,
            self._vr_array(vrs),
            n,
            arr,
            len(values),
        )
        return _check_status("fmi3SetInt64", status)

    def set_uint64(
        self,
        vrs: Sequence[int],
        values: Sequence[int],
    ) -> Fmi3Status:
        n = len(vrs)
        arr = (fmi3UInt64 * len(values))(*values)
        status = self._fmi3SetUInt64(
            self._instance,
            self._vr_array(vrs),
            n,
            arr,
            len(values),
        )
        return _check_status("fmi3SetUInt64", status)

    def set_boolean(
        self,
        vrs: Sequence[int],
        values: Sequence[bool],
    ) -> Fmi3Status:
        n = len(vrs)
        status = self._fmi3SetBoolean(
            self._instance,
            self._vr_array(vrs),
            n,
            self._bool_array(values),
            len(values),
        )
        return _check_status("fmi3SetBoolean", status)

    def set_string(
        self,
        vrs: Sequence[int],
        values: Sequence[str],
    ) -> Fmi3Status:
        n = len(vrs)
        status = self._fmi3SetString(
            self._instance,
            self._vr_array(vrs),
            n,
            self._string_array(values),
            len(values),
        )
        return _check_status("fmi3SetString", status)

    def set_binary(
        self,
        vrs: Sequence[int],
        values: Sequence[bytes],
    ) -> Fmi3Status:
        """Set binary variable values.

        Args:
            vrs: Value references.
            values: Sequence of ``bytes`` objects.
        """
        n = len(vrs)
        n_vals = len(values)
        sizes = (c_size_t * n_vals)(*(len(v) for v in values))
        ptrs = (fmi3Binary * n_vals)()
        for i, v in enumerate(values):
            buf = (c_uint8 * len(v))(*v)
            ptrs[i] = ctypes.cast(buf, fmi3Binary)
        status = self._fmi3SetBinary(
            self._instance,
            self._vr_array(vrs),
            n,
            sizes,
            ptrs,
            n_vals,
        )
        return _check_status("fmi3SetBinary", status)

    def set_clock(
        self,
        vrs: Sequence[int],
        values: Sequence[bool],
    ) -> Fmi3Status:
        n = len(vrs)
        arr = (fmi3Clock * n)(*values)
        status = self._fmi3SetClock(
            self._instance,
            self._vr_array(vrs),
            n,
            arr,
        )
        return _check_status("fmi3SetClock", status)

    # ------------------------------------------------------------------
    # FMU State
    # ------------------------------------------------------------------
    def get_fmu_state(self) -> c_void_p:
        state = fmi3FMUState()
        status = self._fmi3GetFMUState(self._instance, byref(state))
        _check_status("fmi3GetFMUState", status)
        return state

    def set_fmu_state(self, state: c_void_p) -> Fmi3Status:
        status = self._fmi3SetFMUState(self._instance, state)
        return _check_status("fmi3SetFMUState", status)

    def free_fmu_state(self, state: c_void_p) -> Fmi3Status:
        status = self._fmi3FreeFMUState(self._instance, byref(state))
        return _check_status("fmi3FreeFMUState", status)

    def serialized_fmu_state_size(self, state: c_void_p) -> int:
        size = c_size_t()
        status = self._fmi3SerializedFMUStateSize(
            self._instance,
            state,
            byref(size),
        )
        _check_status("fmi3SerializedFMUStateSize", status)
        return size.value

    def serialize_fmu_state(self, state: c_void_p) -> bytes:
        size = self.serialized_fmu_state_size(state)
        buf = (fmi3Byte * size)()
        status = self._fmi3SerializeFMUState(
            self._instance,
            state,
            buf,
            size,
        )
        _check_status("fmi3SerializeFMUState", status)
        return bytes(buf)

    def deserialize_fmu_state(self, data: bytes) -> c_void_p:
        size = len(data)
        buf = (fmi3Byte * size)(*data)
        state = fmi3FMUState()
        status = self._fmi3DeserializeFMUState(
            self._instance,
            buf,
            size,
            byref(state),
        )
        _check_status("fmi3DeserializeFMUState", status)
        return state

    # ------------------------------------------------------------------
    # Directional / adjoint derivatives
    # ------------------------------------------------------------------
    def get_directional_derivative(
        self,
        unknowns: Sequence[int],
        knowns: Sequence[int],
        seed: Sequence[float],
    ) -> list[float]:
        n_unknowns = len(unknowns)
        n_knowns = len(knowns)
        n_seed = len(seed)
        sensitivity = (fmi3Float64 * n_unknowns)()
        status = self._fmi3GetDirectionalDerivative(
            self._instance,
            self._vr_array(unknowns),
            n_unknowns,
            self._vr_array(knowns),
            n_knowns,
            self._float64_array(seed),
            n_seed,
            sensitivity,
            n_unknowns,
        )
        _check_status("fmi3GetDirectionalDerivative", status)
        return list(sensitivity)

    def get_adjoint_derivative(
        self,
        unknowns: Sequence[int],
        knowns: Sequence[int],
        seed: Sequence[float],
    ) -> list[float]:
        n_unknowns = len(unknowns)
        n_knowns = len(knowns)
        n_seed = len(seed)
        sensitivity = (fmi3Float64 * n_knowns)()
        status = self._fmi3GetAdjointDerivative(
            self._instance,
            self._vr_array(unknowns),
            n_unknowns,
            self._vr_array(knowns),
            n_knowns,
            self._float64_array(seed),
            n_seed,
            sensitivity,
            n_knowns,
        )
        _check_status("fmi3GetAdjointDerivative", status)
        return list(sensitivity)

    # ------------------------------------------------------------------
    # Configuration mode
    # ------------------------------------------------------------------
    def enter_configuration_mode(self) -> Fmi3Status:
        status = self._fmi3EnterConfigurationMode(self._instance)
        return _check_status("fmi3EnterConfigurationMode", status)

    def exit_configuration_mode(self) -> Fmi3Status:
        status = self._fmi3ExitConfigurationMode(self._instance)
        return _check_status("fmi3ExitConfigurationMode", status)

    # ------------------------------------------------------------------
    # Clock functions
    # ------------------------------------------------------------------
    def get_interval_decimal(
        self,
        vrs: Sequence[int],
    ) -> tuple[list[float], list[Fmi3IntervalQualifier]]:
        n = len(vrs)
        intervals = (fmi3Float64 * n)()
        qualifiers = (c_int * n)()
        status = self._fmi3GetIntervalDecimal(
            self._instance,
            self._vr_array(vrs),
            n,
            intervals,
            qualifiers,
        )
        _check_status("fmi3GetIntervalDecimal", status)
        return (
            list(intervals),
            [Fmi3IntervalQualifier(q) for q in qualifiers],
        )

    def get_interval_fraction(
        self,
        vrs: Sequence[int],
    ) -> tuple[list[int], list[int], list[Fmi3IntervalQualifier]]:
        n = len(vrs)
        counters = (fmi3UInt64 * n)()
        resolutions = (fmi3UInt64 * n)()
        qualifiers = (c_int * n)()
        status = self._fmi3GetIntervalFraction(
            self._instance,
            self._vr_array(vrs),
            n,
            counters,
            resolutions,
            qualifiers,
        )
        _check_status("fmi3GetIntervalFraction", status)
        return (
            list(counters),
            list(resolutions),
            [Fmi3IntervalQualifier(q) for q in qualifiers],
        )

    def set_interval_decimal(
        self,
        vrs: Sequence[int],
        intervals: Sequence[float],
    ) -> Fmi3Status:
        n = len(vrs)
        status = self._fmi3SetIntervalDecimal(
            self._instance,
            self._vr_array(vrs),
            n,
            self._float64_array(intervals),
        )
        return _check_status("fmi3SetIntervalDecimal", status)

    def get_shift_decimal(self, vrs: Sequence[int]) -> list[float]:
        n = len(vrs)
        shifts = (fmi3Float64 * n)()
        status = self._fmi3GetShiftDecimal(
            self._instance,
            self._vr_array(vrs),
            n,
            shifts,
        )
        _check_status("fmi3GetShiftDecimal", status)
        return list(shifts)

    def set_shift_decimal(
        self,
        vrs: Sequence[int],
        shifts: Sequence[float],
    ) -> Fmi3Status:
        n = len(vrs)
        status = self._fmi3SetShiftDecimal(
            self._instance,
            self._vr_array(vrs),
            n,
            self._float64_array(shifts),
        )
        return _check_status("fmi3SetShiftDecimal", status)

    def evaluate_discrete_states(self) -> Fmi3Status:
        status = self._fmi3EvaluateDiscreteStates(self._instance)
        return _check_status("fmi3EvaluateDiscreteStates", status)

    class UpdateDiscreteStatesResult:
        """Result of :meth:`update_discrete_states`."""

        __slots__ = (
            "discrete_states_need_update",
            "terminate_simulation",
            "nominals_of_continuous_states_changed",
            "values_of_continuous_states_changed",
            "next_event_time_defined",
            "next_event_time",
        )

        def __init__(
            self,
            discrete_states_need_update: bool,
            terminate_simulation: bool,
            nominals_of_continuous_states_changed: bool,
            values_of_continuous_states_changed: bool,
            next_event_time_defined: bool,
            next_event_time: float,
        ) -> None:
            self.discrete_states_need_update = discrete_states_need_update
            self.terminate_simulation = terminate_simulation
            self.nominals_of_continuous_states_changed = (
                nominals_of_continuous_states_changed
            )
            self.values_of_continuous_states_changed = (
                values_of_continuous_states_changed
            )
            self.next_event_time_defined = next_event_time_defined
            self.next_event_time = next_event_time

    def update_discrete_states(self) -> UpdateDiscreteStatesResult:
        """Call ``fmi3UpdateDiscreteStates``.

        Returns:
            An :class:`UpdateDiscreteStatesResult` with the six output
            parameters.
        """
        dsnu = fmi3Boolean(fmi3False)
        ts = fmi3Boolean(fmi3False)
        nocsc = fmi3Boolean(fmi3False)
        vocsc = fmi3Boolean(fmi3False)
        netd = fmi3Boolean(fmi3False)
        net = fmi3Float64(0.0)
        status = self._fmi3UpdateDiscreteStates(
            self._instance,
            byref(dsnu),
            byref(ts),
            byref(nocsc),
            byref(vocsc),
            byref(netd),
            byref(net),
        )
        _check_status("fmi3UpdateDiscreteStates", status)
        return self.UpdateDiscreteStatesResult(
            discrete_states_need_update=bool(dsnu.value),
            terminate_simulation=bool(ts.value),
            nominals_of_continuous_states_changed=bool(nocsc.value),
            values_of_continuous_states_changed=bool(vocsc.value),
            next_event_time_defined=bool(netd.value),
            next_event_time=net.value,
        )

    # ------------------------------------------------------------------
    # Model Exchange functions
    # ------------------------------------------------------------------
    def enter_continuous_time_mode(self) -> Fmi3Status:
        status = self._fmi3EnterContinuousTimeMode(self._instance)
        return _check_status("fmi3EnterContinuousTimeMode", status)

    def completed_integrator_step(
        self,
        no_set_fmu_state_prior: bool = True,
    ) -> tuple[bool, bool]:
        """Call ``fmi3CompletedIntegratorStep``.

        Returns:
            ``(enter_event_mode, terminate_simulation)`` booleans.
        """
        enter_event = fmi3Boolean(fmi3False)
        terminate = fmi3Boolean(fmi3False)
        status = self._fmi3CompletedIntegratorStep(
            self._instance,
            no_set_fmu_state_prior,
            byref(enter_event),
            byref(terminate),
        )
        _check_status("fmi3CompletedIntegratorStep", status)
        return bool(enter_event.value), bool(terminate.value)

    def set_time(self, time: float) -> Fmi3Status:
        status = self._fmi3SetTime(self._instance, time)
        return _check_status("fmi3SetTime", status)

    def set_continuous_states(self, states: Sequence[float]) -> Fmi3Status:
        nx = len(states)
        status = self._fmi3SetContinuousStates(
            self._instance,
            self._float64_array(states),
            nx,
        )
        return _check_status("fmi3SetContinuousStates", status)

    def get_continuous_state_derivatives(self, nx: int) -> list[float]:
        """Get state derivatives.

        Args:
            nx: Number of continuous states.
        """
        derivatives = (fmi3Float64 * nx)()
        status = self._fmi3GetContinuousStateDerivatives(
            self._instance,
            derivatives,
            nx,
        )
        _check_status("fmi3GetContinuousStateDerivatives", status)
        return list(derivatives)

    def get_event_indicators(self, ni: int) -> list[float]:
        """Get event indicators.

        Args:
            ni: Number of event indicators.
        """
        indicators = (fmi3Float64 * ni)()
        status = self._fmi3GetEventIndicators(
            self._instance,
            indicators,
            ni,
        )
        _check_status("fmi3GetEventIndicators", status)
        return list(indicators)

    def get_continuous_states(self, nx: int) -> list[float]:
        """Get continuous state values.

        Args:
            nx: Number of continuous states.
        """
        states = (fmi3Float64 * nx)()
        status = self._fmi3GetContinuousStates(
            self._instance,
            states,
            nx,
        )
        _check_status("fmi3GetContinuousStates", status)
        return list(states)

    def get_nominals_of_continuous_states(self, nx: int) -> list[float]:
        """Get nominals of continuous states.

        Args:
            nx: Number of continuous states.
        """
        nominals = (fmi3Float64 * nx)()
        status = self._fmi3GetNominalsOfContinuousStates(
            self._instance,
            nominals,
            nx,
        )
        _check_status("fmi3GetNominalsOfContinuousStates", status)
        return list(nominals)

    def get_number_of_event_indicators(self) -> int:
        """Query the number of event indicators."""
        n = c_size_t()
        status = self._fmi3GetNumberOfEventIndicators(
            self._instance,
            byref(n),
        )
        _check_status("fmi3GetNumberOfEventIndicators", status)
        return n.value

    def get_number_of_continuous_states(self) -> int:
        """Query the number of continuous states."""
        n = c_size_t()
        status = self._fmi3GetNumberOfContinuousStates(
            self._instance,
            byref(n),
        )
        _check_status("fmi3GetNumberOfContinuousStates", status)
        return n.value

    # ------------------------------------------------------------------
    # Co-Simulation functions
    # ------------------------------------------------------------------
    def enter_step_mode(self) -> Fmi3Status:
        status = self._fmi3EnterStepMode(self._instance)
        return _check_status("fmi3EnterStepMode", status)

    def get_output_derivatives(
        self,
        vrs: Sequence[int],
        orders: Sequence[int],
    ) -> list[float]:
        n = len(vrs)
        values = (fmi3Float64 * n)()
        status = self._fmi3GetOutputDerivatives(
            self._instance,
            self._vr_array(vrs),
            n,
            self._int32_array(orders),
            values,
            n,
        )
        _check_status("fmi3GetOutputDerivatives", status)
        return list(values)

    class DoStepResult:
        """Result of :meth:`do_step`."""

        __slots__ = (
            "status",
            "event_handling_needed",
            "terminate_simulation",
            "early_return",
            "last_successful_time",
        )

        def __init__(
            self,
            status: Fmi3Status,
            event_handling_needed: bool,
            terminate_simulation: bool,
            early_return: bool,
            last_successful_time: float,
        ) -> None:
            self.status = status
            self.event_handling_needed = event_handling_needed
            self.terminate_simulation = terminate_simulation
            self.early_return = early_return
            self.last_successful_time = last_successful_time

    def do_step(
        self,
        current_communication_point: float,
        communication_step_size: float,
        no_set_fmu_state_prior: bool = True,
    ) -> DoStepResult:
        """Advance the Co-Simulation by one step.

        Returns:
            A :class:`DoStepResult` containing the status and the four
            output flags / values.
        """
        event_handling_needed = fmi3Boolean(fmi3False)
        terminate_simulation = fmi3Boolean(fmi3False)
        early_return = fmi3Boolean(fmi3False)
        last_successful_time = fmi3Float64(0.0)

        raw_status = self._fmi3DoStep(
            self._instance,
            current_communication_point,
            communication_step_size,
            no_set_fmu_state_prior,
            byref(event_handling_needed),
            byref(terminate_simulation),
            byref(early_return),
            byref(last_successful_time),
        )
        status = _check_status("fmi3DoStep", raw_status)
        return self.DoStepResult(
            status=status,
            event_handling_needed=bool(event_handling_needed.value),
            terminate_simulation=bool(terminate_simulation.value),
            early_return=bool(early_return.value),
            last_successful_time=last_successful_time.value,
        )

    # ------------------------------------------------------------------
    # Scheduled Execution functions
    # ------------------------------------------------------------------
    def activate_model_partition(
        self,
        clock_reference: int,
        activation_time: float,
    ) -> Fmi3Status:
        status = self._fmi3ActivateModelPartition(
            self._instance,
            clock_reference,
            activation_time,
        )
        return _check_status("fmi3ActivateModelPartition", status)

    # ------------------------------------------------------------------
    # Context manager support
    # ------------------------------------------------------------------
    def __enter__(self) -> Fmi3Slave:
        return self

    def __exit__(self, *_: object) -> None:
        self.free_instance()
        if self._tmpdir is not None:
            self._tmpdir.cleanup()
            self._tmpdir = None

    def __del__(self) -> None:
        if self._tmpdir is not None:
            try:
                self._tmpdir.cleanup()
            except Exception:
                pass
