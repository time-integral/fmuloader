"""
Python ctypes bindings for the FMI 2.0 standard.

This module provides complete Python bindings for loading and interacting with
FMI 2.0 Functional Mock-up Units (FMUs), supporting both Co-Simulation and
Model Exchange interfaces.

Reference: FMI Specification 2.0.5
"""

from __future__ import annotations

import ctypes
import platform
import sys
import tempfile
import zipfile
from ctypes import (
    CDLL,
    CFUNCTYPE,
    POINTER,
    Structure,
    byref,
    c_char_p,
    c_double,
    c_int,
    c_size_t,
    c_uint,
    c_void_p,
)
from enum import IntEnum
from pathlib import Path
from typing import Sequence

# ---------------------------------------------------------------------------
# FMI 2.0 primitive types
# ---------------------------------------------------------------------------
fmi2Component = c_void_p
fmi2ComponentEnvironment = c_void_p
fmi2FMUstate = c_void_p
fmi2ValueReference = c_uint
fmi2Real = c_double
fmi2Integer = c_int
fmi2Boolean = c_int
fmi2Char = ctypes.c_char
fmi2String = c_char_p
fmi2Byte = ctypes.c_char

fmi2True: int = 1
fmi2False: int = 0


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------
class Fmi2Status(IntEnum):
    OK = 0
    WARNING = 1
    DISCARD = 2
    ERROR = 3
    FATAL = 4
    PENDING = 5


class Fmi2Type(IntEnum):
    MODEL_EXCHANGE = 0
    CO_SIMULATION = 1


class Fmi2StatusKind(IntEnum):
    DO_STEP_STATUS = 0
    PENDING_STATUS = 1
    LAST_SUCCESSFUL_TIME = 2
    TERMINATED = 3


# ---------------------------------------------------------------------------
# Callback function types
# ---------------------------------------------------------------------------
# void logger(fmi2ComponentEnvironment, fmi2String instanceName,
#              fmi2Status status, fmi2String category, fmi2String message, ...)
# Note: ctypes CFUNCTYPE does not support variadic arguments, so we define
# the callback with the fixed parameters only.
_fmi2CallbackLogger = CFUNCTYPE(
    None,
    fmi2ComponentEnvironment,
    fmi2String,
    c_int,
    fmi2String,
    fmi2String,
)

# void* allocateMemory(size_t nobj, size_t size)
_fmi2CallbackAllocateMemory = CFUNCTYPE(c_void_p, c_size_t, c_size_t)

# void freeMemory(void* obj)
_fmi2CallbackFreeMemory = CFUNCTYPE(None, c_void_p)

# void stepFinished(fmi2ComponentEnvironment, fmi2Status)
_fmi2StepFinished = CFUNCTYPE(None, fmi2ComponentEnvironment, c_int)


class _Fmi2CallbackFunctions(Structure):
    _fields_ = [
        ("logger", _fmi2CallbackLogger),
        ("allocateMemory", _fmi2CallbackAllocateMemory),
        ("freeMemory", _fmi2CallbackFreeMemory),
        ("stepFinished", _fmi2StepFinished),
        ("componentEnvironment", fmi2ComponentEnvironment),
    ]


class Fmi2EventInfo(Structure):
    _fields_ = [
        ("newDiscreteStatesNeeded", fmi2Boolean),
        ("terminateSimulation", fmi2Boolean),
        ("nominalsOfContinuousStatesChanged", fmi2Boolean),
        ("valuesOfContinuousStatesChanged", fmi2Boolean),
        ("nextEventTimeDefined", fmi2Boolean),
        ("nextEventTime", fmi2Real),
    ]


# ---------------------------------------------------------------------------
# Default callbacks
# ---------------------------------------------------------------------------
def _default_logger(
    _env: object,
    instance_name: bytes | None,
    status: int,
    category: bytes | None,
    message: bytes | None,
) -> None:
    name = instance_name.decode() if instance_name else ""
    cat = category.decode() if category else ""
    msg = message.decode() if message else ""
    status_str = Fmi2Status(status).name
    print(f"[{name}] [{status_str}] [{cat}] {msg}")


def _default_allocate(nobj: int, size: int) -> int:
    return ctypes.cast(
        ctypes.CDLL(None).calloc(nobj, size), c_void_p
    ).value or 0


def _default_free(obj: int) -> None:
    ctypes.CDLL(None).free(obj)


_LOGGER_FUNC = _fmi2CallbackLogger(_default_logger)
_ALLOCATE_FUNC = _fmi2CallbackAllocateMemory(_default_allocate)
_FREE_FUNC = _fmi2CallbackFreeMemory(_default_free)
_STEP_FINISHED_FUNC = _fmi2StepFinished(0)  # NULL


def _make_callbacks(
    use_memory_callbacks: bool = True,
) -> _Fmi2CallbackFunctions:
    """Create the callback struct for fmi2Instantiate."""
    if use_memory_callbacks:
        return _Fmi2CallbackFunctions(
            logger=_LOGGER_FUNC,
            allocateMemory=_ALLOCATE_FUNC,
            freeMemory=_FREE_FUNC,
            stepFinished=_STEP_FINISHED_FUNC,
            componentEnvironment=None,
        )
    # When canNotUseMemoryManagementFunctions=true, pass NULL for alloc/free
    return _Fmi2CallbackFunctions(
        logger=_LOGGER_FUNC,
        allocateMemory=_fmi2CallbackAllocateMemory(0),
        freeMemory=_fmi2CallbackFreeMemory(0),
        stepFinished=_STEP_FINISHED_FUNC,
        componentEnvironment=None,
    )


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
    """Return the FMI 2.0 platform subfolder name, e.g. 'darwin64'.

    The FMI 2.0 standard defines: ``win32``, ``win64``, ``linux32``,
    ``linux64``, ``darwin32``, ``darwin64``.  ARM architectures are
    **not** covered by the standard.  Use the ``binary_dir`` parameter
    of :class:`Fmi2Slave` to specify a custom subfolder name when
    working with non-standard platforms (e.g. ``"aarch64-darwin"``).
    """
    s = platform.system()
    if s == "Darwin":
        return "darwin64"
    if s == "Linux":
        if sys.maxsize > 2**32:
            return "linux64"
        return "linux32"
    if s == "Windows":
        if sys.maxsize > 2**32:
            return "win64"
        return "win32"
    raise RuntimeError(f"Unsupported platform: {s}")


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
            Use this for non-standard platforms such as ``"aarch64-darwin"``.
            When *None*, the standard FMI 2.0 folder name is used.
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


def _path_to_file_uri(p: Path) -> str:
    """Convert an absolute path to a file:/// URI."""
    resolved = p.resolve()
    # On Windows the drive letter must be handled
    if platform.system() == "Windows":
        uri_path = "/" + str(resolved).replace("\\", "/")
    else:
        uri_path = str(resolved)
    return "file://" + uri_path


# ---------------------------------------------------------------------------
# FMI 2.0 error checking
# ---------------------------------------------------------------------------
class Fmi2Error(Exception):
    """Raised when an FMI 2.0 function returns an error status."""

    def __init__(self, func_name: str, status: Fmi2Status) -> None:
        self.func_name = func_name
        self.status = status
        super().__init__(
            f"{func_name} returned {status.name} ({status.value})"
        )


def _check_status(func_name: str, status: int) -> Fmi2Status:
    s = Fmi2Status(status)
    if s in (Fmi2Status.ERROR, Fmi2Status.FATAL):
        raise Fmi2Error(func_name, s)
    return s


# ---------------------------------------------------------------------------
# FMI2 Slave / Instance wrapper
# ---------------------------------------------------------------------------
class Fmi2Slave:
    """Low-level wrapper around an FMI 2.0 shared library instance.

    This class binds all FMI 2.0 C functions via ctypes and provides
    thin Python methods that handle type conversions automatically.

    Args:
        path: Path to an ``.fmu`` archive **or** an already-extracted FMU
            directory that contains ``binaries/``.
        model_identifier: The model identifier – i.e. the shared-library
            file name without extension (e.g. ``"BouncingBall"``).
        binary_dir: Override for the platform subfolder inside
            ``binaries/``.  The FMI 2.0 standard only defines
            ``win32``, ``win64``, ``linux32``, ``linux64``, ``darwin32``
            and ``darwin64``.  For non-standard platforms such as
            Apple Silicon you can pass e.g. ``"aarch64-darwin"``
            to resolve ``binaries/aarch64-darwin/<model_identifier>.dylib``.
            When *None* the standard folder for the current platform is
            tried first, then all subdirectories are scanned as fallback.
        unpack_dir: Where to extract the ``.fmu`` archive.  If *None* a
            temporary directory is used (cleaned up on context-manager
            exit or garbage collection).

    Typical Co-Simulation usage::

        slave = Fmi2Slave("BouncingBall", model_identifier="BouncingBall")

        slave.instantiate("inst1", Fmi2Type.CO_SIMULATION, guid="{...}")
        slave.setup_experiment(start_time=0.0, stop_time=10.0)
        slave.enter_initialization_mode()
        slave.exit_initialization_mode()

        for t in ...:
            slave.do_step(t, step_size)
            values = slave.get_real([vr1, vr2])

        slave.terminate()
        slave.free_instance()

    Typical Model Exchange usage::

        slave = Fmi2Slave("BouncingBall", model_identifier="BouncingBall")

        slave.instantiate("inst1", Fmi2Type.MODEL_EXCHANGE, guid="{...}")
        slave.setup_experiment(start_time=0.0)
        slave.enter_initialization_mode()
        slave.exit_initialization_mode()

        # Initial event iteration
        event_info = slave.new_discrete_states()
        while event_info.newDiscreteStatesNeeded:
            event_info = slave.new_discrete_states()

        slave.enter_continuous_time_mode()

        # Integration loop
        slave.set_time(t)
        derivs = slave.get_derivatives(nx)
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
        self._component: c_void_p | None = None
        self._dll: CDLL | None = None
        self._callbacks: _Fmi2CallbackFunctions | None = None

        # Determine whether we got an .fmu archive or an extracted directory
        if self._path.suffix == ".fmu":
            if unpack_dir is not None:
                self._extract_dir = Path(unpack_dir)
                self._extract_dir.mkdir(parents=True, exist_ok=True)
            else:
                self._tmpdir = tempfile.TemporaryDirectory(
                    prefix="fmuloader_"
                )
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

        # Bind all FMI 2.0 functions
        self._bind_functions()

    # ------------------------------------------------------------------
    # Function binding
    # ------------------------------------------------------------------
    def _bind_functions(self) -> None:
        """Bind all FMI 2.0 C functions from the shared library."""
        dll = self._dll
        assert dll is not None

        # ---- Common functions ----
        self._fmi2GetTypesPlatform = dll.fmi2GetTypesPlatform
        self._fmi2GetTypesPlatform.restype = c_char_p
        self._fmi2GetTypesPlatform.argtypes = []

        self._fmi2GetVersion = dll.fmi2GetVersion
        self._fmi2GetVersion.restype = c_char_p
        self._fmi2GetVersion.argtypes = []

        self._fmi2SetDebugLogging = dll.fmi2SetDebugLogging
        self._fmi2SetDebugLogging.restype = c_int
        self._fmi2SetDebugLogging.argtypes = [
            fmi2Component,
            fmi2Boolean,
            c_size_t,
            POINTER(fmi2String),
        ]

        self._fmi2Instantiate = dll.fmi2Instantiate
        self._fmi2Instantiate.restype = fmi2Component
        self._fmi2Instantiate.argtypes = [
            fmi2String,       # instanceName
            c_int,            # fmuType
            fmi2String,       # fmuGUID
            fmi2String,       # fmuResourceLocation
            POINTER(_Fmi2CallbackFunctions),
            fmi2Boolean,      # visible
            fmi2Boolean,      # loggingOn
        ]

        self._fmi2FreeInstance = dll.fmi2FreeInstance
        self._fmi2FreeInstance.restype = None
        self._fmi2FreeInstance.argtypes = [fmi2Component]

        self._fmi2SetupExperiment = dll.fmi2SetupExperiment
        self._fmi2SetupExperiment.restype = c_int
        self._fmi2SetupExperiment.argtypes = [
            fmi2Component,
            fmi2Boolean,   # toleranceDefined
            fmi2Real,      # tolerance
            fmi2Real,      # startTime
            fmi2Boolean,   # stopTimeDefined
            fmi2Real,      # stopTime
        ]

        self._fmi2EnterInitializationMode = dll.fmi2EnterInitializationMode
        self._fmi2EnterInitializationMode.restype = c_int
        self._fmi2EnterInitializationMode.argtypes = [fmi2Component]

        self._fmi2ExitInitializationMode = dll.fmi2ExitInitializationMode
        self._fmi2ExitInitializationMode.restype = c_int
        self._fmi2ExitInitializationMode.argtypes = [fmi2Component]

        self._fmi2Terminate = dll.fmi2Terminate
        self._fmi2Terminate.restype = c_int
        self._fmi2Terminate.argtypes = [fmi2Component]

        self._fmi2Reset = dll.fmi2Reset
        self._fmi2Reset.restype = c_int
        self._fmi2Reset.argtypes = [fmi2Component]

        # ---- Getters / Setters ----
        self._fmi2GetReal = dll.fmi2GetReal
        self._fmi2GetReal.restype = c_int
        self._fmi2GetReal.argtypes = [
            fmi2Component,
            POINTER(fmi2ValueReference),
            c_size_t,
            POINTER(fmi2Real),
        ]

        self._fmi2GetInteger = dll.fmi2GetInteger
        self._fmi2GetInteger.restype = c_int
        self._fmi2GetInteger.argtypes = [
            fmi2Component,
            POINTER(fmi2ValueReference),
            c_size_t,
            POINTER(fmi2Integer),
        ]

        self._fmi2GetBoolean = dll.fmi2GetBoolean
        self._fmi2GetBoolean.restype = c_int
        self._fmi2GetBoolean.argtypes = [
            fmi2Component,
            POINTER(fmi2ValueReference),
            c_size_t,
            POINTER(fmi2Boolean),
        ]

        self._fmi2GetString = dll.fmi2GetString
        self._fmi2GetString.restype = c_int
        self._fmi2GetString.argtypes = [
            fmi2Component,
            POINTER(fmi2ValueReference),
            c_size_t,
            POINTER(fmi2String),
        ]

        self._fmi2SetReal = dll.fmi2SetReal
        self._fmi2SetReal.restype = c_int
        self._fmi2SetReal.argtypes = [
            fmi2Component,
            POINTER(fmi2ValueReference),
            c_size_t,
            POINTER(fmi2Real),
        ]

        self._fmi2SetInteger = dll.fmi2SetInteger
        self._fmi2SetInteger.restype = c_int
        self._fmi2SetInteger.argtypes = [
            fmi2Component,
            POINTER(fmi2ValueReference),
            c_size_t,
            POINTER(fmi2Integer),
        ]

        self._fmi2SetBoolean = dll.fmi2SetBoolean
        self._fmi2SetBoolean.restype = c_int
        self._fmi2SetBoolean.argtypes = [
            fmi2Component,
            POINTER(fmi2ValueReference),
            c_size_t,
            POINTER(fmi2Boolean),
        ]

        self._fmi2SetString = dll.fmi2SetString
        self._fmi2SetString.restype = c_int
        self._fmi2SetString.argtypes = [
            fmi2Component,
            POINTER(fmi2ValueReference),
            c_size_t,
            POINTER(fmi2String),
        ]

        # ---- FMU state ----
        self._fmi2GetFMUstate = dll.fmi2GetFMUstate
        self._fmi2GetFMUstate.restype = c_int
        self._fmi2GetFMUstate.argtypes = [
            fmi2Component,
            POINTER(fmi2FMUstate),
        ]

        self._fmi2SetFMUstate = dll.fmi2SetFMUstate
        self._fmi2SetFMUstate.restype = c_int
        self._fmi2SetFMUstate.argtypes = [fmi2Component, fmi2FMUstate]

        self._fmi2FreeFMUstate = dll.fmi2FreeFMUstate
        self._fmi2FreeFMUstate.restype = c_int
        self._fmi2FreeFMUstate.argtypes = [
            fmi2Component,
            POINTER(fmi2FMUstate),
        ]

        self._fmi2SerializedFMUstateSize = dll.fmi2SerializedFMUstateSize
        self._fmi2SerializedFMUstateSize.restype = c_int
        self._fmi2SerializedFMUstateSize.argtypes = [
            fmi2Component,
            fmi2FMUstate,
            POINTER(c_size_t),
        ]

        self._fmi2SerializeFMUstate = dll.fmi2SerializeFMUstate
        self._fmi2SerializeFMUstate.restype = c_int
        self._fmi2SerializeFMUstate.argtypes = [
            fmi2Component,
            fmi2FMUstate,
            POINTER(fmi2Byte),
            c_size_t,
        ]

        self._fmi2DeSerializeFMUstate = dll.fmi2DeSerializeFMUstate
        self._fmi2DeSerializeFMUstate.restype = c_int
        self._fmi2DeSerializeFMUstate.argtypes = [
            fmi2Component,
            POINTER(fmi2Byte),
            c_size_t,
            POINTER(fmi2FMUstate),
        ]

        # ---- Directional derivatives ----
        self._fmi2GetDirectionalDerivative = dll.fmi2GetDirectionalDerivative
        self._fmi2GetDirectionalDerivative.restype = c_int
        self._fmi2GetDirectionalDerivative.argtypes = [
            fmi2Component,
            POINTER(fmi2ValueReference),
            c_size_t,
            POINTER(fmi2ValueReference),
            c_size_t,
            POINTER(fmi2Real),
            POINTER(fmi2Real),
        ]

        # ---- Model Exchange functions ----
        self._fmi2EnterEventMode = dll.fmi2EnterEventMode
        self._fmi2EnterEventMode.restype = c_int
        self._fmi2EnterEventMode.argtypes = [fmi2Component]

        self._fmi2NewDiscreteStates = dll.fmi2NewDiscreteStates
        self._fmi2NewDiscreteStates.restype = c_int
        self._fmi2NewDiscreteStates.argtypes = [
            fmi2Component,
            POINTER(Fmi2EventInfo),
        ]

        self._fmi2EnterContinuousTimeMode = dll.fmi2EnterContinuousTimeMode
        self._fmi2EnterContinuousTimeMode.restype = c_int
        self._fmi2EnterContinuousTimeMode.argtypes = [fmi2Component]

        self._fmi2CompletedIntegratorStep = dll.fmi2CompletedIntegratorStep
        self._fmi2CompletedIntegratorStep.restype = c_int
        self._fmi2CompletedIntegratorStep.argtypes = [
            fmi2Component,
            fmi2Boolean,
            POINTER(fmi2Boolean),
            POINTER(fmi2Boolean),
        ]

        self._fmi2SetTime = dll.fmi2SetTime
        self._fmi2SetTime.restype = c_int
        self._fmi2SetTime.argtypes = [fmi2Component, fmi2Real]

        self._fmi2SetContinuousStates = dll.fmi2SetContinuousStates
        self._fmi2SetContinuousStates.restype = c_int
        self._fmi2SetContinuousStates.argtypes = [
            fmi2Component,
            POINTER(fmi2Real),
            c_size_t,
        ]

        self._fmi2GetDerivatives = dll.fmi2GetDerivatives
        self._fmi2GetDerivatives.restype = c_int
        self._fmi2GetDerivatives.argtypes = [
            fmi2Component,
            POINTER(fmi2Real),
            c_size_t,
        ]

        self._fmi2GetEventIndicators = dll.fmi2GetEventIndicators
        self._fmi2GetEventIndicators.restype = c_int
        self._fmi2GetEventIndicators.argtypes = [
            fmi2Component,
            POINTER(fmi2Real),
            c_size_t,
        ]

        self._fmi2GetContinuousStates = dll.fmi2GetContinuousStates
        self._fmi2GetContinuousStates.restype = c_int
        self._fmi2GetContinuousStates.argtypes = [
            fmi2Component,
            POINTER(fmi2Real),
            c_size_t,
        ]

        self._fmi2GetNominalsOfContinuousStates = (
            dll.fmi2GetNominalsOfContinuousStates
        )
        self._fmi2GetNominalsOfContinuousStates.restype = c_int
        self._fmi2GetNominalsOfContinuousStates.argtypes = [
            fmi2Component,
            POINTER(fmi2Real),
            c_size_t,
        ]

        # ---- Co-Simulation functions ----
        self._fmi2SetRealInputDerivatives = dll.fmi2SetRealInputDerivatives
        self._fmi2SetRealInputDerivatives.restype = c_int
        self._fmi2SetRealInputDerivatives.argtypes = [
            fmi2Component,
            POINTER(fmi2ValueReference),
            c_size_t,
            POINTER(fmi2Integer),
            POINTER(fmi2Real),
        ]

        self._fmi2GetRealOutputDerivatives = dll.fmi2GetRealOutputDerivatives
        self._fmi2GetRealOutputDerivatives.restype = c_int
        self._fmi2GetRealOutputDerivatives.argtypes = [
            fmi2Component,
            POINTER(fmi2ValueReference),
            c_size_t,
            POINTER(fmi2Integer),
            POINTER(fmi2Real),
        ]

        self._fmi2DoStep = dll.fmi2DoStep
        self._fmi2DoStep.restype = c_int
        self._fmi2DoStep.argtypes = [
            fmi2Component,
            fmi2Real,       # currentCommunicationPoint
            fmi2Real,       # communicationStepSize
            fmi2Boolean,    # noSetFMUStatePriorToCurrentPoint
        ]

        self._fmi2CancelStep = dll.fmi2CancelStep
        self._fmi2CancelStep.restype = c_int
        self._fmi2CancelStep.argtypes = [fmi2Component]

        self._fmi2GetStatus = dll.fmi2GetStatus
        self._fmi2GetStatus.restype = c_int
        self._fmi2GetStatus.argtypes = [
            fmi2Component,
            c_int,
            POINTER(c_int),
        ]

        self._fmi2GetRealStatus = dll.fmi2GetRealStatus
        self._fmi2GetRealStatus.restype = c_int
        self._fmi2GetRealStatus.argtypes = [
            fmi2Component,
            c_int,
            POINTER(fmi2Real),
        ]

        self._fmi2GetIntegerStatus = dll.fmi2GetIntegerStatus
        self._fmi2GetIntegerStatus.restype = c_int
        self._fmi2GetIntegerStatus.argtypes = [
            fmi2Component,
            c_int,
            POINTER(fmi2Integer),
        ]

        self._fmi2GetBooleanStatus = dll.fmi2GetBooleanStatus
        self._fmi2GetBooleanStatus.restype = c_int
        self._fmi2GetBooleanStatus.argtypes = [
            fmi2Component,
            c_int,
            POINTER(fmi2Boolean),
        ]

        self._fmi2GetStringStatus = dll.fmi2GetStringStatus
        self._fmi2GetStringStatus.restype = c_int
        self._fmi2GetStringStatus.argtypes = [
            fmi2Component,
            c_int,
            POINTER(fmi2String),
        ]

    # ------------------------------------------------------------------
    # Helper to convert Python lists → ctypes arrays
    # ------------------------------------------------------------------
    @staticmethod
    def _vr_array(vrs: Sequence[int]) -> ctypes.Array[c_uint]:
        arr_type = fmi2ValueReference * len(vrs)
        return arr_type(*vrs)

    @staticmethod
    def _real_array(vals: Sequence[float]) -> ctypes.Array[c_double]:
        arr_type = fmi2Real * len(vals)
        return arr_type(*vals)

    @staticmethod
    def _int_array(vals: Sequence[int]) -> ctypes.Array[c_int]:
        arr_type = fmi2Integer * len(vals)
        return arr_type(*vals)

    @staticmethod
    def _bool_array(vals: Sequence[bool | int]) -> ctypes.Array[c_int]:
        arr_type = fmi2Boolean * len(vals)
        return arr_type(*(fmi2True if v else fmi2False for v in vals))

    @staticmethod
    def _string_array(vals: Sequence[str]) -> ctypes.Array[c_char_p]:
        arr_type = fmi2String * len(vals)
        return arr_type(*(v.encode("utf-8") for v in vals))

    # ------------------------------------------------------------------
    # Common functions
    # ------------------------------------------------------------------
    def get_types_platform(self) -> str:
        """Return the FMI types platform string."""
        return self._fmi2GetTypesPlatform().decode()

    def get_version(self) -> str:
        """Return the FMI version string."""
        return self._fmi2GetVersion().decode()

    def set_debug_logging(
        self,
        logging_on: bool,
        categories: Sequence[str] | None = None,
    ) -> Fmi2Status:
        cats: Sequence[str] = categories or []
        n = len(cats)
        if n > 0:
            arr = self._string_array(cats)
            status = self._fmi2SetDebugLogging(
                self._component, fmi2True if logging_on else fmi2False, n, arr
            )
        else:
            status = self._fmi2SetDebugLogging(
                self._component,
                fmi2True if logging_on else fmi2False,
                0,
                None,
            )
        return _check_status("fmi2SetDebugLogging", status)

    def instantiate(
        self,
        instance_name: str,
        fmu_type: Fmi2Type,
        *,
        guid: str,
        resource_location: str | None = None,
        visible: bool = False,
        logging_on: bool = False,
        use_memory_callbacks: bool = True,
    ) -> None:
        """Instantiate the FMU.

        Args:
            instance_name: Name for this FMU instance.
            fmu_type: Co-Simulation or Model Exchange.
            guid: The GUID from the modelDescription.xml.  Must match
                the GUID compiled into the FMU binary.
            resource_location: ``file:///`` URI pointing to the FMU's
                ``resources/`` directory.  When *None* it is derived
                from the extracted FMU path.
            visible: Whether a simulator UI should be shown.
            logging_on: Whether debug logging is initially enabled.
            use_memory_callbacks: When *False* the ``allocateMemory``
                and ``freeMemory`` callback pointers are set to *NULL*
                (for FMUs that declare
                ``canNotUseMemoryManagementFunctions="true"``).
        """
        if resource_location is None:
            resources_dir = self._extract_dir / "resources"
            if not resources_dir.exists():
                resources_dir = self._extract_dir
            resource_location = _path_to_file_uri(resources_dir)

        self._callbacks = _make_callbacks(
            use_memory_callbacks=use_memory_callbacks
        )

        component = self._fmi2Instantiate(
            instance_name.encode("utf-8"),
            int(fmu_type),
            guid.encode("utf-8"),
            resource_location.encode("utf-8"),
            byref(self._callbacks),
            fmi2True if visible else fmi2False,
            fmi2True if logging_on else fmi2False,
        )
        if not component:
            raise RuntimeError(
                f"fmi2Instantiate returned NULL for {instance_name!r}"
            )
        self._component = component

    def free_instance(self) -> None:
        """Free the FMU instance and release resources."""
        if self._component is not None:
            self._fmi2FreeInstance(self._component)
            self._component = None

    def setup_experiment(
        self,
        start_time: float = 0.0,
        stop_time: float | None = None,
        tolerance: float | None = None,
    ) -> Fmi2Status:
        tolerance_defined = fmi2True if tolerance is not None else fmi2False
        tol_val = tolerance if tolerance is not None else 0.0
        stop_defined = fmi2True if stop_time is not None else fmi2False
        stop_val = stop_time if stop_time is not None else 0.0

        status = self._fmi2SetupExperiment(
            self._component,
            tolerance_defined,
            tol_val,
            start_time,
            stop_defined,
            stop_val,
        )
        return _check_status("fmi2SetupExperiment", status)

    def enter_initialization_mode(self) -> Fmi2Status:
        status = self._fmi2EnterInitializationMode(self._component)
        return _check_status("fmi2EnterInitializationMode", status)

    def exit_initialization_mode(self) -> Fmi2Status:
        status = self._fmi2ExitInitializationMode(self._component)
        return _check_status("fmi2ExitInitializationMode", status)

    def terminate(self) -> Fmi2Status:
        status = self._fmi2Terminate(self._component)
        return _check_status("fmi2Terminate", status)

    def reset(self) -> Fmi2Status:
        status = self._fmi2Reset(self._component)
        return _check_status("fmi2Reset", status)

    # ------------------------------------------------------------------
    # Getting variable values
    # ------------------------------------------------------------------
    def get_real(self, vrs: Sequence[int]) -> list[float]:
        n = len(vrs)
        values = (fmi2Real * n)()
        status = self._fmi2GetReal(
            self._component, self._vr_array(vrs), n, values
        )
        _check_status("fmi2GetReal", status)
        return list(values)

    def get_integer(self, vrs: Sequence[int]) -> list[int]:
        n = len(vrs)
        values = (fmi2Integer * n)()
        status = self._fmi2GetInteger(
            self._component, self._vr_array(vrs), n, values
        )
        _check_status("fmi2GetInteger", status)
        return list(values)

    def get_boolean(self, vrs: Sequence[int]) -> list[bool]:
        n = len(vrs)
        values = (fmi2Boolean * n)()
        status = self._fmi2GetBoolean(
            self._component, self._vr_array(vrs), n, values
        )
        _check_status("fmi2GetBoolean", status)
        return [bool(v) for v in values]

    def get_string(self, vrs: Sequence[int]) -> list[str]:
        n = len(vrs)
        values = (fmi2String * n)()
        status = self._fmi2GetString(
            self._component, self._vr_array(vrs), n, values
        )
        _check_status("fmi2GetString", status)
        return [v.decode("utf-8") if v else "" for v in values]

    # ------------------------------------------------------------------
    # Setting variable values
    # ------------------------------------------------------------------
    def set_real(self, vrs: Sequence[int], values: Sequence[float]) -> Fmi2Status:
        n = len(vrs)
        status = self._fmi2SetReal(
            self._component,
            self._vr_array(vrs),
            n,
            self._real_array(values),
        )
        return _check_status("fmi2SetReal", status)

    def set_integer(
        self, vrs: Sequence[int], values: Sequence[int]
    ) -> Fmi2Status:
        n = len(vrs)
        status = self._fmi2SetInteger(
            self._component,
            self._vr_array(vrs),
            n,
            self._int_array(values),
        )
        return _check_status("fmi2SetInteger", status)

    def set_boolean(
        self, vrs: Sequence[int], values: Sequence[bool | int]
    ) -> Fmi2Status:
        n = len(vrs)
        status = self._fmi2SetBoolean(
            self._component,
            self._vr_array(vrs),
            n,
            self._bool_array(values),
        )
        return _check_status("fmi2SetBoolean", status)

    def set_string(
        self, vrs: Sequence[int], values: Sequence[str]
    ) -> Fmi2Status:
        n = len(vrs)
        status = self._fmi2SetString(
            self._component,
            self._vr_array(vrs),
            n,
            self._string_array(values),
        )
        return _check_status("fmi2SetString", status)

    # ------------------------------------------------------------------
    # FMU State
    # ------------------------------------------------------------------
    def get_fmu_state(self) -> c_void_p:
        state = fmi2FMUstate()
        status = self._fmi2GetFMUstate(self._component, byref(state))
        _check_status("fmi2GetFMUstate", status)
        return state

    def set_fmu_state(self, state: c_void_p) -> Fmi2Status:
        status = self._fmi2SetFMUstate(self._component, state)
        return _check_status("fmi2SetFMUstate", status)

    def free_fmu_state(self, state: c_void_p) -> Fmi2Status:
        status = self._fmi2FreeFMUstate(self._component, byref(state))
        return _check_status("fmi2FreeFMUstate", status)

    def serialized_fmu_state_size(self, state: c_void_p) -> int:
        size = c_size_t()
        status = self._fmi2SerializedFMUstateSize(
            self._component, state, byref(size)
        )
        _check_status("fmi2SerializedFMUstateSize", status)
        return size.value

    def serialize_fmu_state(self, state: c_void_p) -> bytes:
        size = self.serialized_fmu_state_size(state)
        buf = (fmi2Byte * size)()
        status = self._fmi2SerializeFMUstate(
            self._component, state, buf, size
        )
        _check_status("fmi2SerializeFMUstate", status)
        return bytes(buf)

    def deserialize_fmu_state(self, data: bytes) -> c_void_p:
        size = len(data)
        buf = (fmi2Byte * size)(*data)
        state = fmi2FMUstate()
        status = self._fmi2DeSerializeFMUstate(
            self._component, buf, size, byref(state)
        )
        _check_status("fmi2DeSerializeFMUstate", status)
        return state

    # ------------------------------------------------------------------
    # Directional derivatives
    # ------------------------------------------------------------------
    def get_directional_derivative(
        self,
        v_unknown_ref: Sequence[int],
        v_known_ref: Sequence[int],
        dv_known: Sequence[float],
    ) -> list[float]:
        n_unknown = len(v_unknown_ref)
        n_known = len(v_known_ref)
        dv_unknown = (fmi2Real * n_unknown)()
        status = self._fmi2GetDirectionalDerivative(
            self._component,
            self._vr_array(v_unknown_ref),
            n_unknown,
            self._vr_array(v_known_ref),
            n_known,
            self._real_array(dv_known),
            dv_unknown,
        )
        _check_status("fmi2GetDirectionalDerivative", status)
        return list(dv_unknown)

    # ------------------------------------------------------------------
    # Model Exchange functions
    # ------------------------------------------------------------------
    def enter_event_mode(self) -> Fmi2Status:
        status = self._fmi2EnterEventMode(self._component)
        return _check_status("fmi2EnterEventMode", status)

    def new_discrete_states(self) -> Fmi2EventInfo:
        event_info = Fmi2EventInfo()
        status = self._fmi2NewDiscreteStates(
            self._component, byref(event_info)
        )
        _check_status("fmi2NewDiscreteStates", status)
        return event_info

    def enter_continuous_time_mode(self) -> Fmi2Status:
        status = self._fmi2EnterContinuousTimeMode(self._component)
        return _check_status("fmi2EnterContinuousTimeMode", status)

    def completed_integrator_step(
        self, no_set_fmu_state_prior: bool = True
    ) -> tuple[bool, bool]:
        """Call fmi2CompletedIntegratorStep.

        Returns:
            (enter_event_mode, terminate_simulation) booleans.
        """
        enter_event = fmi2Boolean(fmi2False)
        terminate = fmi2Boolean(fmi2False)
        status = self._fmi2CompletedIntegratorStep(
            self._component,
            fmi2True if no_set_fmu_state_prior else fmi2False,
            byref(enter_event),
            byref(terminate),
        )
        _check_status("fmi2CompletedIntegratorStep", status)
        return bool(enter_event.value), bool(terminate.value)

    def set_time(self, time: float) -> Fmi2Status:
        status = self._fmi2SetTime(self._component, time)
        return _check_status("fmi2SetTime", status)

    def set_continuous_states(self, states: Sequence[float]) -> Fmi2Status:
        nx = len(states)
        status = self._fmi2SetContinuousStates(
            self._component, self._real_array(states), nx
        )
        return _check_status("fmi2SetContinuousStates", status)

    def get_derivatives(self, nx: int) -> list[float]:
        """Get state derivatives.

        Args:
            nx: Number of continuous states.
        """
        derivatives = (fmi2Real * nx)()
        status = self._fmi2GetDerivatives(self._component, derivatives, nx)
        _check_status("fmi2GetDerivatives", status)
        return list(derivatives)

    def get_event_indicators(self, ni: int) -> list[float]:
        """Get event indicators.

        Args:
            ni: Number of event indicators.
        """
        indicators = (fmi2Real * ni)()
        status = self._fmi2GetEventIndicators(
            self._component, indicators, ni
        )
        _check_status("fmi2GetEventIndicators", status)
        return list(indicators)

    def get_continuous_states(self, nx: int) -> list[float]:
        """Get continuous state values.

        Args:
            nx: Number of continuous states.
        """
        states = (fmi2Real * nx)()
        status = self._fmi2GetContinuousStates(self._component, states, nx)
        _check_status("fmi2GetContinuousStates", status)
        return list(states)

    def get_nominals_of_continuous_states(self, nx: int) -> list[float]:
        """Get nominals of continuous states.

        Args:
            nx: Number of continuous states.
        """
        nominals = (fmi2Real * nx)()
        status = self._fmi2GetNominalsOfContinuousStates(
            self._component, nominals, nx
        )
        _check_status("fmi2GetNominalsOfContinuousStates", status)
        return list(nominals)

    # ------------------------------------------------------------------
    # Co-Simulation functions
    # ------------------------------------------------------------------
    def do_step(
        self,
        current_communication_point: float,
        communication_step_size: float,
        no_set_fmu_state_prior: bool = True,
    ) -> Fmi2Status:
        status = self._fmi2DoStep(
            self._component,
            current_communication_point,
            communication_step_size,
            fmi2True if no_set_fmu_state_prior else fmi2False,
        )
        return _check_status("fmi2DoStep", status)

    def cancel_step(self) -> Fmi2Status:
        status = self._fmi2CancelStep(self._component)
        return _check_status("fmi2CancelStep", status)

    def set_real_input_derivatives(
        self,
        vrs: Sequence[int],
        orders: Sequence[int],
        values: Sequence[float],
    ) -> Fmi2Status:
        n = len(vrs)
        status = self._fmi2SetRealInputDerivatives(
            self._component,
            self._vr_array(vrs),
            n,
            self._int_array(orders),
            self._real_array(values),
        )
        return _check_status("fmi2SetRealInputDerivatives", status)

    def get_real_output_derivatives(
        self,
        vrs: Sequence[int],
        orders: Sequence[int],
    ) -> list[float]:
        n = len(vrs)
        values = (fmi2Real * n)()
        status = self._fmi2GetRealOutputDerivatives(
            self._component,
            self._vr_array(vrs),
            n,
            self._int_array(orders),
            values,
        )
        _check_status("fmi2GetRealOutputDerivatives", status)
        return list(values)

    def get_status(self, kind: Fmi2StatusKind) -> Fmi2Status:
        value = c_int()
        status = self._fmi2GetStatus(
            self._component, int(kind), byref(value)
        )
        _check_status("fmi2GetStatus", status)
        return Fmi2Status(value.value)

    def get_real_status(self, kind: Fmi2StatusKind) -> float:
        value = fmi2Real()
        status = self._fmi2GetRealStatus(
            self._component, int(kind), byref(value)
        )
        _check_status("fmi2GetRealStatus", status)
        return value.value

    def get_integer_status(self, kind: Fmi2StatusKind) -> int:
        value = fmi2Integer()
        status = self._fmi2GetIntegerStatus(
            self._component, int(kind), byref(value)
        )
        _check_status("fmi2GetIntegerStatus", status)
        return value.value

    def get_boolean_status(self, kind: Fmi2StatusKind) -> bool:
        value = fmi2Boolean()
        status = self._fmi2GetBooleanStatus(
            self._component, int(kind), byref(value)
        )
        _check_status("fmi2GetBooleanStatus", status)
        return bool(value.value)

    def get_string_status(self, kind: Fmi2StatusKind) -> str:
        value = fmi2String()
        status = self._fmi2GetStringStatus(
            self._component, int(kind), byref(value)
        )
        _check_status("fmi2GetStringStatus", status)
        return value.value.decode("utf-8") if value.value else ""

    # ------------------------------------------------------------------
    # Context manager support
    # ------------------------------------------------------------------
    def __enter__(self) -> Fmi2Slave:
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
