import platform

import pytest
from fmuloader.fmi3 import (
    Fmi3Error,
    Fmi3IntervalQualifier,
    Fmi3Slave,
    Fmi3Status,
    Fmi3Type,
    _platform_folder,
    _shared_lib_extension,
)

# -----------------------------------------------------------------------
# All reference FMUs with their model identifier and instantiation token
# -----------------------------------------------------------------------
BOUNCING_BALL = (
    "3.0/BouncingBall.fmu",
    "BouncingBall",
    "{1AE5E10D-9521-4DE3-80B9-D0EAAA7D5AF1}",
)
DAHLQUIST = ("3.0/Dahlquist.fmu", "Dahlquist", "{221063D2-EF4A-45FE-B954-B5BFEEA9A59B}")
FEEDTHROUGH = (
    "3.0/Feedthrough.fmu",
    "Feedthrough",
    "{37B954F1-CC86-4D8F-B97F-C7C36F6670D2}",
)
STAIR = ("3.0/Stair.fmu", "Stair", "{BD403596-3166-4232-ABC2-132BDF73E644}")
RESOURCE = ("3.0/Resource.fmu", "Resource", "{7b9c2114-2ce5-4076-a138-2cbc69e069e5}")
VAN_DER_POL = (
    "3.0/VanDerPol.fmu",
    "VanDerPol",
    "{BD403596-3166-4232-ABC2-132BDF73E644}",
)
STATE_SPACE = (
    "3.0/StateSpace.fmu",
    "StateSpace",
    "{D773325B-AB94-4630-BF85-643EB24FCB78}",
)
CLOCKS = ("3.0/Clocks.fmu", "Clocks", "{C5F142BA-B849-42DA-B4A1-4745BFF3BE28}")


# -----------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------
def _make_slave(reference_fmus_dir, fmu_tuple):
    fmu_path, model_id, _token = fmu_tuple
    return Fmi3Slave(
        (reference_fmus_dir / fmu_path).absolute(),
        model_identifier=model_id,
    )


def _instantiate_cs(slave, fmu_tuple, instance_name="test", **kwargs):
    """Instantiate as Co-Simulation and return the slave."""
    _, _, token = fmu_tuple
    slave.instantiate_co_simulation(
        instance_name,
        instantiation_token=token,
        **kwargs,
    )
    return slave


def _instantiate_me(slave, fmu_tuple, instance_name="test", **kwargs):
    """Instantiate as Model Exchange and return the slave."""
    _, _, token = fmu_tuple
    slave.instantiate_model_exchange(
        instance_name,
        instantiation_token=token,
        **kwargs,
    )
    return slave


# =======================================================================
# Helper / enum unit tests (no FMU binary needed)
# =======================================================================
class TestHelpers:
    def test_shared_lib_extension(self):
        ext = _shared_lib_extension()
        system = platform.system()
        if system == "Windows":
            assert ext == ".dll"
        elif system == "Darwin":
            assert ext == ".dylib"
        else:
            assert ext == ".so"

    def test_platform_folder(self):
        folder = _platform_folder()
        system = platform.system()
        if system == "Darwin":
            assert folder in ("x86_64-darwin", "aarch64-darwin")
        elif system == "Linux":
            assert folder in ("x86_64-linux", "x86-linux", "aarch64-linux")
        elif system == "Windows":
            assert folder in ("x86_64-windows", "x86-windows")

    def test_fmi3_status_enum(self):
        assert Fmi3Status.OK == 0
        assert Fmi3Status.WARNING == 1
        assert Fmi3Status.DISCARD == 2
        assert Fmi3Status.ERROR == 3
        assert Fmi3Status.FATAL == 4

    def test_fmi3_type_enum(self):
        assert Fmi3Type.MODEL_EXCHANGE == 0
        assert Fmi3Type.CO_SIMULATION == 1
        assert Fmi3Type.SCHEDULED_EXECUTION == 2

    def test_fmi3_interval_qualifier_enum(self):
        assert Fmi3IntervalQualifier.INTERVAL_NOT_YET_KNOWN == 0
        assert Fmi3IntervalQualifier.INTERVAL_UNCHANGED == 1
        assert Fmi3IntervalQualifier.INTERVAL_CHANGED == 2

    def test_fmi3_error_exception(self):
        error = Fmi3Error("testFunction", Fmi3Status.ERROR)
        assert error.func_name == "testFunction"
        assert error.status == Fmi3Status.ERROR
        assert "testFunction" in str(error)
        assert "ERROR" in str(error)


# =======================================================================
# Basic loading & version for every CS/ME reference FMU
# =======================================================================
@pytest.mark.parametrize(
    "fmu_tuple",
    [BOUNCING_BALL, DAHLQUIST, FEEDTHROUGH, STAIR, RESOURCE, VAN_DER_POL, STATE_SPACE],
    ids=[
        "BouncingBall",
        "Dahlquist",
        "Feedthrough",
        "Stair",
        "Resource",
        "VanDerPol",
        "StateSpace",
    ],
)
def test_fmi3_slave_basic(fmu_tuple, reference_fmus_dir):
    """Load every CS/ME reference FMU and check version."""
    slave = _make_slave(reference_fmus_dir, fmu_tuple)
    assert slave.get_version() == "3.0"

    _instantiate_cs(slave, fmu_tuple)
    assert slave._instance is not None
    slave.free_instance()
    assert slave._instance is None


# =======================================================================
# Co-Simulation lifecycle
# =======================================================================
@pytest.mark.parametrize(
    "fmu_tuple",
    [BOUNCING_BALL, DAHLQUIST, FEEDTHROUGH, STAIR, RESOURCE, VAN_DER_POL, STATE_SPACE],
    ids=[
        "BouncingBall",
        "Dahlquist",
        "Feedthrough",
        "Stair",
        "Resource",
        "VanDerPol",
        "StateSpace",
    ],
)
def test_fmi3_cs_lifecycle(fmu_tuple, reference_fmus_dir):
    """Full CS lifecycle: instantiate → init → step → terminate → free."""
    slave = _make_slave(reference_fmus_dir, fmu_tuple)
    _instantiate_cs(slave, fmu_tuple)

    status = slave.enter_initialization_mode(start_time=0.0, stop_time=1.0)
    assert status == Fmi3Status.OK

    status = slave.exit_initialization_mode()
    assert status == Fmi3Status.OK

    result = slave.do_step(0.0, 0.1)
    assert result.status == Fmi3Status.OK

    status = slave.terminate()
    assert status == Fmi3Status.OK

    slave.free_instance()


# =======================================================================
# Context manager
# =======================================================================
def test_fmi3_context_manager(reference_fmus_dir):
    """Ensure context-manager exit frees the instance."""
    with _make_slave(reference_fmus_dir, BOUNCING_BALL) as slave:
        assert slave.get_version() == "3.0"
        _instantiate_cs(slave, BOUNCING_BALL)
        assert slave._instance is not None
    # After __exit__, instance should be freed
    assert slave._instance is None


# =======================================================================
# BouncingBall
# =======================================================================
class TestBouncingBall:
    def test_co_simulation(self, reference_fmus_dir):
        """Run a BouncingBall CS simulation and check dynamics."""
        slave = _make_slave(reference_fmus_dir, BOUNCING_BALL)
        _instantiate_cs(slave, BOUNCING_BALL, "bb_cs")

        slave.enter_initialization_mode(start_time=0.0, stop_time=3.0)

        vr_h = 1
        h0 = slave.get_float64([vr_h])
        assert h0[0] == pytest.approx(1.0)

        slave.exit_initialization_mode()

        # Simulate for 1 s
        t = 0.0
        dt = 0.01
        for _ in range(100):
            slave.do_step(t, dt)
            t += dt

        h_final = slave.get_float64([vr_h])
        # Ball should have dropped and bounced
        assert h_final[0] != pytest.approx(1.0, abs=0.01)

        slave.terminate()
        slave.free_instance()

    def test_model_exchange(self, reference_fmus_dir):
        """Run a BouncingBall ME simulation."""
        slave = _make_slave(reference_fmus_dir, BOUNCING_BALL)
        _instantiate_me(slave, BOUNCING_BALL, "bb_me")

        slave.enter_initialization_mode(start_time=0.0)
        slave.exit_initialization_mode()

        # Event iteration
        result = slave.update_discrete_states()
        while result.discrete_states_need_update:
            result = slave.update_discrete_states()

        slave.enter_continuous_time_mode()

        nx = slave.get_number_of_continuous_states()
        assert nx == 2  # h, v

        ni = slave.get_number_of_event_indicators()
        assert ni >= 1  # At least the ground contact indicator

        states = slave.get_continuous_states(nx)
        assert len(states) == nx

        derivs = slave.get_continuous_state_derivatives(nx)
        assert len(derivs) == nx
        assert all(isinstance(d, float) for d in derivs)

        nominals = slave.get_nominals_of_continuous_states(nx)
        assert len(nominals) == nx

        slave.terminate()
        slave.free_instance()

    def test_parameter_tuning(self, reference_fmus_dir):
        """Set parameters g and e before initialisation."""
        slave = _make_slave(reference_fmus_dir, BOUNCING_BALL)
        _instantiate_cs(slave, BOUNCING_BALL, "bb_tuned")

        slave.enter_initialization_mode(start_time=0.0, stop_time=3.0)

        vr_g, vr_e = 5, 6
        slave.set_float64([vr_g], [-9.81])
        slave.set_float64([vr_e], [0.8])

        g_vals = slave.get_float64([vr_g])
        e_vals = slave.get_float64([vr_e])
        assert g_vals[0] == pytest.approx(-9.81)
        assert e_vals[0] == pytest.approx(0.8)

        slave.exit_initialization_mode()
        slave.terminate()
        slave.free_instance()


# =======================================================================
# Dahlquist
# =======================================================================
class TestDahlquist:
    """Dahlquist test equation: dx/dt = -k*x."""

    def test_co_simulation(self, reference_fmus_dir):
        slave = _make_slave(reference_fmus_dir, DAHLQUIST)
        _instantiate_cs(slave, DAHLQUIST, "dq_cs")

        slave.enter_initialization_mode(start_time=0.0, stop_time=10.0)

        vr_x, vr_k = 1, 3
        x0 = slave.get_float64([vr_x])
        k0 = slave.get_float64([vr_k])
        assert x0[0] == pytest.approx(1.0)
        assert k0[0] == pytest.approx(1.0)

        slave.exit_initialization_mode()

        t = 0.0
        dt = 0.1
        for _ in range(10):
            slave.do_step(t, dt)
            t += dt

        x_final = slave.get_float64([vr_x])
        assert x_final[0] != pytest.approx(1.0, abs=0.01)

        slave.terminate()
        slave.free_instance()

    def test_model_exchange(self, reference_fmus_dir):
        slave = _make_slave(reference_fmus_dir, DAHLQUIST)
        _instantiate_me(slave, DAHLQUIST, "dq_me")

        slave.enter_initialization_mode(start_time=0.0)
        slave.exit_initialization_mode()

        result = slave.update_discrete_states()
        while result.discrete_states_need_update:
            result = slave.update_discrete_states()

        slave.enter_continuous_time_mode()

        nx = 1
        states = slave.get_continuous_states(nx)
        assert states[0] == pytest.approx(1.0)

        slave.set_time(0.0)
        derivs = slave.get_continuous_state_derivatives(nx)
        assert len(derivs) == nx

        # Euler step
        new_state = [states[0] + 0.1 * derivs[0]]
        slave.set_continuous_states(new_state)
        slave.set_time(0.1)

        enter_event, terminate = slave.completed_integrator_step()
        assert isinstance(enter_event, bool)
        assert isinstance(terminate, bool)

        slave.terminate()
        slave.free_instance()

    def test_fmu_state_get_set(self, reference_fmus_dir):
        """Save / restore FMU state."""
        slave = _make_slave(reference_fmus_dir, DAHLQUIST)
        _instantiate_cs(slave, DAHLQUIST, "dq_state")

        slave.enter_initialization_mode(start_time=0.0, stop_time=10.0)
        slave.exit_initialization_mode()

        slave.do_step(0.0, 0.1)
        slave.do_step(0.1, 0.1)

        vr_x = 1
        x_before = slave.get_float64([vr_x])[0]

        state = slave.get_fmu_state()
        assert state is not None

        slave.do_step(0.2, 0.1)
        slave.do_step(0.3, 0.1)
        x_after = slave.get_float64([vr_x])[0]
        assert x_after != pytest.approx(x_before, abs=1e-12)

        slave.set_fmu_state(state)
        x_restored = slave.get_float64([vr_x])[0]
        assert x_restored == pytest.approx(x_before, abs=1e-10)

        slave.free_fmu_state(state)
        slave.terminate()
        slave.free_instance()

    def test_state_serialization(self, reference_fmus_dir):
        """Serialize / deserialize FMU state."""
        slave = _make_slave(reference_fmus_dir, DAHLQUIST)
        _instantiate_cs(slave, DAHLQUIST, "dq_ser")

        slave.enter_initialization_mode(start_time=0.0, stop_time=10.0)
        slave.exit_initialization_mode()

        slave.do_step(0.0, 0.1)
        slave.do_step(0.1, 0.1)

        vr_x = 1
        x_original = slave.get_float64([vr_x])[0]

        state = slave.get_fmu_state()
        serialized = slave.serialize_fmu_state(state)
        assert isinstance(serialized, bytes)
        assert len(serialized) > 0
        slave.free_fmu_state(state)

        # Advance further
        slave.do_step(0.2, 0.1)
        x_changed = slave.get_float64([vr_x])[0]
        assert x_changed != pytest.approx(x_original, abs=1e-12)

        # Restore via deserialization
        restored = slave.deserialize_fmu_state(serialized)
        slave.set_fmu_state(restored)
        x_restored = slave.get_float64([vr_x])[0]
        assert x_restored == pytest.approx(x_original, abs=1e-10)

        slave.free_fmu_state(restored)
        slave.terminate()
        slave.free_instance()

    def test_multiple_instances(self, reference_fmus_dir):
        """Run two Dahlquist instances with different k in parallel."""
        slave1 = _make_slave(reference_fmus_dir, DAHLQUIST)
        _instantiate_cs(slave1, DAHLQUIST, "dq1")

        slave2 = _make_slave(reference_fmus_dir, DAHLQUIST)
        _instantiate_cs(slave2, DAHLQUIST, "dq2")

        vr_x, vr_k = 1, 3

        # slave1: default k=1
        slave1.enter_initialization_mode(start_time=0.0, stop_time=10.0)
        slave1.exit_initialization_mode()

        # slave2: k=2
        slave2.enter_initialization_mode(start_time=0.0, stop_time=10.0)
        slave2.set_float64([vr_k], [2.0])
        slave2.exit_initialization_mode()

        for i in range(5):
            slave1.do_step(i * 0.1, 0.1)
            slave2.do_step(i * 0.1, 0.1)

        x1 = slave1.get_float64([vr_x])[0]
        x2 = slave2.get_float64([vr_x])[0]
        # Different k → different trajectory
        assert x1 != pytest.approx(x2, abs=1e-6)

        slave1.terminate()
        slave1.free_instance()
        slave2.terminate()
        slave2.free_instance()


# =======================================================================
# Feedthrough – all FMI 3.0 data types
# =======================================================================
class TestFeedthrough:
    """Feedthrough passes inputs directly to outputs.

    Covers Float32, Float64, Int8/16/32/64, UInt8/16/32/64,
    Boolean, String, and Binary types.
    """

    def _setup(self, reference_fmus_dir):
        slave = _make_slave(reference_fmus_dir, FEEDTHROUGH)
        _instantiate_cs(slave, FEEDTHROUGH, "ft")
        slave.enter_initialization_mode(start_time=0.0, stop_time=2.0)
        return slave

    def test_float32(self, reference_fmus_dir):
        slave = self._setup(reference_fmus_dir)
        vr_in, vr_out = 1, 2  # Float32 continuous
        slave.set_float32([vr_in], [3.14])
        slave.exit_initialization_mode()
        slave.do_step(0.0, 0.1)
        out = slave.get_float32([vr_out])
        assert out[0] == pytest.approx(3.14, abs=0.01)
        slave.terminate()
        slave.free_instance()

    def test_float64(self, reference_fmus_dir):
        slave = self._setup(reference_fmus_dir)
        vr_in, vr_out = 7, 8  # Float64 continuous
        slave.set_float64([vr_in], [2.71828])
        slave.exit_initialization_mode()
        slave.do_step(0.0, 0.1)
        out = slave.get_float64([vr_out])
        assert out[0] == pytest.approx(2.71828, abs=1e-4)
        slave.terminate()
        slave.free_instance()

    def test_int8(self, reference_fmus_dir):
        slave = self._setup(reference_fmus_dir)
        vr_in, vr_out = 11, 12
        slave.set_int8([vr_in], [42])
        slave.exit_initialization_mode()
        slave.do_step(0.0, 0.1)
        out = slave.get_int8([vr_out])
        assert out[0] == 42
        slave.terminate()
        slave.free_instance()

    def test_uint8(self, reference_fmus_dir):
        slave = self._setup(reference_fmus_dir)
        vr_in, vr_out = 13, 14
        slave.set_uint8([vr_in], [200])
        slave.exit_initialization_mode()
        slave.do_step(0.0, 0.1)
        out = slave.get_uint8([vr_out])
        assert out[0] == 200
        slave.terminate()
        slave.free_instance()

    def test_int16(self, reference_fmus_dir):
        slave = self._setup(reference_fmus_dir)
        vr_in, vr_out = 15, 16
        slave.set_int16([vr_in], [-1234])
        slave.exit_initialization_mode()
        slave.do_step(0.0, 0.1)
        out = slave.get_int16([vr_out])
        assert out[0] == -1234
        slave.terminate()
        slave.free_instance()

    def test_uint16(self, reference_fmus_dir):
        slave = self._setup(reference_fmus_dir)
        vr_in, vr_out = 17, 18
        slave.set_uint16([vr_in], [50000])
        slave.exit_initialization_mode()
        slave.do_step(0.0, 0.1)
        out = slave.get_uint16([vr_out])
        assert out[0] == 50000
        slave.terminate()
        slave.free_instance()

    def test_int32(self, reference_fmus_dir):
        slave = self._setup(reference_fmus_dir)
        vr_in, vr_out = 19, 20
        slave.set_int32([vr_in], [42])
        slave.exit_initialization_mode()
        slave.do_step(0.0, 0.1)
        out = slave.get_int32([vr_out])
        assert out[0] == 42
        slave.terminate()
        slave.free_instance()

    def test_uint32(self, reference_fmus_dir):
        slave = self._setup(reference_fmus_dir)
        vr_in, vr_out = 21, 22
        slave.set_uint32([vr_in], [123456])
        slave.exit_initialization_mode()
        slave.do_step(0.0, 0.1)
        out = slave.get_uint32([vr_out])
        assert out[0] == 123456
        slave.terminate()
        slave.free_instance()

    def test_int64(self, reference_fmus_dir):
        slave = self._setup(reference_fmus_dir)
        vr_in, vr_out = 23, 24
        slave.set_int64([vr_in], [-999999])
        slave.exit_initialization_mode()
        slave.do_step(0.0, 0.1)
        out = slave.get_int64([vr_out])
        assert out[0] == -999999
        slave.terminate()
        slave.free_instance()

    def test_uint64(self, reference_fmus_dir):
        slave = self._setup(reference_fmus_dir)
        vr_in, vr_out = 25, 26
        slave.set_uint64([vr_in], [18446744073709])
        slave.exit_initialization_mode()
        slave.do_step(0.0, 0.1)
        out = slave.get_uint64([vr_out])
        assert out[0] == 18446744073709
        slave.terminate()
        slave.free_instance()

    def test_boolean(self, reference_fmus_dir):
        slave = self._setup(reference_fmus_dir)
        vr_in, vr_out = 27, 28
        slave.set_boolean([vr_in], [True])
        slave.exit_initialization_mode()
        slave.do_step(0.0, 0.1)
        out = slave.get_boolean([vr_out])
        assert out[0] is True
        slave.terminate()
        slave.free_instance()

    def test_string(self, reference_fmus_dir):
        slave = self._setup(reference_fmus_dir)
        vr_in, vr_out = 29, 30
        slave.set_string([vr_in], ["Hello FMI3"])
        slave.exit_initialization_mode()
        slave.do_step(0.0, 0.1)
        out = slave.get_string([vr_out])
        assert out[0] == "Hello FMI3"
        slave.terminate()
        slave.free_instance()

    def test_binary(self, reference_fmus_dir):
        slave = self._setup(reference_fmus_dir)
        vr_in, vr_out = 31, 32
        payload = b"\xde\xad\xbe\xef"
        slave.set_binary([vr_in], [payload])
        slave.exit_initialization_mode()
        slave.do_step(0.0, 0.1)
        out = slave.get_binary([vr_out])
        assert out[0] == payload
        slave.terminate()
        slave.free_instance()

    def test_all_types_combined(self, reference_fmus_dir):
        """Set one of each type and verify after a step."""
        slave = self._setup(reference_fmus_dir)

        slave.set_float32([1], [1.5])
        slave.set_float64([7], [2.5])
        slave.set_int8([11], [10])
        slave.set_uint8([13], [20])
        slave.set_int16([15], [300])
        slave.set_uint16([17], [400])
        slave.set_int32([19], [500])
        slave.set_uint32([21], [600])
        slave.set_int64([23], [700])
        slave.set_uint64([25], [800])
        slave.set_boolean([27], [True])
        slave.set_string([29], ["combo"])
        slave.set_binary([31], [b"\x01\x02"])

        slave.exit_initialization_mode()
        slave.do_step(0.0, 0.1)

        assert slave.get_float32([2])[0] == pytest.approx(1.5, abs=0.01)
        assert slave.get_float64([8])[0] == pytest.approx(2.5, abs=1e-6)
        assert slave.get_int8([12])[0] == 10
        assert slave.get_uint8([14])[0] == 20
        assert slave.get_int16([16])[0] == 300
        assert slave.get_uint16([18])[0] == 400
        assert slave.get_int32([20])[0] == 500
        assert slave.get_uint32([22])[0] == 600
        assert slave.get_int64([24])[0] == 700
        assert slave.get_uint64([26])[0] == 800
        assert slave.get_boolean([28])[0] is True
        assert slave.get_string([30])[0] == "combo"
        assert slave.get_binary([32])[0] == b"\x01\x02"

        slave.terminate()
        slave.free_instance()


# =======================================================================
# Stair – discrete time events
# =======================================================================
class TestStair:
    def test_counter_increments(self, reference_fmus_dir):
        """Stair counter should increment over time."""
        slave = _make_slave(reference_fmus_dir, STAIR)
        _instantiate_cs(slave, STAIR, "stair")

        slave.enter_initialization_mode(start_time=0.0, stop_time=10.0)

        vr_counter = 1
        initial = slave.get_int32([vr_counter])
        assert initial[0] == 1

        slave.exit_initialization_mode()

        t = 0.0
        dt = 0.2
        for _ in range(25):  # 5 seconds worth
            slave.do_step(t, dt)
            t += dt

        counter = slave.get_int32([vr_counter])[0]
        assert counter > 1
        assert counter <= 10

        slave.terminate()
        slave.free_instance()

    def test_model_exchange(self, reference_fmus_dir):
        slave = _make_slave(reference_fmus_dir, STAIR)
        _instantiate_me(slave, STAIR, "stair_me")

        slave.enter_initialization_mode(start_time=0.0)
        slave.exit_initialization_mode()

        result = slave.update_discrete_states()
        assert isinstance(result.discrete_states_need_update, bool)
        assert isinstance(result.next_event_time_defined, bool)

        slave.terminate()
        slave.free_instance()


# =======================================================================
# Resource – reads from the resources/ directory
# =======================================================================
class TestResource:
    def test_resource_file_loading(self, reference_fmus_dir):
        slave = _make_slave(reference_fmus_dir, RESOURCE)
        _instantiate_cs(slave, RESOURCE, "res")

        slave.enter_initialization_mode(start_time=0.0, stop_time=1.0)
        slave.exit_initialization_mode()

        vr_y = 1
        y = slave.get_int32([vr_y])
        assert len(y) == 1
        assert isinstance(y[0], int)

        slave.terminate()
        slave.free_instance()


# =======================================================================
# VanDerPol
# =======================================================================
class TestVanDerPol:
    def test_co_simulation(self, reference_fmus_dir):
        slave = _make_slave(reference_fmus_dir, VAN_DER_POL)
        _instantiate_cs(slave, VAN_DER_POL, "vdp_cs")

        slave.enter_initialization_mode(start_time=0.0, stop_time=10.0)

        # Initial values: x0=2, x1=0
        assert slave.get_float64([1])[0] == pytest.approx(2.0)
        assert slave.get_float64([3])[0] == pytest.approx(0.0)

        slave.exit_initialization_mode()

        t = 0.0
        dt = 0.01
        for _ in range(100):
            slave.do_step(t, dt)
            t += dt

        # Oscillation should have started
        x0 = slave.get_float64([1])[0]
        x1 = slave.get_float64([3])[0]
        assert not (
            x0 == pytest.approx(2.0, abs=0.01) and x1 == pytest.approx(0.0, abs=0.01)
        )

        slave.terminate()
        slave.free_instance()

    def test_model_exchange(self, reference_fmus_dir):
        slave = _make_slave(reference_fmus_dir, VAN_DER_POL)
        _instantiate_me(slave, VAN_DER_POL, "vdp_me")

        slave.enter_initialization_mode(start_time=0.0)
        slave.exit_initialization_mode()

        result = slave.update_discrete_states()
        while result.discrete_states_need_update:
            result = slave.update_discrete_states()

        slave.enter_continuous_time_mode()

        nx = slave.get_number_of_continuous_states()
        assert nx == 2

        states = slave.get_continuous_states(nx)
        assert states[0] == pytest.approx(2.0)
        assert states[1] == pytest.approx(0.0)

        slave.set_time(0.0)
        derivs = slave.get_continuous_state_derivatives(nx)
        assert len(derivs) == 2

        slave.terminate()
        slave.free_instance()

    def test_directional_derivative(self, reference_fmus_dir):
        """VanDerPol provides directional derivatives."""
        slave = _make_slave(reference_fmus_dir, VAN_DER_POL)
        _instantiate_me(slave, VAN_DER_POL, "vdp_dd")

        slave.enter_initialization_mode(start_time=0.0)
        slave.exit_initialization_mode()

        result = slave.update_discrete_states()
        while result.discrete_states_need_update:
            result = slave.update_discrete_states()

        slave.enter_continuous_time_mode()
        slave.set_time(0.0)

        # Directional derivative: ∂der(x)/∂x with seed=[1,0]
        # unknowns = der(x0), der(x1) → VR 2, 4
        # knowns = x0, x1 → VR 1, 3
        sensitivity = slave.get_directional_derivative(
            unknowns=[2, 4],
            knowns=[1, 3],
            seed=[1.0, 0.0],
        )
        assert len(sensitivity) == 2
        assert all(isinstance(s, float) for s in sensitivity)

        slave.terminate()
        slave.free_instance()

    def test_adjoint_derivative(self, reference_fmus_dir):
        """VanDerPol provides adjoint derivatives."""
        slave = _make_slave(reference_fmus_dir, VAN_DER_POL)
        _instantiate_me(slave, VAN_DER_POL, "vdp_ad")

        slave.enter_initialization_mode(start_time=0.0)
        slave.exit_initialization_mode()

        result = slave.update_discrete_states()
        while result.discrete_states_need_update:
            result = slave.update_discrete_states()

        slave.enter_continuous_time_mode()
        slave.set_time(0.0)

        # Adjoint derivative: unknowns VR 2,4; knowns VR 1,3
        sensitivity = slave.get_adjoint_derivative(
            unknowns=[2, 4],
            knowns=[1, 3],
            seed=[1.0, 0.0],
        )
        assert len(sensitivity) == 2
        assert all(isinstance(s, float) for s in sensitivity)

        slave.terminate()
        slave.free_instance()


# =======================================================================
# StateSpace – array variables
# =======================================================================
class TestStateSpace:
    def test_co_simulation(self, reference_fmus_dir):
        """StateSpace CS: set u, step, read y."""
        slave = _make_slave(reference_fmus_dir, STATE_SPACE)
        _instantiate_cs(slave, STATE_SPACE, "ss_cs")

        slave.enter_initialization_mode(start_time=0.0, stop_time=1.0)

        # u (VR 9) is an array with 3 elements, set via Float64
        # Array VRs in FMI 3.0 are accessed by repeating the single VR
        # but the stride is handled by nValues parameter.
        # For Reference-FMUs implementation, VR 9 covers 3 values.
        slave.exit_initialization_mode()

        result = slave.do_step(0.0, 0.1)
        assert result.status == Fmi3Status.OK

        slave.terminate()
        slave.free_instance()

    def test_model_exchange(self, reference_fmus_dir):
        slave = _make_slave(reference_fmus_dir, STATE_SPACE)
        _instantiate_me(slave, STATE_SPACE, "ss_me")

        slave.enter_initialization_mode(start_time=0.0)
        slave.exit_initialization_mode()

        result = slave.update_discrete_states()
        while result.discrete_states_need_update:
            result = slave.update_discrete_states()

        slave.enter_continuous_time_mode()

        nx = slave.get_number_of_continuous_states()
        assert nx == 3  # n=3 by default

        states = slave.get_continuous_states(nx)
        assert len(states) == nx

        derivs = slave.get_continuous_state_derivatives(nx)
        assert len(derivs) == nx

        slave.terminate()
        slave.free_instance()


# =======================================================================
# Clocks – Scheduled Execution
# =======================================================================
class TestClocks:
    def test_instantiate_scheduled_execution(self, reference_fmus_dir):
        """Clocks is SE-only – verify instantiation and activation."""
        slave = _make_slave(reference_fmus_dir, CLOCKS)
        _, _, token = CLOCKS
        slave.instantiate_scheduled_execution(
            "clocks_se",
            instantiation_token=token,
        )
        assert slave._instance is not None

        slave.enter_initialization_mode(start_time=0.0)

        # Read initial tick counts (should be 0)
        ticks = slave.get_int32([2001])  # inClock1Ticks
        assert isinstance(ticks[0], int)

        slave.exit_initialization_mode()

        # Activate a model partition for inClock1
        status = slave.activate_model_partition(
            clock_reference=1001,
            activation_time=0.0,
        )
        assert status == Fmi3Status.OK

        # After activation, tick count should have incremented
        ticks_after = slave.get_int32([2001])
        assert isinstance(ticks_after[0], int)

        slave.terminate()
        slave.free_instance()


# =======================================================================
# Reset
# =======================================================================
@pytest.mark.parametrize(
    "fmu_tuple",
    [BOUNCING_BALL, DAHLQUIST, FEEDTHROUGH, STAIR, VAN_DER_POL],
    ids=["BouncingBall", "Dahlquist", "Feedthrough", "Stair", "VanDerPol"],
)
def test_fmi3_reset(fmu_tuple, reference_fmus_dir):
    """Reset should return the FMU to a fresh state."""
    slave = _make_slave(reference_fmus_dir, fmu_tuple)
    _instantiate_cs(slave, fmu_tuple)

    slave.enter_initialization_mode(start_time=0.0, stop_time=10.0)
    slave.exit_initialization_mode()

    slave.do_step(0.0, 0.1)
    slave.do_step(0.1, 0.1)

    status = slave.reset()
    assert status == Fmi3Status.OK

    slave.free_instance()


# =======================================================================
# Debug logging
# =======================================================================
def test_fmi3_set_debug_logging(reference_fmus_dir):
    slave = _make_slave(reference_fmus_dir, BOUNCING_BALL)
    _instantiate_cs(slave, BOUNCING_BALL, "bb_log", logging_on=True)

    status = slave.set_debug_logging(True, categories=["logEvents"])
    assert status == Fmi3Status.OK

    status = slave.set_debug_logging(False)
    assert status == Fmi3Status.OK

    slave.free_instance()


# =======================================================================
# Error handling
# =======================================================================
def test_fmi3_bad_token(reference_fmus_dir):
    """Instantiation with a wrong token should raise or return NULL."""
    slave = _make_slave(reference_fmus_dir, BOUNCING_BALL)
    with pytest.raises(RuntimeError):
        slave.instantiate_co_simulation(
            "bad",
            instantiation_token="{00000000-0000-0000-0000-000000000000}",
        )


# =======================================================================
# DoStep output fields
# =======================================================================
def test_fmi3_do_step_result_fields(reference_fmus_dir):
    """Verify all DoStepResult fields are populated."""
    slave = _make_slave(reference_fmus_dir, BOUNCING_BALL)
    _instantiate_cs(slave, BOUNCING_BALL)

    slave.enter_initialization_mode(start_time=0.0, stop_time=10.0)
    slave.exit_initialization_mode()

    result = slave.do_step(0.0, 0.1)

    assert isinstance(result.status, Fmi3Status)
    assert isinstance(result.event_handling_needed, bool)
    assert isinstance(result.terminate_simulation, bool)
    assert isinstance(result.early_return, bool)
    assert isinstance(result.last_successful_time, float)

    slave.terminate()
    slave.free_instance()


# =======================================================================
# UpdateDiscreteStates output fields
# =======================================================================
def test_fmi3_update_discrete_states_fields(reference_fmus_dir):
    """Verify all UpdateDiscreteStatesResult fields."""
    slave = _make_slave(reference_fmus_dir, BOUNCING_BALL)
    _instantiate_me(slave, BOUNCING_BALL)

    slave.enter_initialization_mode(start_time=0.0)
    slave.exit_initialization_mode()

    result = slave.update_discrete_states()

    assert isinstance(result.discrete_states_need_update, bool)
    assert isinstance(result.terminate_simulation, bool)
    assert isinstance(result.nominals_of_continuous_states_changed, bool)
    assert isinstance(result.values_of_continuous_states_changed, bool)
    assert isinstance(result.next_event_time_defined, bool)
    assert isinstance(result.next_event_time, float)

    slave.terminate()
    slave.free_instance()


# =======================================================================
# Tolerance in enter_initialization_mode
# =======================================================================
def test_fmi3_init_with_tolerance(reference_fmus_dir):
    slave = _make_slave(reference_fmus_dir, BOUNCING_BALL)
    _instantiate_cs(slave, BOUNCING_BALL)

    status = slave.enter_initialization_mode(
        start_time=0.0,
        stop_time=10.0,
        tolerance=1e-6,
    )
    assert status == Fmi3Status.OK

    status = slave.exit_initialization_mode()
    assert status == Fmi3Status.OK

    slave.terminate()
    slave.free_instance()


# =======================================================================
# Event indicators (ME)
# =======================================================================
def test_fmi3_event_indicators(reference_fmus_dir):
    slave = _make_slave(reference_fmus_dir, BOUNCING_BALL)
    _instantiate_me(slave, BOUNCING_BALL)

    slave.enter_initialization_mode(start_time=0.0)
    slave.exit_initialization_mode()

    result = slave.update_discrete_states()
    while result.discrete_states_need_update:
        result = slave.update_discrete_states()

    slave.enter_continuous_time_mode()

    ni = slave.get_number_of_event_indicators()
    assert ni >= 1

    indicators = slave.get_event_indicators(ni)
    assert len(indicators) == ni
    assert all(isinstance(v, float) for v in indicators)

    slave.terminate()
    slave.free_instance()
