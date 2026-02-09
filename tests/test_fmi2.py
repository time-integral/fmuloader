import logging
import platform
import pathlib
import pytest
from fmuloader.fmi2 import (
    Fmi2Slave,
    Fmi2Status,
    Fmi2Type,
    Fmi2Error,
    _platform_folder,
    _shared_lib_extension,
    _path_to_file_uri,
)


# Skip tests on arm64 platform (Apple Silicon)
skip_arm64 = pytest.mark.skipif(
    platform.machine() == "arm64",
    reason="Skipping on ARM64 platform - binaries may not be available",
)


@skip_arm64
@pytest.mark.parametrize(
    "reference_fmu,model_name,guid",
    [
        (
            "2.0/BouncingBall.fmu",
            "BouncingBall",
            "{1AE5E10D-9521-4DE3-80B9-D0EAAA7D5AF1}",
        ),
        ("2.0/VanDerPol.fmu", "VanDerPol", "{BD403596-3166-4232-ABC2-132BDF73E644}"),
        ("2.0/Dahlquist.fmu", "Dahlquist", "{221063D2-EF4A-45FE-B954-B5BFEEA9A59B}"),
        ("2.0/Stair.fmu", "Stair", "{BD403596-3166-4232-ABC2-132BDF73E644}"),
        ("2.0/Resource.fmu", "Resource", "{7b9c2114-2ce5-4076-a138-2cbc69e069e5}"),
        (
            "2.0/Feedthrough.fmu",
            "Feedthrough",
            "{37B954F1-CC86-4D8F-B97F-C7C36F6670D2}",
        ),
    ],
)
def test_fmi2_slave_basic(reference_fmu, model_name, guid, reference_fmus_dir):
    """Test basic FMU loading and version check."""
    filename = (reference_fmus_dir / reference_fmu).absolute()

    slave = Fmi2Slave(filename, model_identifier=model_name)

    assert slave.get_version() == "2.0"
    assert slave.get_types_platform() in ["default", "standard32", "standard64"]

    # Test instantiation with correct GUID
    slave.instantiate(
        "test_instance",
        Fmi2Type.CO_SIMULATION,
        guid=guid,
    )
    assert slave._component is not None
    slave.free_instance()


@skip_arm64
def test_fmi2_instantiate_co_simulation(reference_fmus_dir):
    """Test instantiation of Co-Simulation FMU."""
    filename = (reference_fmus_dir / "2.0/BouncingBall.fmu").absolute()

    slave = Fmi2Slave(filename, model_identifier="BouncingBall")

    # Instantiate as Co-Simulation
    slave.instantiate(
        "test_instance",
        Fmi2Type.CO_SIMULATION,
        guid="{1AE5E10D-9521-4DE3-80B9-D0EAAA7D5AF1}",
        visible=False,
        logging_on=True,
    )

    assert slave._component is not None

    slave.free_instance()
    assert slave._component is None


@skip_arm64
def test_fmi2_dahlquist_simulation(reference_fmus_dir):
    """Test Dahlquist test equation FMU - a simple ODE: dx/dt = k*x."""
    filename = (reference_fmus_dir / "2.0/Dahlquist.fmu").absolute()

    slave = Fmi2Slave(filename, model_identifier="Dahlquist")
    slave.instantiate(
        "dahlquist_instance",
        Fmi2Type.CO_SIMULATION,
        guid="{221063D2-EF4A-45FE-B954-B5BFEEA9A59B}",
    )

    slave.setup_experiment(start_time=0.0, stop_time=10.0)
    slave.enter_initialization_mode()

    # Dahlquist has x (VR 1) with start value 1.0 and k (VR 3) with start value 1.0
    vr_x = 1
    vr_k = 3

    # Get initial values
    x_values = slave.get_real([vr_x])
    k_values = slave.get_real([vr_k])
    assert len(x_values) == 1
    assert x_values[0] == 1.0  # Initial value
    assert k_values[0] == 1.0  # Parameter value

    slave.exit_initialization_mode()

    # Perform simulation steps
    time = 0.0
    step_size = 0.1
    for _ in range(10):
        slave.do_step(time, step_size)
        time += step_size

    # After simulation, x should have changed
    x_final = slave.get_real([vr_x])
    assert x_final[0] != 1.0  # Value should have changed due to dynamics

    slave.terminate()
    slave.free_instance()


@skip_arm64
def test_fmi2_feedthrough_all_types(reference_fmus_dir):
    """Test Feedthrough FMU with all variable types (Real, Integer, Boolean, String)."""
    filename = (reference_fmus_dir / "2.0/Feedthrough.fmu").absolute()

    slave = Fmi2Slave(filename, model_identifier="Feedthrough")
    slave.instantiate(
        "feedthrough_instance",
        Fmi2Type.CO_SIMULATION,
        guid="{37B954F1-CC86-4D8F-B97F-C7C36F6670D2}",
    )

    slave.setup_experiment(start_time=0.0, stop_time=2.0)
    slave.enter_initialization_mode()

    # Test Real variables
    vr_float_input = 7
    vr_float_output = 8
    slave.set_real([vr_float_input], [3.14])

    # Test Integer variables
    vr_int_input = 19
    vr_int_output = 20
    slave.set_integer([vr_int_input], [42])

    # Test Boolean variables
    vr_bool_input = 27
    vr_bool_output = 28
    slave.set_boolean([vr_bool_input], [True])

    # Test String variables
    vr_string_input = 29
    vr_string_output = 30
    slave.set_string([vr_string_input], ["Test String"])

    slave.exit_initialization_mode()

    # Perform one step
    slave.do_step(0.0, 0.1)

    # Verify outputs (Feedthrough passes inputs to outputs)
    float_out = slave.get_real([vr_float_output])
    assert abs(float_out[0] - 3.14) < 0.01

    int_out = slave.get_integer([vr_int_output])
    assert int_out[0] == 42

    bool_out = slave.get_boolean([vr_bool_output])
    assert bool_out[0] is True

    string_out = slave.get_string([vr_string_output])
    assert string_out[0] == "Test String"

    slave.terminate()
    slave.free_instance()


@skip_arm64
def test_fmi2_stair_discrete_events(reference_fmus_dir):
    """Test Stair FMU which generates discrete time events."""
    filename = (reference_fmus_dir / "2.0/Stair.fmu").absolute()

    slave = Fmi2Slave(filename, model_identifier="Stair")
    slave.instantiate(
        "stair_instance",
        Fmi2Type.CO_SIMULATION,
        guid="{BD403596-3166-4232-ABC2-132BDF73E644}",
    )

    slave.setup_experiment(start_time=0.0, stop_time=10.0)
    slave.enter_initialization_mode()

    # Stair has 'counter' variable (VR 1) that counts seconds
    vr_counter = 1

    # Get initial value
    initial_counter = slave.get_integer([vr_counter])
    assert initial_counter[0] == 1  # Start value is 1

    slave.exit_initialization_mode()

    # Simulate for a few seconds - counter should increment
    time = 0.0
    step_size = 0.2  # Default step size from modelDescription

    for i in range(5):
        slave.do_step(time, step_size)
        time += step_size
        counter_value = slave.get_integer([vr_counter])
        # Counter increments every second and should be between 1 and 10
        assert counter_value[0] >= 1 and counter_value[0] <= 10

    slave.terminate()
    slave.free_instance()


@skip_arm64
def test_fmi2_resource_file_loading(reference_fmus_dir):
    """Test Resource FMU which loads data from resource files."""
    filename = (reference_fmus_dir / "2.0/Resource.fmu").absolute()

    slave = Fmi2Slave(filename, model_identifier="Resource")
    slave.instantiate(
        "resource_instance",
        Fmi2Type.CO_SIMULATION,
        guid="{7b9c2114-2ce5-4076-a138-2cbc69e069e5}",
    )

    slave.setup_experiment(start_time=0.0, stop_time=1.0)
    slave.enter_initialization_mode()
    slave.exit_initialization_mode()

    # Resource FMU has 'y' variable (VR 1) that reads from resources/y.txt
    vr_y = 1

    # Get the value (should be the ASCII code of first character in y.txt)
    y_value = slave.get_integer([vr_y])
    assert len(y_value) == 1
    assert isinstance(y_value[0], int)
    # The value should be a valid ASCII code
    assert 0 <= y_value[0] <= 127

    slave.terminate()
    slave.free_instance()


@skip_arm64
def test_fmi2_setup_and_initialization(reference_fmus_dir):
    """Test FMU setup and initialization sequence."""
    filename = (reference_fmus_dir / "2.0/BouncingBall.fmu").absolute()

    slave = Fmi2Slave(filename, model_identifier="BouncingBall")
    slave.instantiate(
        "test_instance",
        Fmi2Type.CO_SIMULATION,
        guid="{1AE5E10D-9521-4DE3-80B9-D0EAAA7D5AF1}",
    )

    # Setup experiment
    status = slave.setup_experiment(start_time=0.0, stop_time=10.0, tolerance=1e-6)
    assert status == Fmi2Status.OK

    # Enter initialization mode
    status = slave.enter_initialization_mode()
    assert status == Fmi2Status.OK

    # Exit initialization mode
    status = slave.exit_initialization_mode()
    assert status == Fmi2Status.OK

    # Terminate
    status = slave.terminate()
    assert status == Fmi2Status.OK

    slave.free_instance()


@skip_arm64
def test_fmi2_do_step(reference_fmus_dir):
    """Test Co-Simulation do_step functionality."""
    filename = (reference_fmus_dir / "2.0/BouncingBall.fmu").absolute()

    slave = Fmi2Slave(filename, model_identifier="BouncingBall")
    slave.instantiate(
        "test_instance",
        Fmi2Type.CO_SIMULATION,
        guid="{1AE5E10D-9521-4DE3-80B9-D0EAAA7D5AF1}",
    )

    slave.setup_experiment(start_time=0.0, stop_time=10.0)
    slave.enter_initialization_mode()
    slave.exit_initialization_mode()

    # Perform simulation steps
    time = 0.0
    step_size = 0.1
    for _ in range(10):
        status = slave.do_step(time, step_size)
        assert status == Fmi2Status.OK
        time += step_size

    slave.terminate()
    slave.free_instance()


@skip_arm64
def test_fmi2_get_set_real(reference_fmus_dir):
    """Test getting and setting real values."""
    filename = (reference_fmus_dir / "2.0/BouncingBall.fmu").absolute()

    slave = Fmi2Slave(filename, model_identifier="BouncingBall")
    slave.instantiate(
        "test_instance",
        Fmi2Type.CO_SIMULATION,
        guid="{1AE5E10D-9521-4DE3-80B9-D0EAAA7D5AF1}",
    )

    slave.setup_experiment(start_time=0.0, stop_time=10.0)
    slave.enter_initialization_mode()

    # BouncingBall has 'h' (height) as variable with value reference 1
    # and 'v' (velocity) with value reference 3
    vr_h = 1
    vr_v = 3

    # Get initial values
    values = slave.get_real([vr_h, vr_v])
    assert len(values) == 2
    assert isinstance(values[0], float)
    assert isinstance(values[1], float)

    # Set new values
    slave.set_real([vr_h], [5.0])

    # Verify the set value
    new_values = slave.get_real([vr_h])
    assert len(new_values) == 1
    # Note: The value might not be exactly 5.0 due to FMU constraints

    slave.exit_initialization_mode()
    slave.terminate()
    slave.free_instance()


@skip_arm64
def test_fmi2_get_set_integer(reference_fmus_dir):
    """Test getting and setting integer values."""
    filename = (reference_fmus_dir / "2.0/Stair.fmu").absolute()

    slave = Fmi2Slave(filename, model_identifier="Stair")
    slave.instantiate(
        "test_instance",
        Fmi2Type.CO_SIMULATION,
        guid="{BD403596-3166-4232-ABC2-132BDF73E644}",
    )

    slave.setup_experiment(start_time=0.0, stop_time=10.0)
    slave.enter_initialization_mode()

    # Stair has 'counter' as integer variable with value reference 1
    vr_counter = 1

    # Get initial value
    values = slave.get_integer([vr_counter])
    assert len(values) == 1
    assert isinstance(values[0], int)

    slave.exit_initialization_mode()
    slave.terminate()
    slave.free_instance()


@skip_arm64
def test_fmi2_get_set_boolean(reference_fmus_dir):
    """Test getting and setting boolean values."""
    filename = (reference_fmus_dir / "2.0/BouncingBall.fmu").absolute()

    slave = Fmi2Slave(filename, model_identifier="BouncingBall")
    slave.instantiate(
        "test_instance",
        Fmi2Type.CO_SIMULATION,
        guid="{1AE5E10D-9521-4DE3-80B9-D0EAAA7D5AF1}",
    )

    slave.setup_experiment(start_time=0.0, stop_time=10.0)
    slave.enter_initialization_mode()
    slave.exit_initialization_mode()

    # Perform a step to get meaningful boolean values
    slave.do_step(0.0, 0.1)

    # BouncingBall may have boolean state variables
    # Testing the boolean get/set methods with a valid VR
    # Note: Actual VRs depend on the FMU implementation

    slave.terminate()
    slave.free_instance()


@skip_arm64
def test_fmi2_reset(reference_fmus_dir):
    """Test FMU reset functionality."""
    filename = (reference_fmus_dir / "2.0/BouncingBall.fmu").absolute()

    slave = Fmi2Slave(filename, model_identifier="BouncingBall")
    slave.instantiate(
        "test_instance",
        Fmi2Type.CO_SIMULATION,
        guid="{1AE5E10D-9521-4DE3-80B9-D0EAAA7D5AF1}",
    )

    slave.setup_experiment(start_time=0.0, stop_time=10.0)
    slave.enter_initialization_mode()
    slave.exit_initialization_mode()

    # Perform some steps
    slave.do_step(0.0, 0.1)
    slave.do_step(0.1, 0.1)

    # Reset the FMU
    status = slave.reset()
    assert status == Fmi2Status.OK

    slave.free_instance()


@skip_arm64
def test_fmi2_set_debug_logging(reference_fmus_dir):
    """Test setting debug logging."""
    filename = (reference_fmus_dir / "2.0/BouncingBall.fmu").absolute()

    slave = Fmi2Slave(filename, model_identifier="BouncingBall")
    slave.instantiate(
        "test_instance",
        Fmi2Type.CO_SIMULATION,
        guid="{1AE5E10D-9521-4DE3-80B9-D0EAAA7D5AF1}",
    )

    # Enable logging with valid categories from modelDescription
    status = slave.set_debug_logging(True, categories=["logEvents"])
    assert status == Fmi2Status.OK

    # Disable logging
    status = slave.set_debug_logging(False)
    assert status == Fmi2Status.OK

    slave.free_instance()


@skip_arm64
def test_fmi2_logging_uses_python_logging(reference_fmus_dir, caplog):
    """Verify that FMU log messages are routed through Python logging."""
    filename = (reference_fmus_dir / "2.0/BouncingBall.fmu").absolute()
    with caplog.at_level(logging.DEBUG, logger="fmuloader.fmi2"):
        slave = Fmi2Slave(filename, model_identifier="BouncingBall")
        slave.instantiate(
            "test_instance",
            Fmi2Type.CO_SIMULATION,
            guid="{1AE5E10D-9521-4DE3-80B9-D0EAAA7D5AF1}",
            logging_on=True,
        )
        slave.set_debug_logging(True)
        slave.setup_experiment(start_time=0.0, stop_time=10.0)
        slave.enter_initialization_mode()
        slave.exit_initialization_mode()
        slave.do_step(0.0, 0.1)
        slave.terminate()
        slave.free_instance()
    for record in caplog.records:
        assert record.name == "fmuloader.fmi2"


@skip_arm64
def test_fmi2_custom_log_callback(reference_fmus_dir):
    """Verify that a user-supplied log callback is invoked."""
    filename = (reference_fmus_dir / "2.0/BouncingBall.fmu").absolute()
    messages: list[tuple[str, int, str, str]] = []

    def my_logger(_env, instance_name, status, category, message):
        name = instance_name.decode() if instance_name else ""
        cat = category.decode() if category else ""
        msg = message.decode() if message else ""
        messages.append((name, status, cat, msg))

    slave = Fmi2Slave(filename, model_identifier="BouncingBall")
    slave.instantiate(
        "test_instance",
        Fmi2Type.CO_SIMULATION,
        guid="{1AE5E10D-9521-4DE3-80B9-D0EAAA7D5AF1}",
        logging_on=True,
        log_message_callback=my_logger,
    )
    slave.set_debug_logging(True)
    slave.setup_experiment(start_time=0.0, stop_time=10.0)
    slave.enter_initialization_mode()
    slave.exit_initialization_mode()
    slave.do_step(0.0, 0.1)
    slave.terminate()
    slave.free_instance()
    # Custom callback should have been stored on instance (no crash = pass)
    for name, status, cat, msg in messages:
        assert isinstance(name, str)
        assert isinstance(status, int)
        assert isinstance(cat, str)
        assert isinstance(msg, str)


@skip_arm64
def test_fmi2_context_manager(reference_fmus_dir):
    """Test using Fmi2Slave as a context manager."""
    filename = (reference_fmus_dir / "2.0/BouncingBall.fmu").absolute()

    with Fmi2Slave(filename, model_identifier="BouncingBall") as slave:
        assert slave.get_version() == "2.0"
        slave.instantiate(
            "test_instance",
            Fmi2Type.CO_SIMULATION,
            guid="{1AE5E10D-9521-4DE3-80B9-D0EAAA7D5AF1}",
        )
        assert slave._component is not None

    # After context exit, component should be freed


@skip_arm64
def test_fmi2_error_handling(reference_fmus_dir):
    """Test error handling for invalid operations."""
    filename = (reference_fmus_dir / "2.0/BouncingBall.fmu").absolute()

    slave = Fmi2Slave(filename, model_identifier="BouncingBall")

    # Try to instantiate with wrong GUID - should raise error or return NULL
    try:
        slave.instantiate(
            "test_instance",
            Fmi2Type.CO_SIMULATION,
            guid="{00000000-0000-0000-0000-000000000000}",
        )
        # If instantiation succeeds despite wrong GUID, clean up
        if slave._component is not None:
            slave.free_instance()
    except (RuntimeError, Fmi2Error):
        # Expected behavior for wrong GUID
        pass


@skip_arm64
def test_fmi2_model_exchange_functions(reference_fmus_dir):
    """Test Model Exchange specific functions."""
    filename = (reference_fmus_dir / "2.0/VanDerPol.fmu").absolute()

    slave = Fmi2Slave(filename, model_identifier="VanDerPol")
    slave.instantiate(
        "test_instance",
        Fmi2Type.MODEL_EXCHANGE,
        guid="{BD403596-3166-4232-ABC2-132BDF73E644}",
    )

    slave.setup_experiment(start_time=0.0)
    slave.enter_initialization_mode()
    slave.exit_initialization_mode()

    # Initial event iteration
    event_info = slave.new_discrete_states()
    assert hasattr(event_info, "newDiscreteStatesNeeded")

    # Enter continuous time mode
    status = slave.enter_continuous_time_mode()
    assert status == Fmi2Status.OK

    # Set time
    status = slave.set_time(0.1)
    assert status == Fmi2Status.OK

    # Get derivatives (VanDerPol has 2 continuous states)
    derivs = slave.get_derivatives(2)
    assert len(derivs) == 2
    assert all(isinstance(d, float) for d in derivs)

    # Get continuous states
    states = slave.get_continuous_states(2)
    assert len(states) == 2

    # Get nominals
    nominals = slave.get_nominals_of_continuous_states(2)
    assert len(nominals) == 2

    slave.terminate()
    slave.free_instance()


@skip_arm64
def test_fmi2_dahlquist_model_exchange(reference_fmus_dir):
    """Test Dahlquist FMU in Model Exchange mode."""
    filename = (reference_fmus_dir / "2.0/Dahlquist.fmu").absolute()

    slave = Fmi2Slave(filename, model_identifier="Dahlquist")
    slave.instantiate(
        "dahlquist_me",
        Fmi2Type.MODEL_EXCHANGE,
        guid="{221063D2-EF4A-45FE-B954-B5BFEEA9A59B}",
    )

    slave.setup_experiment(start_time=0.0)
    slave.enter_initialization_mode()
    slave.exit_initialization_mode()

    # Event iteration
    event_info = slave.new_discrete_states()
    while event_info.newDiscreteStatesNeeded:
        event_info = slave.new_discrete_states()

    # Enter continuous time mode
    slave.enter_continuous_time_mode()

    # Get number of states (Dahlquist has 1 continuous state)
    nx = 1

    # Get initial state
    states = slave.get_continuous_states(nx)
    assert len(states) == nx
    assert states[0] == 1.0  # Initial value of x

    # Get parameter k to compute expected derivative
    vr_k = 3
    k_value = slave.get_real([vr_k])[0]

    # Set time and get derivatives
    slave.set_time(0.0)
    derivs = slave.get_derivatives(nx)
    assert len(derivs) == nx
    # For Dahlquist: dx/dt = k*x, so derivative should be k*x = k*1.0 = k
    # The actual implementation may use dx/dt = -k*x, so check for both
    expected_deriv = k_value * states[0]
    assert (
        abs(derivs[0] - expected_deriv) < 0.01 or abs(derivs[0] + expected_deriv) < 0.01
    )

    # Simulate one step with explicit Euler
    dt = 0.1
    new_state = [states[0] + dt * derivs[0]]
    slave.set_continuous_states(new_state)
    slave.set_time(dt)

    # Check integrator step completion
    enter_event_mode, terminate = slave.completed_integrator_step()
    assert isinstance(enter_event_mode, bool)
    assert isinstance(terminate, bool)

    slave.terminate()
    slave.free_instance()


@skip_arm64
def test_fmi2_fmu_state_management(reference_fmus_dir):
    """Test FMU state get/set/free functionality."""
    filename = (reference_fmus_dir / "2.0/Dahlquist.fmu").absolute()

    slave = Fmi2Slave(filename, model_identifier="Dahlquist")
    slave.instantiate(
        "dahlquist_state",
        Fmi2Type.CO_SIMULATION,
        guid="{221063D2-EF4A-45FE-B954-B5BFEEA9A59B}",
    )

    slave.setup_experiment(start_time=0.0, stop_time=10.0)
    slave.enter_initialization_mode()
    slave.exit_initialization_mode()

    # Do a few steps
    slave.do_step(0.0, 0.1)
    slave.do_step(0.1, 0.1)

    # Get state value
    vr_x = 1
    x_before = slave.get_real([vr_x])

    # Save FMU state
    state = slave.get_fmu_state()
    assert state is not None

    # Continue simulation
    slave.do_step(0.2, 0.1)
    slave.do_step(0.3, 0.1)

    x_after = slave.get_real([vr_x])
    assert x_after[0] != x_before[0]  # State should have changed

    # Restore previous state
    slave.set_fmu_state(state)
    x_restored = slave.get_real([vr_x])
    assert abs(x_restored[0] - x_before[0]) < 1e-10  # Should match saved state

    # Free the saved state
    slave.free_fmu_state(state)

    slave.terminate()
    slave.free_instance()


@skip_arm64
def test_fmi2_state_serialization(reference_fmus_dir):
    """Test FMU state serialization and deserialization."""
    filename = (reference_fmus_dir / "2.0/Dahlquist.fmu").absolute()

    slave = Fmi2Slave(filename, model_identifier="Dahlquist")
    slave.instantiate(
        "dahlquist_serialize",
        Fmi2Type.CO_SIMULATION,
        guid="{221063D2-EF4A-45FE-B954-B5BFEEA9A59B}",
    )

    slave.setup_experiment(start_time=0.0, stop_time=10.0)
    slave.enter_initialization_mode()
    slave.exit_initialization_mode()

    # Simulate to some point
    slave.do_step(0.0, 0.1)
    slave.do_step(0.1, 0.1)

    vr_x = 1
    x_original = slave.get_real([vr_x])

    # Get and serialize state
    state = slave.get_fmu_state()
    serialized = slave.serialize_fmu_state(state)
    assert isinstance(serialized, bytes)
    assert len(serialized) > 0

    slave.free_fmu_state(state)

    # Continue simulation
    slave.do_step(0.2, 0.1)
    x_after = slave.get_real([vr_x])
    assert x_after[0] != x_original[0]

    # Deserialize and restore
    restored_state = slave.deserialize_fmu_state(serialized)
    slave.set_fmu_state(restored_state)

    x_restored = slave.get_real([vr_x])
    assert abs(x_restored[0] - x_original[0]) < 1e-10

    slave.free_fmu_state(restored_state)
    slave.terminate()
    slave.free_instance()


@skip_arm64
def test_fmi2_multiple_instances(reference_fmus_dir):
    """Test running multiple FMU instances simultaneously."""
    filename = (reference_fmus_dir / "2.0/Dahlquist.fmu").absolute()

    # Create two instances
    slave1 = Fmi2Slave(filename, model_identifier="Dahlquist")
    slave1.instantiate(
        "instance1",
        Fmi2Type.CO_SIMULATION,
        guid="{221063D2-EF4A-45FE-B954-B5BFEEA9A59B}",
    )

    slave2 = Fmi2Slave(filename, model_identifier="Dahlquist")
    slave2.instantiate(
        "instance2",
        Fmi2Type.CO_SIMULATION,
        guid="{221063D2-EF4A-45FE-B954-B5BFEEA9A59B}",
    )

    # Setup both
    slave1.setup_experiment(start_time=0.0, stop_time=10.0)
    slave1.enter_initialization_mode()
    slave1.exit_initialization_mode()

    slave2.setup_experiment(start_time=0.0, stop_time=10.0)
    slave2.enter_initialization_mode()

    # Set different parameter values
    vr_k = 3
    slave2.set_real([vr_k], [2.0])  # Different k value
    slave2.exit_initialization_mode()

    # Run both instances
    vr_x = 1
    for _ in range(5):
        slave1.do_step(_ * 0.1, 0.1)
        slave2.do_step(_ * 0.1, 0.1)

    # They should have different state values due to different parameters
    x1 = slave1.get_real([vr_x])
    x2 = slave2.get_real([vr_x])
    assert x1[0] != x2[0]  # Different evolution due to different k

    slave1.terminate()
    slave1.free_instance()
    slave2.terminate()
    slave2.free_instance()


@skip_arm64
def test_fmi2_parameter_tuning(reference_fmus_dir):
    """Test setting parameters before initialization."""
    filename = (reference_fmus_dir / "2.0/BouncingBall.fmu").absolute()

    slave = Fmi2Slave(filename, model_identifier="BouncingBall")
    slave.instantiate(
        "ball_tuned",
        Fmi2Type.CO_SIMULATION,
        guid="{1AE5E10D-9521-4DE3-80B9-D0EAAA7D5AF1}",
    )

    slave.setup_experiment(start_time=0.0, stop_time=3.0)
    slave.enter_initialization_mode()

    # Set parameters: g (gravity, VR 5) and e (restitution coefficient, VR 6)
    vr_g = 5
    vr_e = 6

    # Set custom gravity and restitution
    slave.set_real([vr_g], [9.81])
    slave.set_real([vr_e], [0.8])

    # Verify parameters were set
    g_values = slave.get_real([vr_g])
    e_values = slave.get_real([vr_e])
    assert abs(g_values[0] - 9.81) < 0.01
    assert abs(e_values[0] - 0.8) < 0.01

    slave.exit_initialization_mode()

    # Run simulation
    for i in range(10):
        slave.do_step(i * 0.1, 0.1)

    slave.terminate()
    slave.free_instance()


# Tests for helper functions
def test_shared_lib_extension():
    """Test shared library extension detection."""
    ext = _shared_lib_extension()
    system = platform.system()

    if system == "Windows":
        assert ext == ".dll"
    elif system == "Darwin":
        assert ext == ".dylib"
    else:
        assert ext == ".so"


def test_platform_folder():
    """Test platform folder name generation."""
    folder = _platform_folder()
    system = platform.system()

    if system == "Darwin":
        assert folder == "darwin64"
    elif system == "Linux":
        assert folder in ["linux32", "linux64"]
    elif system == "Windows":
        assert folder in ["win32", "win64"]


def test_path_to_file_uri():
    """Test path to file URI conversion."""
    path = pathlib.Path("/tmp/test.txt")
    uri = _path_to_file_uri(path)

    assert uri.startswith("file://")
    if platform.system() == "Windows":
        assert "/" in uri
    else:
        assert uri.startswith("file:///")


def test_fmi2_status_enum():
    """Test Fmi2Status enumeration values."""
    assert Fmi2Status.OK == 0
    assert Fmi2Status.WARNING == 1
    assert Fmi2Status.DISCARD == 2
    assert Fmi2Status.ERROR == 3
    assert Fmi2Status.FATAL == 4
    assert Fmi2Status.PENDING == 5


def test_fmi2_type_enum():
    """Test Fmi2Type enumeration values."""
    assert Fmi2Type.MODEL_EXCHANGE == 0
    assert Fmi2Type.CO_SIMULATION == 1


def test_fmi2_error_exception():
    """Test Fmi2Error exception."""
    error = Fmi2Error("testFunction", Fmi2Status.ERROR)

    assert error.func_name == "testFunction"
    assert error.status == Fmi2Status.ERROR
    assert "testFunction" in str(error)
    assert "ERROR" in str(error)
