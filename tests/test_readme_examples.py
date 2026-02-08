"""Tests that mirror the README.md code examples using real reference FMUs.

Each test corresponds to a code block in the README, ensuring the
documented API actually works.
"""

import platform

import pytest
from fmuloader.fmi2 import Fmi2Slave, Fmi2Type
from fmuloader.fmi3 import Fmi3Slave

# FMI 2.0 reference FMUs don't ship aarch64-darwin binaries
skip_arm64 = pytest.mark.skipif(
    platform.machine() == "arm64",
    reason="FMI 2.0 reference FMU binaries not available on ARM64",
)


class TestReadmeFmi2CoSimulation:
    """README § FMI 2.0 Co-Simulation."""

    @skip_arm64
    def test_fmi2_co_simulation(self, reference_fmus_dir):
        # -- mirrors README example: FMI 2.0 Co-Simulation --
        from fmuloader.fmi2 import Fmi2Slave, Fmi2Type

        slave = Fmi2Slave(
            reference_fmus_dir / "2.0/BouncingBall.fmu",
            model_identifier="BouncingBall",
        )

        slave.instantiate(
            "instance1",
            Fmi2Type.CO_SIMULATION,
            guid="{1AE5E10D-9521-4DE3-80B9-D0EAAA7D5AF1}",
        )
        slave.setup_experiment(start_time=0.0, stop_time=10.0)
        slave.enter_initialization_mode()
        slave.exit_initialization_mode()

        t, dt = 0.0, 0.01
        while t < 1.0:  # shortened from 10.0 for speed
            slave.do_step(t, dt)
            t += dt

        values = slave.get_real([1, 3])
        assert len(values) == 2
        slave.terminate()
        slave.free_instance()


class TestReadmeFmi3CoSimulation:
    """README § FMI 3.0 Co-Simulation."""

    def test_fmi3_co_simulation(self, reference_fmus_dir):
        # -- mirrors README example: FMI 3.0 Co-Simulation --
        from fmuloader.fmi3 import Fmi3Slave

        slave = Fmi3Slave(
            reference_fmus_dir / "3.0/BouncingBall.fmu",
            model_identifier="BouncingBall",
        )

        slave.instantiate_co_simulation(
            "instance1",
            instantiation_token="{1AE5E10D-9521-4DE3-80B9-D0EAAA7D5AF1}",
        )
        slave.enter_initialization_mode(start_time=0.0, stop_time=10.0)
        slave.exit_initialization_mode()

        t, dt = 0.0, 0.01
        while t < 1.0:  # shortened from 10.0 for speed
            result = slave.do_step(t, dt)
            t += dt

        values = slave.get_float64([1, 3])
        assert len(values) == 2
        slave.terminate()
        slave.free_instance()


class TestReadmeFmi3ModelExchange:
    """README § FMI 3.0 Model Exchange."""

    def test_fmi3_model_exchange(self, reference_fmus_dir):
        # -- mirrors README example: FMI 3.0 Model Exchange --
        from fmuloader.fmi3 import Fmi3Slave

        slave = Fmi3Slave(
            reference_fmus_dir / "3.0/VanDerPol.fmu",
            model_identifier="VanDerPol",
        )

        slave.instantiate_model_exchange(
            "instance1",
            instantiation_token="{BD403596-3166-4232-ABC2-132BDF73E644}",
        )
        slave.enter_initialization_mode(start_time=0.0)
        slave.exit_initialization_mode()

        result = slave.update_discrete_states()
        while result.discrete_states_need_update:
            result = slave.update_discrete_states()
        slave.enter_continuous_time_mode()

        nx = slave.get_number_of_continuous_states()
        assert nx == 2
        slave.set_time(0.0)
        derivs = slave.get_continuous_state_derivatives(nx)
        assert len(derivs) == nx

        slave.terminate()
        slave.free_instance()


class TestReadmeFmi3ScheduledExecution:
    """README § FMI 3.0 Scheduled Execution."""

    def test_fmi3_scheduled_execution(self, reference_fmus_dir):
        # -- mirrors README example: FMI 3.0 Scheduled Execution --
        from fmuloader.fmi3 import Fmi3Slave

        slave = Fmi3Slave(
            reference_fmus_dir / "3.0/Clocks.fmu",
            model_identifier="Clocks",
        )

        slave.instantiate_scheduled_execution(
            "instance1",
            instantiation_token="{C5F142BA-B849-42DA-B4A1-4745BFF3BE28}",
        )
        slave.enter_initialization_mode(start_time=0.0)
        slave.exit_initialization_mode()

        slave.activate_model_partition(clock_reference=1001, activation_time=0.0)

        slave.terminate()
        slave.free_instance()
