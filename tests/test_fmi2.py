import pytest
from fmuloader.fmi2 import Fmi2Slave
import pathlib


@pytest.mark.parametrize(
    "reference_fmu",
    [
        "2.0/BouncingBall.fmu",
        "2.0/VanDerPol.fmu",
        "2.0/Dahlquist.fmu",
        "2.0/Stair.fmu",
        "2.0/Resource.fmu",
    ],
)
def test_metadata(reference_fmu, reference_fmus_dir):
    filename = (reference_fmus_dir / reference_fmu).absolute()

    slave = Fmi2Slave(filename, model_identifier=pathlib.Path(filename).stem)
    
    assert slave.get_version() == "2.0"
