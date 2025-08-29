
import numpy as np
from nptyping import NDArray, Shape, Float


def decay_schedule(
        init_value: float,
        min_value: float,
        decay_ratio: float,
        max_steps: int,
        log_start: int=-2,
        log_base: int=10
) -> NDArray[Shape["*"], Float]:
    decay_steps = int(max_steps*decay_ratio)
    rem_steps = max_steps - decay_steps

    values = np.logspace(log_start, 0, decay_steps, base=log_base, endpoint=True)[::-1]
    values = (values - values.min()) / (values.max() - values.min())
    values = (init_value - min_value) * values + min_value
    values = np.pad(values, (0, rem_steps), 'edge')
    return values
