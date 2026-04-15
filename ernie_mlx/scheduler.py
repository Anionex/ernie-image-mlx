"""FlowMatch Euler Discrete Scheduler for MLX."""
import mlx.core as mx
import numpy as np

from .config import SchedulerConfig


class FlowMatchEulerScheduler:
    def __init__(self, config: SchedulerConfig = SchedulerConfig()):
        self.config = config
        self.sigmas = None
        self.timesteps = None

    def set_timesteps(self, num_inference_steps: int):
        sigmas = np.linspace(1.0, 0.0, num_inference_steps + 1)
        sigmas = mx.array(sigmas, dtype=mx.float32)

        # Apply shift: sigma' = shift * sigma / (1 + (shift - 1) * sigma)
        shift = self.config.shift
        sigmas = shift * sigmas / (1 + (shift - 1) * sigmas)

        self.sigmas = sigmas
        # Timesteps are sigmas * num_train_timesteps (continuous)
        self.timesteps = sigmas[:-1] * self.config.num_train_timesteps

    def step(self, model_output: mx.array, step_index: int, sample: mx.array) -> mx.array:
        sigma = self.sigmas[step_index]
        sigma_next = self.sigmas[step_index + 1]
        dt = sigma_next - sigma
        prev_sample = sample + dt * model_output
        return prev_sample
