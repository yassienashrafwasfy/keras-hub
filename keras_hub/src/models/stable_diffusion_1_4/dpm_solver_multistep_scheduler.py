import numpy as np
from keras import layers
from keras import ops


class DPMSolverMultistepScheduler(layers.Layer):
    """DPM-Solver++ (2M) multistep scheduler with Karras sigmas.

    This scheduler implements the second-order multistep DPM-Solver++ sampler
    used by Stable Diffusion v1.4. The model is trained to predict noise
    (`epsilon`), which this scheduler converts to a data (`x0`) prediction
    before taking a solver step. With `use_karras_sigmas=True`, the inference
    noise levels follow the Karras et al. schedule.

    Because the second-order update depends on the previous step's data
    prediction, the caller must carry that prediction across steps. The first
    and last inference steps fall back to the first-order update.

    Args:
        num_train_timesteps: int. The number of diffusion steps used to train
            the model.
        beta_start: float. The starting `beta` value.
        beta_end: float. The final `beta` value.
        beta_schedule: str. The schedule used to interpolate betas. Only
            `"scaled_linear"` is supported.
        solver_order: int. The order of the solver. Only `2` is supported.
        use_karras_sigmas: bool. Whether to use the Karras sigma schedule.
        **kwargs: other keyword arguments passed to `keras.layers.Layer`,
            including `name`, `dtype` etc.

    References:
    - [DPM-Solver++](https://arxiv.org/abs/2211.01095).
    - [Elucidating the Design Space of Diffusion-Based Generative Models](
    https://arxiv.org/abs/2206.00364).
    """

    def __init__(
        self,
        num_train_timesteps=1000,
        beta_start=0.00085,
        beta_end=0.012,
        beta_schedule="scaled_linear",
        solver_order=2,
        use_karras_sigmas=True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        if beta_schedule != "scaled_linear":
            raise NotImplementedError(
                "Only `beta_schedule='scaled_linear'` is supported. "
                f"Received: beta_schedule={beta_schedule}"
            )
        if solver_order != 2:
            raise NotImplementedError(
                "Only `solver_order=2` is supported. "
                f"Received: solver_order={solver_order}"
            )
        self.num_train_timesteps = int(num_train_timesteps)
        self.beta_start = float(beta_start)
        self.beta_end = float(beta_end)
        self.beta_schedule = beta_schedule
        self.solver_order = int(solver_order)
        self.use_karras_sigmas = bool(use_karras_sigmas)

        # `scaled_linear` betas and the resulting variance-preserving schedule.
        betas = (
            ops.linspace(
                beta_start**0.5,
                beta_end**0.5,
                self.num_train_timesteps,
                dtype="float32",
            )
            ** 2
        )
        alphas_cumprod = ops.cumprod(ops.subtract(1.0, betas))
        # `sigma` here is the k-diffusion noise level: `sqrt((1 - a) / a)`.
        train_sigmas = ops.sqrt(
            ops.divide(ops.subtract(1.0, alphas_cumprod), alphas_cumprod)
        )
        train_sigmas = ops.convert_to_numpy(train_sigmas)
        self._sigma_min = float(train_sigmas[0])
        self._sigma_max = float(train_sigmas[-1])
        self._log_train_sigmas = np.log(train_sigmas)

        # Build a default schedule so the functional model can be traced.
        self.set_timesteps(25)

    def _sigma_to_timestep(self, sigma):
        # Map noise levels back to continuous training timesteps by
        # interpolating against the log of the training sigmas, which are
        # monotonically increasing in the training timestep. This is eager
        # metadata used for validation, so it is computed with NumPy.
        log_sigma = np.log(ops.convert_to_numpy(sigma))
        indices = np.arange(self.num_train_timesteps, dtype="float32")
        return np.interp(log_sigma, self._log_train_sigmas, indices)

    def set_timesteps(self, num_steps):
        """Precompute the per-step noise levels for `num_steps` inference steps.

        Builds, for steps `0 .. num_steps`, the variance-preserving `alpha_t`,
        `sigma_t` and `lambda_t` arrays. The final entry is the sentinel for
        zero noise (`sigma = 0`).
        """
        num_steps = int(num_steps)
        self.num_inference_steps = num_steps

        rho = 7.0
        ramp = ops.linspace(0.0, 1.0, num_steps, dtype="float32")
        min_inv_rho = self._sigma_min ** (1.0 / rho)
        max_inv_rho = self._sigma_max ** (1.0 / rho)
        sigmas = (max_inv_rho + ramp * (min_inv_rho - max_inv_rho)) ** rho
        # `sigmas` is descending (high noise first). Append the zero sentinel.
        sigmas = ops.concatenate(
            [sigmas, ops.zeros((1,), dtype="float32")], axis=0
        )
        self.sigmas = sigmas
        # The diffusion timesteps fed to the UNet, indexable inside the loop.
        self.timesteps = ops.convert_to_tensor(
            self._sigma_to_timestep(sigmas[:-1]), dtype="float32"
        )

        # Variance-preserving conversion from the k-diffusion `sigma`:
        # `alpha_t = 1 / sqrt(1 + sigma**2)`, `sigma_t = sigma * alpha_t`.
        alpha_t = ops.divide(1.0, ops.sqrt(ops.add(1.0, ops.square(sigmas))))
        sigma_t = ops.multiply(sigmas, alpha_t)
        self.alpha_t = alpha_t
        self.sigma_t = sigma_t
        # `lambda_t = log(alpha_t) - log(sigma_t) = -log(sigma)`. The sentinel
        # is `+inf`; it is only read on lower-order steps where it is unused.
        self.lambda_t = ops.subtract(ops.log(alpha_t), ops.log(sigma_t))

    def convert_model_output(self, model_output, step, sample):
        """Convert an `epsilon` prediction to a data (`x0`) prediction."""
        alpha_t = ops.take(self.alpha_t, step)
        sigma_t = ops.take(self.sigma_t, step)
        return ops.divide(
            ops.subtract(sample, ops.multiply(sigma_t, model_output)), alpha_t
        )

    def step(self, x0, prev_x0, step, sample):
        """Take one DPM-Solver++ (2M) step.

        Args:
            x0: The data prediction at the current step.
            prev_x0: The data prediction at the previous step.
            step: The index of the current inference step.
            sample: The current latent sample.

        Returns:
            The latent sample for the next step.
        """
        step = ops.convert_to_tensor(step, dtype="int32")
        next_step = ops.add(step, 1)
        prev_step = ops.subtract(step, 1)

        alpha_t = ops.take(self.alpha_t, step)
        alpha_s = ops.take(self.alpha_t, next_step)
        sigma_t = ops.take(self.sigma_t, step)
        sigma_s = ops.take(self.sigma_t, next_step)
        lambda_t = ops.take(self.lambda_t, step)
        lambda_s = ops.take(self.lambda_t, next_step)
        lambda_p = ops.take(self.lambda_t, prev_step)

        # `exp(-h)` written from alphas/sigmas so the zero-noise sentinel stays
        # finite (it evaluates to `0`).
        exp_neg_h = ops.divide(
            ops.multiply(alpha_t, sigma_s), ops.multiply(sigma_t, alpha_s)
        )
        sigma_ratio = ops.divide(sigma_s, sigma_t)

        first_order = ops.subtract(
            ops.multiply(sigma_ratio, sample),
            ops.multiply(
                alpha_s, ops.multiply(ops.subtract(exp_neg_h, 1.0), x0)
            ),
        )

        h = ops.subtract(lambda_s, lambda_t)
        h_last = ops.subtract(lambda_t, lambda_p)
        r = ops.divide(h_last, h)
        inv_2r = ops.divide(1.0, ops.multiply(2.0, r))
        d = ops.subtract(
            ops.multiply(ops.add(1.0, inv_2r), x0),
            ops.multiply(inv_2r, prev_x0),
        )
        second_order = ops.subtract(
            ops.multiply(sigma_ratio, sample),
            ops.multiply(
                alpha_s, ops.multiply(ops.subtract(exp_neg_h, 1.0), d)
            ),
        )

        # The first and last steps fall back to the first-order update.
        lower_order = ops.logical_or(
            ops.equal(step, 0),
            ops.equal(step, self.num_inference_steps - 1),
        )
        return ops.where(lower_order, first_order, second_order)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "num_train_timesteps": self.num_train_timesteps,
                "beta_start": self.beta_start,
                "beta_end": self.beta_end,
                "beta_schedule": self.beta_schedule,
                "solver_order": self.solver_order,
                "use_karras_sigmas": self.use_karras_sigmas,
            }
        )
        return config
