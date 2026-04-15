"""
Modified Walker2d environments for cross-task generalization experiments.

Each wrapper modifies the physical dynamics of Walker2d-v3 after construction
by directly editing the MuJoCo model parameters. The observation and action
spaces remain identical so that pre-trained DT weights are compatible.
"""

import gymnasium as gym
import numpy as np


class ModifiedWalker2dEnv(gym.Wrapper):
    """Base wrapper that modifies Walker2d dynamics after env creation."""

    def __init__(self, mass_scale=1.0, friction_scale=1.0):
        env = gym.make("Walker2d-v4")
        super().__init__(env)
        self.mass_scale = mass_scale
        self.friction_scale = friction_scale

        # Store original values for reference
        self._original_mass = self.unwrapped.model.body_mass.copy()
        self._original_friction = self.unwrapped.model.geom_friction.copy()

        # Apply modifications
        self.unwrapped.model.body_mass[:] = self._original_mass * mass_scale
        self.unwrapped.model.geom_friction[:] = self._original_friction * friction_scale

    def reset(self, **kwargs):
        obs, info = super().reset(**kwargs)
        # Re-apply modifications after reset (some envs reset model params)
        self.unwrapped.model.body_mass[:] = self._original_mass * self.mass_scale
        self.unwrapped.model.geom_friction[:] = self._original_friction * self.friction_scale
        return obs, info

    def __repr__(self):
        return (f"ModifiedWalker2d(mass_scale={self.mass_scale}, "
                f"friction_scale={self.friction_scale})")


# =============================================================================
# Pre-defined target domain configurations
# =============================================================================

TARGET_CONFIGS = {
    "heavy": {"mass_scale": 1.5, "friction_scale": 1.0},
    "light": {"mass_scale": 0.5, "friction_scale": 1.0},
    "slippery": {"mass_scale": 1.0, "friction_scale": 0.5},
    "heavy_slippery": {"mass_scale": 1.5, "friction_scale": 0.5},
}


def make_modified_walker2d(config_name):
    """Create a modified Walker2d env from a named config."""
    if config_name == "source":
        return gym.make("Walker2d-v4")
    cfg = TARGET_CONFIGS[config_name]
    return ModifiedWalker2dEnv(**cfg)


def get_all_target_names():
    return list(TARGET_CONFIGS.keys())


if __name__ == "__main__":
    # Quick sanity check
    for name, cfg in TARGET_CONFIGS.items():
        env = ModifiedWalker2dEnv(**cfg)
        obs, info = env.reset()
        print(f"{name}: obs shape={obs.shape}, "
              f"mass[1]={env.unwrapped.model.body_mass[1]:.2f}, "
              f"friction[0,0]={env.unwrapped.model.geom_friction[0,0]:.4f}")
        for _ in range(10):
            obs, r, terminated, truncated, info = env.step(env.action_space.sample())
            d = terminated or truncated
        print(f"  10-step random reward sum: {r:.4f}, done={d}")
        env.close()
    print("All environments OK.")
