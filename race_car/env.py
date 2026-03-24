"""Defines the environment for simulating the race car system."""

import gymnasium as gym
import numpy as np

from race_car.model import export_acados_integrator
from race_car.utils.config import CarParams, get_default_car_params
from race_car.utils.scaling import get_transformation_matrices
from race_car.utils.track import get_track


class RaceCarEnv(gym.Env):

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(
        self,
        render_mode: str | None = None,
        car_params: CarParams | None = None,
        dimensionless: bool = True,
        mpc_car_params: CarParams | None = None,
    ):
        if car_params is None or mpc_car_params is None:
            mpc_car_params = car_params = get_default_car_params()
            print("Car parameters not provided. Using default parameters.")
            # raise ValueError("Car parameters not provided in the env.")

        self.car_params = car_params
        self.dimensionless = dimensionless  # whether to use the dimensionless formulation

        self.nx = 4  # number of states
        self.nu = 2  # number of inputs            

        # integrator for simulating the dynamics (always dimensional)
        self.integrator = export_acados_integrator(car_params=car_params, dimensionless=False)

        # state and action bounds
        large_number = 1e3  # vehicle states unbounded for now (going offroad is acceptable during learning)
        state_bounds = large_number * np.ones((self.nx,), dtype=np.float32)  # just the original states
        obs_ub = np.concatenate(( state_bounds, np.array([car_params.D_max.item(), car_params.delta_max.item()], dtype=np.float32)))
        obs_lb = np.concatenate((-state_bounds, np.array([car_params.D_min.item(), car_params.delta_min.item()], dtype=np.float32)))
        act_ub = np.array([car_params.dD_max.item(), car_params.ddelta_max.item()], dtype=np.float32)
        act_lb = np.array([car_params.dD_min.item(), car_params.ddelta_min.item()], dtype=np.float32)

        # add a 5% margin to the bounds (prevents episode truncation due to numerical issues)
        obs_ub *= 1.05
        obs_lb *= 1.05
        act_ub *= 1.05
        act_lb *= 1.05

        if dimensionless:
            # use agent instead of env parameters (allow for mismatches)
            Ms, Ma, _ = get_transformation_matrices(
                mpc_car_params
            )  # s(physical) = Ms * s(dimensionless)

            # remove the control input rate from the variable transformation
            # self.Ma = self.Ms[self.nx:, self.nx:]  # only inputs
            # self.Ms = self.Ms[:self.nx, :self.nx]  # only states

            Ms_inv = np.linalg.inv(Ms)
            Ma_inv = np.linalg.inv(Ma)

            # scale the state and action bounds
            # obs_ub = self.Ms_inv @ obs_ub  # needed if the vehicle states are bounded or inputs dimensional
            # obs_lb = self.Ms_inv @ obs_lb  # needed if the vehicle states are bounded or inputs dimensional
            act_ub = Ma_inv @ act_ub
            act_lb = Ma_inv @ act_lb

            # define helper functions
            self.dim2nondim_s = lambda s: Ms_inv[:self.nx, :self.nx] @ s
            self.nondim2dim_s = lambda s: Ms[:self.nx, :self.nx] @ s
            self.dim2nondim_a = lambda a: Ma_inv @ a
            self.nondim2dim_a = lambda a: Ma @ a

        self.time_scale = get_transformation_matrices(mpc_car_params)[2].item()  # for scaling the time in the reward

        self.action_space = gym.spaces.Box(np.float32(act_lb), np.float32(act_ub))
        self.observation_space = gym.spaces.Box(np.float32(obs_lb), np.float32(obs_ub))

        self.reset_needed = True
        self.s = None  # physical state
        self.previous_action = None

        self.s_max = get_track(car_params)[0][-1]  # [m] total length of the track
        self.dt = car_params.dt.item()  # [s] time step for the simulation

        # For rendering
        if not (render_mode is None or render_mode in self.metadata["render_modes"]):
            raise ValueError(
                f"render_mode must be one of {self.metadata['render_modes']}"
            )
        self.render_mode = render_mode
        self.screen_width = 600
        self.screen_height = 400
        self.window = None
        self.clock = None


    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict]:
        """Execute the dynamics of the race car."""
        if self.reset_needed:
            raise RuntimeError("Call reset before using the step method.")
        
        # scale the MPC output back if dimensionless
        if self.dimensionless:
            action = self.nondim2dim_a(action)

        # MPC outputs the control rate, integrate
        action = self.previous_action + self.dt * action

        # simulate one time step
        s_prev = self.s
        self.s = self.integrator.simulate(x=self.s, u=action)
        if self.render_mode is not None:
            self.state_log.append(self.s)  # save trajectory for plotting
        self.previous_action = action  # this is already dimensionless
        self.total_time += self.dt

        # scale the observation if needed
        obs = self.dim2nondim_s(self.s) if self.dimensionless else self.s

        # concatenate the action into observations (MPC uses control rates as inputs)
        obs = np.concatenate([obs, action])

        # calculate the reward (TODO: check the reward/penalty formulation)
        # reward speed in the right direction
        # r = s_prev[3] * np.cos(s_prev[2])  # velocity * cos(heading error)

        # progress per time step (dimensionless)
        # r = (self.s[0] - s_prev[0]) / self.car_params.l.item() * self.time_scale / self.dt

        # combination of relative progress and time
        r = self.s[0] / self.s_max - 50 * self.dt / self.time_scale

        # terminate in minimum time
        # r = -self.dt / self.time_scale

        # progress along the track (relative to car length)
        # r = (self.s[0] - s_prev[0]) / self.car_params.l.item()

        # check for termination (reaching the end of the track)
        # TODO: add total race time info
        if self.s[0] > self.s_max:
            term = True
        else:
            term = False

        # check for truncation (out of bounds or timeout)
        out_of_bounds = not self.observation_space.contains(np.array(obs, dtype=np.float32))
        timeout = self.total_time > 500 * self.dt
        if out_of_bounds or timeout:
            trunc = True
            r -= 100.0
        else:
            trunc = False
        
        self.reset_needed = trunc or term
        info = {}

        return obs, r, term, trunc, info


    def reset(
        self, *, seed: int | None = None, options: dict | None = None
    ) -> tuple[np.ndarray, dict]:  # type: ignore
        if seed is not None:
            super().reset(seed=seed)
            self.observation_space.seed(seed)
            self.action_space.seed(seed)

        # start from standstill at the beginning of the track
        s0 = 0.0  # [m]
        self.s = np.array([s0, 0.0, 0.0, 0.0])
        self.previous_action = np.zeros((self.nu,))
        self.state_log = []
        self.total_time = 0.0

        # nondimensionalize if needed
        if self.dimensionless:
            self.s = self.dim2nondim_s(self.s)
            self.previous_action = self.dim2nondim_a(self.previous_action)

        self.reset_needed = False        

        obs = np.concatenate([self.s, self.previous_action])  # MPC uses control rates as inputs
        return obs, {}


    def render(self):
        import pygame
        import matplotlib
        matplotlib.use('Agg')  # Non-interactive backend — must be set before importing pyplot
        import matplotlib.pyplot as plt
        import matplotlib.cm as cm
        import io

        from race_car.utils.plotting import frenet2global, get_track

        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((self.screen_width, self.screen_height))
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        if not self.state_log:
            if self.render_mode == "human":
                return None
            else:  # rgb_array — must return a valid frame
                return np.zeros((self.screen_height, self.screen_width, 3), dtype=np.uint8)
        else:
            state_log = np.array(self.state_log)

        # --- Run your matplotlib plotting logic ---
        s = state_log[:, 0]
        n = state_log[:, 1]
        alpha = state_log[:, 2]
        v = state_log[:, 3]

        distance = 0.12 * self.car_params.l.item() / get_default_car_params().l.item()
        [x, y, _, _] = frenet2global(s=s, n=n, alpha=alpha, v=v, car_params=self.car_params)
        [_, Xref, Yref, Psiref, _] = get_track(self.car_params)

        fig, ax = plt.subplots(figsize=(self.screen_width / 100, self.screen_height / 100), dpi=100)

        ax.set_ylim(np.min(Yref) - 5 * self.car_params.l.item(), np.max(Yref) + 5 * self.car_params.l.item())
        ax.set_xlim(np.min(Xref) - 5 * self.car_params.l.item(), np.max(Xref) + 5 * self.car_params.l.item())
        ax.set_ylabel('$y$ [m]')
        ax.set_xlabel('$x$ [m]')
        ax.set_aspect('equal', 'box')

        if self.total_time:
            ax.set_title("Lap time: {0:.2f} s".format(self.total_time))

        ax.plot(Xref, Yref, '--', color='k')

        Xboundleft  = Xref - distance * np.sin(Psiref)
        Yboundleft  = Yref + distance * np.cos(Psiref)
        Xboundright = Xref + distance * np.sin(Psiref)
        Yboundright = Yref - distance * np.cos(Psiref)
        ax.plot(Xboundleft,  Yboundleft,  color='k', linewidth=1)
        ax.plot(Xboundright, Yboundright, color='k', linewidth=1)
        ax.plot(x, y, '-b')

        heatmap = ax.scatter(x, y, c=v, cmap=cm.rainbow, edgecolor='none', marker='o')
        cbar = fig.colorbar(heatmap, ax=ax, fraction=0.035)
        cbar.set_label("velocity in [m/s]")

        # --- Convert matplotlib figure → numpy array → pygame surface ---
        buf = io.BytesIO()
        fig.savefig(buf, format='raw', dpi=100)
        buf.seek(0)
        raw_data = np.frombuffer(buf.getvalue(), dtype=np.uint8)
        raw_data = raw_data.reshape((int(fig.get_figheight() * 100), int(fig.get_figwidth() * 100), 4))
        plt.close(fig)  # Important: free memory

        # RGBA → RGB, then into a pygame surface matching screen size
        rgb_data = raw_data[:, :, :3]
        canvas = pygame.surfarray.make_surface(np.transpose(rgb_data, (1, 0, 2)))
        canvas = pygame.transform.scale(canvas, (self.screen_width, self.screen_height))

        if self.render_mode == "human":
            self.window.blit(canvas, (0, 0))
            pygame.event.pump()
            pygame.display.update()
            self.clock.tick(self.metadata["render_fps"])
        else:  # rgb_array
            return np.transpose(np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2))


    def close(self):
        if self.window is not None:
            import pygame

            pygame.display.quit()
            pygame.quit()


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    from race_car.utils.config import get_large_car_params
    dimensionless = True

    # create envs for the default and similar parameters
    params_ref = get_default_car_params()
    env_ref = RaceCarEnv(
        car_params=params_ref,
        dimensionless=dimensionless,
        mpc_car_params=params_ref
    )

    params_sim = get_large_car_params()
    env_sim = RaceCarEnv(
        car_params=params_sim,
        dimensionless=dimensionless,
        mpc_car_params=params_sim
    )

    seed = 0
    obs_ref = env_ref.reset(seed=seed)[0]
    obs_sim = env_sim.reset(seed=seed)[0]

    if dimensionless:
        assert env_ref.action_space == env_sim.action_space
        assert env_ref.observation_space == env_sim.observation_space
        assert np.allclose(obs_ref, obs_sim)

    # compare the two envs with random actions
    diffs = []
    obs_ref_log = []
    obs_sim_log = []
    obs_ref_log.append(obs_ref)
    obs_sim_log.append(obs_sim)
    act_log = []
    for _ in range(10):
        action = env_ref.action_space.sample()
        obs_ref, reward_ref, done_ref, truncated_ref, info_ref = env_ref.step(action)
        obs_sim, reward_sim, done_sim, truncated_sim, info_sim = env_sim.step(action)

        if done_ref or done_sim or truncated_ref or truncated_sim:
            seed += 1
            env_ref.reset(seed=seed)
            env_sim.reset(seed=seed)
            print("reset")

        obs_ref_log.append(obs_ref)
        obs_sim_log.append(obs_sim)
        diffs.append(np.max(np.abs(obs_ref - obs_sim)))
        act_log.append(action)
    print(f"max diff: {np.max(diffs)}")

    # plot the results
    obs_ref_log = np.array(obs_ref_log)
    obs_sim_log = np.array(obs_sim_log)
    act_log = np.array(act_log)
    nx = 4
    nu = 2
    fig, ax = plt.subplots(nx + nu, 1, sharex=True)
    labels = ["ref", "sim"]
    for i in range(nx):
        ax[i].plot(obs_ref_log[:, i], color="b", label=labels[0])
        ax[i].plot(obs_sim_log[:, i], color="r", linestyle="--", label=labels[1])
        ax[i].grid()
        ax[i].set_ylabel(env_ref.integrator.acados_sim.model.x_labels[i])
    ax[0].legend()
    for i in range(nu):
        ax[nx+i].step(
            list(range(obs_ref_log.shape[0])),
            np.append([act_log[0, i]], act_log[:, i]),
            where="post",
            color="b",
        )
        ax[nx+i].set_ylabel(env_ref.integrator.acados_sim.model.u_labels[i])
        ax[nx+i].grid()
    ax[-1].set_xlabel("$k$")

    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, hspace=0.4)
    fig.align_ylabels(ax)
    plt.show(block=False)
    print("ok")
    print("Press ENTER to close the plot")
    input()
    plt.close()

    env_ref.close()
    env_sim.close()

