from math import inf
from pathlib import Path
from typing import Any

import gymnasium as gym
import newton
import numpy as np
import warp as wp
from gymnasium.core import RenderFrame
from newton.sensors import SensorIMU
from numpy.typing import NDArray

from ..robot import LwmrRobotConfig, add_lwmr_robot
from .utils import create_viewer_viser, quat_to_rpy, world_to_body

ObsType = NDArray
InfoType = dict[str, Any]
OptType = dict[str, Any]
WaypointType = NDArray


class LwmrPlaneEnv(gym.Env):
    metadata = {"render_modes": ["viser", "none"], "render_fps": 60}

    def __init__(
        self,
        # TODO: this is only for compatibility with rllib
        config=None,
        *,
        robot_config: LwmrRobotConfig = LwmrRobotConfig(),
        waypoints: list[WaypointType] | None = None,
        solver_name: str = "MuJoCo",
        add_step: bool = False,
        n_lookahead: int = 3,
        hit_radius: float = 0.01,
        world_bound: float = 20.0,
        max_steps: int = 256,
        sim_freq: int = 240,
        control_freq: int = 5,
        frame_freq: int | None = None,
        num_worlds: int = 1,
        device: str = "cuda",
        quiet: bool = False,
        render_mode: str = "none",
        max_viewer_worlds: int = 16,
        validate: bool = False,
        fixed_base: bool = False,
        viewer_port: int = 8080,
        viewer_spacing: float = 0.8,
        viewer_output_path: str = "./recordings/lwmr_plane.viser",
    ):
        super().__init__()

        default_waypoints = [
            np.array([0.5, 0.0]),
            np.array([0.5, 0.5]),
            np.array([1.0, 0.5]),
        ]

        # TODO: consider validating arguments
        self.robot_config = robot_config
        self.waypoints = waypoints if waypoints is not None else default_waypoints
        self.solver_name = solver_name
        self.add_step = add_step
        self.n_lookahead = n_lookahead
        self.hit_radius = hit_radius
        self.world_bound = world_bound
        self.max_steps = max_steps
        self.sim_freq = sim_freq
        self.control_freq = control_freq
        self.frame_freq = frame_freq
        self.num_worlds = num_worlds
        self.device = device if device == "cuda" and wp.get_device().is_cuda else "cpu"
        self.quiet = quiet
        self.render_mode = render_mode
        self.max_viewer_worlds = max_viewer_worlds
        self.validate = validate
        self.fixed_base = fixed_base
        self.viewer_port = viewer_port
        self.viewer_spacing = viewer_spacing
        self.viewer_output_path = viewer_output_path
        self._start_simulation()

    def _start_simulation(self):

        self.steps = 0

        # TODO: clone? per world?
        self.waypoint_index = 0

        # Set global quiet mode for Warp before newton is initialized in the environment
        wp.config.quiet = self.quiet

        # `frame_freq` is related to both `step()` and `render()`
        frame_freq = self.frame_freq if self.frame_freq is not None else self.metadata["render_fps"]

        assert frame_freq
        assert self.sim_freq % frame_freq == 0, "`sim_freq` must be a multiple of `frame_freq`"
        self.sim_steps_per_frame = self.sim_freq // frame_freq

        assert self.sim_freq % self.control_freq == 0, "`sim_freq` must be a multiple of `control_freq`"
        self.sim_steps_per_control = self.sim_freq // self.control_freq

        self.sim_time = 0.0
        self.frame_dt = 1.0 / frame_freq
        self.sim_dt = 1.0 / self.sim_freq
        self.control_steps_counter = 0

        drop_height = self.robot_config.wh_radius + 0.05

        #
        # region World
        # Default world -1; collides with all worlds
        # Use body=-1 to attach shapes to the static world frame
        #

        builder = newton.ModelBuilder()

        # TODO: explore ground characteristics (friction, restitution, etc in ShapeConfig)
        # TODO: make environment configuration (e.g., initial conditions, randomization parameters) configurable via kwargs
        # # Set defaults before adding shapes
        # builder.default_shape_cfg.ke = 1.0e6
        # builder.default_shape_cfg.kd = 1000.0
        # builder.default_shape_cfg.mu = 0.5
        # builder.default_shape_cfg.is_hydroelastic = True
        # builder.default_shape_cfg.sdf_max_resolution = 64  # Primitive SDF defaults

        builder.add_ground_plane()

        #
        # region Env
        # Create shared world and ground plane
        #

        builder.validate_inertia_detailed = self.validate

        initial_xform = wp.transform(p=(0.0, 0.0, drop_height))

        robot_builder = newton.ModelBuilder()

        self.chassis, _, _, _ = add_lwmr_robot(
            robot_builder,
            initial_xform,
            self.robot_config,
            fixed_base=self.fixed_base,
        )

        assert not self.robot_config.add_imu, "IMU is currently not supported."
        if self.robot_config.add_imu:
            # Add an imu at the chassis center
            robot_builder.add_site(body=self.chassis, label="imu")

        assert not self.robot_config.add_camera, "Camera is currently not supported."
        if self.robot_config.add_camera:
            robot_builder.add_site(
                body=self.chassis,
                xform=wp.transform(
                    p=wp.vec3(0.5, 0, 0.2),
                    q=wp.quat_from_axis_angle(wp.vec3(0, 1, 0), 3.14159 / 4),  # type: ignore
                ),
                type=newton.GeoType.BOX,
                scale=(0.05, 0.05, 0.02),
                visible=True,
                label="camera",
            )

        assert self.num_worlds == 1, "Multiple worlds are currently not supported."
        for _ in range(self.num_worlds):
            builder.begin_world()

            if self.add_step:
                # Add a step to the world for the robot to drive over
                hz = (self.robot_config.wh_radius / 2.7) * np.random.uniform(0.8, 1.2)
                pos = (0.5, 0.0, hz)
                rot = wp.quat_rpy(0.0, 0.0, np.random.uniform(-0.2, 0.2))

                builder.add_shape_box(
                    body=-1,
                    hx=0.1,
                    hy=0.5,
                    hz=hz,
                    xform=wp.transform(p=pos, q=rot),
                    color=(0.5, 0.5, 0.5),
                )

            builder.add_builder(robot_builder)

            builder.end_world()

        # NOTE: duplicate worlds can be created more simply with:
        # builder.replicate(robot_builder, ...)

        self.model = builder.finalize(device=self.device)

        if self.robot_config.add_imu:
            self.imu = SensorIMU(self.model, sites="imu")

        #
        # region State
        #

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()
        self.contacts = self.model.contacts()

        # NOTE: using stateless actuators
        self.actuators = self.model.actuators
        assert self.control.joint_target_vel
        self.actuator_indices = self.actuators[0].indices.numpy()
        self.actuation_values = self.control.joint_target_vel.numpy()

        #
        # region Solver
        #

        # TODO: support different solvers and configurations (e.g., iterations, tolerance, etc)
        assert self.solver_name == "MuJoCo", f"Unsupported solver: {self.solver_name}"
        # using_generalized_coordinates = solver_name in ["MuJoCo", "Featherstone"]

        self.solver = newton.solvers.SolverMuJoCo(self.model, iterations=100, ls_iterations=50, njmax=100)

        assert self.state_0.joint_q and self.state_0.joint_qd
        newton.eval_ik(self.model, self.state_0, self.state_0.joint_q, self.state_0.joint_qd)

        assert self.state_0.joint_q and self.state_0.joint_qd
        newton.eval_fk(self.model, self.state_0.joint_q, self.state_0.joint_qd, self.state_0)

        # Capture the simulation as a CUDA graph (if running on GPU)
        if self.model.device.is_cuda and wp.get_device().is_cuda:
            with wp.ScopedCapture() as capture:
                self._simulate()
            self.graph = capture.graph
        else:
            self.graph = None

        #
        # region Viewer
        #

        viewer = None

        render_mode = self.render_mode.lower() if self.render_mode else "none"
        assert render_mode in self.metadata["render_modes"], f"Unsupported render mode: {render_mode}"

        if render_mode == "viser":
            # TODO: log warning if `viewer_output_path` already exists and will be overwritten
            recording_path = Path(self.viewer_output_path).resolve()
            recording_path.parent.mkdir(parents=True, exist_ok=True)

            viewer = create_viewer_viser(str(recording_path), quiet=self.quiet, port=self.viewer_port)

            max_viewer_worlds = min(self.model.world_count, self.max_viewer_worlds)
            viewer.set_model(self.model, max_worlds=max_viewer_worlds)
            viewer.set_world_offsets(spacing=(self.viewer_spacing, self.viewer_spacing, 0.0))

            # Set the initial camera pose (this is a bit of a workaround)
            viewer._server.initial_camera.position = (-0.748, -0.626, 0.576)
            viewer._server.initial_camera.look_at = (0.000, 0.000, 0.000)
            viewer._server.initial_camera.up = (0.000, 0.000, 1.000)
            viewer._server.initial_camera.fov = 1.3090
            viewer._server.initial_camera.near = 0.01
            viewer._server.initial_camera.far = 1000

        # Render initial state before the first step
        self.viewer = viewer
        self.render()

        #
        # region Gym
        # Gym parameters and setup
        #

        # TODO: take into account multiple worlds
        # num_actuators_per_world...

        # Control the four wheel motors
        num_actuators = len(self.actuators[0].indices)
        assert num_actuators == 4, f"Expected 4 actuators, but got {num_actuators}"
        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(4,))
        self.prev_action = np.zeros(4, dtype=np.float32)

        # chassis linear velocity: (vx, vy)
        # chassis angular velocity: (yaw_rate)
        # heading sincos: (sin(yaw), cos(yaw))
        # waypoint relative position: (dx, dy, dd) * self.n_lookahead
        # progress along path: (index / num_waypoints)
        # cross track error: (cte)
        # previous action: (4,)
        obs_dim = 2 + 1 + 2 + (3 * 3) + 1 + 1 + 4
        self.observation_space = gym.spaces.Box(low=-inf, high=inf, shape=(obs_dim,))

        # region Debug

        if not self.quiet:
            print("Number of worlds:", self.model.world_count)
            print(f"Model finalized (device={self.model.device})")
            print("  Num bodies:", self.model.body_count)
            print("  Num shapes:", self.model.shape_count)
            print("  Num joints:", self.model.joint_count)
            print("State, Contacts and Control objects created")
            print("  State body count:", self.state_0.body_count)
            print("  State joint dof count:", self.state_0.joint_dof_count)
            assert self.control.joint_act
            print("  Control size:", self.control.joint_act.size)
            print("Solver created:", type(self.solver).__name__)
            print("Simulation configured")
            print(f"  Frame dt: {self.frame_dt:.4f} s")
            print(f"  Physics dt: {self.sim_dt:.4f} s")

            if self.graph:
                print("CUDA graph captured for optimized execution")
            else:
                print("Running on CPU (no CUDA graph)")

    # region Observation

    def _cross_track_error(self, pos: NDArray) -> float:
        # TODO: figure out initial waypoint (at origin?)
        # if self.waypoint_index == 0 or self.waypoint_index >= len(self.waypoints):
        #     return 0.0
        if self.waypoint_index >= len(self.waypoints):
            return 0.0
        a = self.waypoints[self.waypoint_index - 1] if self.waypoint_index > 0 else np.array([0.0, 0.0])
        b = self.waypoints[self.waypoint_index]
        ab = b - a
        L = float(np.linalg.norm(ab)) + 1e-8
        return float(np.cross(ab, pos - a)) / L

    def _get_obs2(self) -> ObsType:
        assert self.state_0.body_q

        q = self.state_0.body_q.numpy()[self.chassis]

        if not np.all(np.isfinite(q)):
            print("BAD q:", q)
            raise RuntimeError("Non-finite body_q in _get_obs")

        pos = q[:2].astype(np.float32)

        quat = q[3:]

        if not np.all(np.isfinite(quat)):
            print("BAD quat:", quat)
            raise RuntimeError("Non-finite quaternion in _get_obs")

        _, _, yaw = quat_to_rpy(quat)

        if not np.isfinite(yaw):
            print("BAD yaw from quat:", quat, "yaw:", yaw)
            raise RuntimeError("Non-finite yaw in _get_obs")

        action = self.action.astype(np.float32) if hasattr(self, "action") else np.zeros(4, dtype=np.float32)

        obs = np.concatenate(
            [
                pos,
                np.array([np.sin(yaw), np.cos(yaw)], dtype=np.float32),
                action,
            ]
        ).astype(np.float32)

        if not np.all(np.isfinite(obs)):
            print("BAD obs:", obs)
            print("q:", q)
            print("pos:", pos)
            print("yaw:", yaw)
            print("action:", action)
            raise RuntimeError("Non-finite observation")

        return obs

    def _get_obs(self) -> ObsType:

        # TODO: add config for frequency of sensor updates and cache
        if self.robot_config.add_imu:
            self.imu.update(self.state_0)
            # acc = self.imu.accelerometer.numpy()  # (n_sensors, 3) linear acceleration
            # gyro = self.imu.gyroscope.numpy()  # (n_sensors, 3) angular velocity

        # TODO: remove asserts? check performance implications
        assert self.state_0.body_q
        q = self.state_0.body_q.numpy()[self.chassis]
        pos = q[:2]
        _, _, yaw = quat_to_rpy(q[3:])
        heading_sincos = np.sin(yaw), np.cos(yaw)

        assert self.state_0.body_qd
        qd = self.state_0.body_qd.numpy()[self.chassis]
        qd_rel = world_to_body(yaw, qd[:2])
        omega = qd[5]

        progress = self.waypoint_index / len(self.waypoints)
        cross_track_error = self._cross_track_error(pos[:2])

        # Relative waypoints (body frame), padded if past end
        rel = np.zeros((self.n_lookahead, 3), dtype=np.float32)
        for k in range(self.n_lookahead):
            idx = self.waypoint_index + k
            if idx < len(self.waypoints):
                d_world = self.waypoints[idx] - pos[:2]
                d_body = world_to_body(yaw, d_world)
                rel[k, 0:2] = d_body
                rel[k, 2] = np.linalg.norm(d_world)
            else:
                # Repeat the last waypoint's relative info (zeros if finished)
                rel[k] = rel[k - 1]

        obs = np.concatenate(
            [
                qd_rel[:2],
                [omega],
                heading_sincos,
                rel.flatten(),
                [progress],
                [cross_track_error],
                self.prev_action,
            ]
        ).astype(np.float32)

        assert obs.shape == self.observation_space.shape, (
            f"Observation shape {obs.shape} does not match expected shape {self.observation_space.shape}"
        )
        return obs

    def _get_info(self) -> InfoType:
        q = self.state_0.body_q.numpy()[self.chassis]  # type: ignore
        qd = self.state_0.body_qd.numpy()[self.chassis]  # type: ignore

        widx = self.waypoint_index
        target = self.waypoints[widx] if widx < len(self.waypoints) else np.array([0.0, 0.0])
        dist = float(np.linalg.norm(target - q[:2])) if widx < len(self.waypoints) else 0.0

        info = {
            "pos": q[:2],
            "yaw": quat_to_rpy(q[3:])[2],
            "vel": qd[:2],
            "dist": dist,
        }

        if dist < self.hit_radius:
            self.waypoint_index += 1
            info["event"] = "finished" if self.waypoint_index >= len(self.waypoints) else "waypoint_hit"

        return info

    # region Reset

    def reset(self, *, seed: int | None = None, options: OptType | None = None) -> tuple[ObsType, InfoType]:
        super().reset(seed=seed)
        # TODO: don't reset sim_time if specified (useful for visualization and debugging)
        # self.sim_time = 0.0

        # issac will do a complete tear down and reinitialization similarly to this
        if self.steps != 0:
            self.prev_action = np.zeros(4, dtype=np.float32)
            self.waypoint_index = 0
            self.prev_dist = None
            self.prev_segment_progress = None
            self.steps = 0
            self._start_simulation()

        """
        Must reset
        - state, control, actuator, contacts, ?solver?
        self._np_random, _ = gym.utils.seeding.np_random(seed)
        """

        obs = self._get_obs()
        info = self._get_info()

        return obs, info

    # region step

    def _simulate(self) -> None:

        for _ in range(self.sim_steps_per_frame):
            # Not accounting for control delay
            if self.control_steps_counter >= self.sim_steps_per_control:
                self.control_steps_counter = 0
                assert self.control.joint_f
                self.control.joint_f.zero_()

                for actuator in self.actuators:
                    actuator.step(self.state_0, self.control, dt=self.frame_dt)

            self.control_steps_counter += 1

            self.state_0.clear_forces()
            self.model.collide(self.state_0, self.contacts)
            self.solver.step(
                state_in=self.state_0,
                state_out=self.state_1,
                control=self.control,
                contacts=self.contacts,
                dt=self.sim_dt,
            )
            self.state_0, self.state_1 = self.state_1, self.state_0

    def _compute_reward(self) -> tuple[float, InfoType]:
        info = {}
        reward = float(np.mean(self.action))
        return reward, info

    def step(self, action: NDArray) -> tuple[ObsType, float, bool, bool, InfoType]:

        action = np.asarray(action, dtype=np.float32)

        if not np.all(np.isfinite(action)):
            raise RuntimeError("Non-finite action received by env")

        action = np.clip(action, self.action_space.low, self.action_space.high)  # type: ignore
        self.action = action.copy()
        motors = action * 12.0

        self.actuation_values[self.actuator_indices] = np.tile(motors, self.model.world_count)  # type: ignore
        wp.copy(self.control.joint_target_vel, wp.array(self.actuation_values))  # type: ignore

        if self.graph:
            wp.capture_launch(self.graph)
        else:
            self._simulate()

        self.sim_time += self.frame_dt
        self.steps += 1

        obs = self._get_obs()
        info = self._get_info()

        reward, reward_info = self._compute_reward()
        info.update(reward_info)

        at_end = self.waypoint_index >= len(self.waypoints)
        out_of_bounds = np.any(np.abs(info["pos"]) > self.world_bound)
        terminated = at_end or out_of_bounds

        truncated = self.steps >= self.max_steps and not terminated

        self.prev_action = action

        return obs, float(reward), bool(terminated), bool(truncated), info

    # region render
    def render(self, mode="viser") -> RenderFrame | list[RenderFrame] | None:
        if self.viewer:
            self.viewer.begin_frame(self.sim_time)
            self.viewer.log_state(self.state_0)
            self.viewer.end_frame()

    # region close
    def close(self) -> None:
        # TODO: handle "(viser) Server stopped" message at start somehow
        if self.viewer:
            self.viewer.close()
        super().close()
