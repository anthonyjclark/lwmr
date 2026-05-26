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

# TODO: add more specificity
ObsType = NDArray
InfoType = dict[str, Any]
OptType = dict[str, Any]
WaypointType = NDArray


class LwmrPlaneEnv(gym.Env):
    metadata = {"render_modes": ["viser", "none"], "render_fps": 60}

    def __init__(
        self,
        robot_config: LwmrRobotConfig = LwmrRobotConfig(),
        waypoints: list[WaypointType] = [np.array([0.5, 0.0]), np.array([0.5, 0.5])],
        solver_name: str = "MuJoCo",
        sim_freq: int = 600,
        control_freq: int = 5,
        frame_freq: int | None = None,
        num_worlds: int = 1,
        device: str = "cuda",
        quiet: bool = False,
        render_mode: str = "none",
        max_viewer_worlds: int = 16,
        viewer_port: int = 8080,
        viewer_spacing: float = 0.8,
        viewer_output_path: str = "./recordings/lwmr_plane.viser",
    ):
        super().__init__()

        # TODO: consider validating arguments

        self.quiet = quiet

        # TODO: clone? per world?
        self.waypoints = waypoints
        self.waypoint_index = 0

        # Set global quiet mode for Warp before newton is initialized in the environment
        if quiet:
            wp.config.quiet = True

        # `frame_freq` is related to both `step()` and `render()`
        frame_freq = frame_freq if frame_freq is not None else self.metadata["render_fps"]

        assert frame_freq
        assert sim_freq % frame_freq == 0, "`sim_freq` must be a multiple of `frame_freq`"
        self.sim_steps_per_frame = sim_freq // frame_freq

        assert sim_freq % control_freq == 0, "`sim_freq` must be a multiple of `control_freq`"
        self.sim_steps_per_control = sim_freq // control_freq

        self.sim_time = 0.0
        self.frame_dt = 1.0 / frame_freq
        self.sim_dt = 1.0 / sim_freq
        self.control_steps_counter = 0

        drop_height = robot_config.wh_radius + 0.05

        #
        # region World
        # (default world -1; collides with all worlds)
        #

        builder = newton.ModelBuilder()

        # Use body=-1 to attach shapes to the static world frame:

        # # Set defaults before adding shapes
        # builder.default_shape_cfg.ke = 1.0e6
        # builder.default_shape_cfg.kd = 1000.0
        # builder.default_shape_cfg.mu = 0.5
        # builder.default_shape_cfg.is_hydroelastic = True
        # builder.default_shape_cfg.sdf_max_resolution = 64  # Primitive SDF defaults

        # TODO: explore ground characteristics (friction, restitution, etc in ShapeConfig)
        # TODO: make environment configuration (e.g., initial conditions, randomization parameters) configurable via kwargs
        # NOTE: The default plane (`.add_ground_plane`) is unsuitable for visualization
        # plane_size = 0.5
        # plane_eqn: tuple[float, float, float, float] = *world.up_vector, 0
        # world.add_shape_plane(body=-1, plane=plane_eqn, width=plane_size, length=plane_size)
        builder.add_ground_plane()

        #
        # region Env
        # Create shared world and ground plane
        #

        # TODO: add config
        # validate_inertia: bool = False,
        # builder.validate_inertia_detailed = validate_inertia

        initial_xform = wp.transform(p=(0.0, 0.0, drop_height))
        fixed_base = False
        # fixed_base = True  # TODO: make this configurable and test both cases

        robot_builder = newton.ModelBuilder()

        self.chassis, _, _, _ = add_lwmr_robot(
            robot_builder,
            initial_xform,
            robot_config,
            fixed_base=fixed_base,
        )

        assert not robot_config.add_imu, "IMU is currently not supported."
        if robot_config.add_imu:
            # Add an imu at the chassis center
            robot_builder.add_site(body=self.chassis, label="imu")

        assert num_worlds == 1, "Multiple worlds are currently not supported."
        for _ in range(num_worlds):
            builder.begin_world()

            # Add a step to the world for the robot to drive over
            # hz = (robot_config.wh_radius / 2.5) * np.random.uniform(0.8, 1.2)
            hz = (robot_config.wh_radius / 5.5) * np.random.uniform(0.8, 1.2)
            pos = (0.5, 0.0, hz)
            rot = wp.quat_rpy(0.0, 0.0, np.random.uniform(-0.2, 0.2))

            # builder.add_shape_box(
            #     body=-1,
            #     hx=0.1,
            #     hy=0.5,
            #     hz=hz,
            #     xform=wp.transform(p=pos, q=rot),
            #     color=(0.5, 0.5, 0.5),
            # )

            builder.add_builder(robot_builder)

            builder.end_world()

        # # Add a site with offset and rotation
        # camera_site = robot_builder.add_site(
        #     body=chassis_body,
        #     xform=wp.transform(
        #         wp.vec3(0.5, 0, 0.2),  # Position
        #         wp.quat_from_axis_angle(wp.vec3(0, 1, 0), 3.14159 / 4),  # Orientation
        #     ),
        #     type=newton.GeoType.BOX,
        #     scale=(0.05, 0.05, 0.02),
        #     visible=True,
        #     label="camera",
        # )

        # # TODO: replicate vs vec
        # # Should this be world.replicate?
        # num_worlds = 1
        # builder.replicate(robot_builder, world_count=num_worlds)  # , spacing=(2.0, 0.0, 0.0))
        # # world.add_world(robot_builder)
        # # world.color()

        self.model = builder.finalize(device=device)

        if robot_config.add_imu:
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
        # region Viewer
        #

        viewer = None

        assert render_mode in self.metadata["render_modes"], f"Unsupported render mode: {render_mode}"
        if render_mode == "viser":
            # TODO: warn if `viewer_output_path` already exists and will be overwritten
            recording_path = Path(viewer_output_path).resolve()
            recording_path.parent.mkdir(parents=True, exist_ok=True)

            viewer = create_viewer_viser(str(recording_path), quiet=quiet, port=viewer_port)

            max_viewer_worlds = min(self.model.world_count, max_viewer_worlds)
            viewer.set_model(self.model, max_worlds=max_viewer_worlds)
            viewer.set_world_offsets(spacing=(viewer_spacing, viewer_spacing, 0.0))

            axes = [
                ("x-axes", (1.0, 0.0, 0.001)),
                ("y-axes", (0.0, 1.0, 0.001)),
                ("z-axes", (0.0, 0.0, 1.0)),
            ]

            # Add axes to the viewer for reference
            for label, axis in axes:
                starts = wp.array([wp.vec3(0, 0, 0.001)])
                ends = wp.array([wp.vec3(*axis)])
                viewer.log_arrows(label, starts, ends, axis, width=0.04)

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
        # region Solver
        #

        # TODO: support different solvers and configurations (e.g., iterations, tolerance, etc)
        assert solver_name == "MuJoCo", f"Unsupported solver: {solver_name}"
        using_generalized_coordinates = solver_name in ["MuJoCo", "Featherstone"]

        # TODO: try other solvers
        # TODO: try other parameter values
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

        # TODO: replace with actual observation space from the robot
        # chassis linear velocity: (vx, vy)
        # chassis angular velocity: (yaw_rate)
        # heading sincos: (sin(yaw), cos(yaw))
        # waypoint relative position: (dx, dy, dyaw) * NUM_WAYPOINT_LOOKAHEAD
        # progress along path: (??)
        # cross track error: (??)
        # previous action: (4,)
        obs_dim = 2 + 1 + 2 + (3 * 3) + 1 + 1 + 4
        self.observation_space = gym.spaces.Box(low=-inf, high=inf, shape=(obs_dim,))

        # region Debug

        # if not quiet:
        #     print("Number of worlds:", world.world_count)
        #     print(f"Model finalized (device={self.model.device})")
        #     print("  Num bodies:", self.model.body_count)
        #     print("  Num shapes:", self.model.shape_count)
        #     print("  Num joints:", self.model.joint_count)
        #     print("State, Contacts and Control objects created")
        #     print("  State body count:", self.state_0.body_count)
        #     print("  State joint dof count:", self.state_0.joint_dof_count)
        #     print(f"  Control size: {self.control.joint_act.size}")
        #     print("Solver created:", type(self.solver).__name__)
        #     print("Simulation configured")
        #     print(f"  Frame rate: {fps} Hz")
        #     print(f"  Frame dt: {self.frame_dt:.4f} s")
        #     print(f"  Physics substeps: {self.sim_substeps}")
        #     print(f"  Physics dt: {self.sim_dt:.4f} s")

        #     if self.graph:
        #         print("CUDA graph captured for optimized execution")
        #     else:
        #         print("Running on CPU (no CUDA graph)")

        # # # NOTE: generalized coordinate (MuJoCo) solvers use joint_q and joint_qd
        # # #       maximal coordinate solvers (XPBD) use body_q and body_qd
        # # print(f"self.model.joint_dof_count: {self.model.joint_dof_count}")
        # # print(f"self.control.joint_f: {self.control.joint_f}")
        # # print(f"self.control.joint_target_pos: {self.control.joint_target_pos}")
        # # print(f"self.control.joint_target_vel: {self.control.joint_target_vel}")
        # # print(f"self.control.joint_act: {self.control.joint_act}")
        # # print(f"self.state_0.joint_q: {self.state_0.joint_q}")
        # # print(f"self.state_0.joint_qd: {self.state_0.joint_qd}")
        # # print(f"self.model.joint_q: {self.model.joint_q}")
        # # print(f"self.model.joint_qd: {self.model.joint_qd}")
        # # print(f"self.state_0.body_q: {self.state_0.body_q}")
        # # print(f"self.state_0.body_qd: {self.state_0.body_qd}")
        # # print(f"self.model.body_q: {self.model.body_q}")
        # # print(f"self.model.body_qd: {self.model.body_qd}")
        # # print(f"self.model.actuators: {self.model.actuators}")
        # # print("Body worlds:", self.model.body_world.numpy().tolist())
        # # print("Shape worlds:", self.model.shape_world.numpy().tolist())
        # # print("Joint worlds:", self.model.joint_world.numpy().tolist())
        # # print("Joint worlds:", self.model.particle_world.numpy().tolist())
        # # print("Joint worlds:", self.model.articulation_world.numpy().tolist())
        # # print("Joint worlds:", self.model.equality_constraint_world.numpy().tolist())

        # # # rigid bodies
        # # if self.body_count:
        # #     s.body_q = wp.clone(self.body_q, requires_grad=requires_grad)
        # #     s.body_qd = wp.clone(self.body_qd, requires_grad=requires_grad)
        # #     s.body_f = wp.zeros_like(self.body_qd, requires_grad=requires_grad)

        # # # joints
        # # if self.joint_count:
        # #     s.joint_q = wp.clone(self.joint_q, requires_grad=requires_grad)
        # #     s.joint_qd = wp.clone(self.joint_qd, requires_grad=requires_grad)

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

    def _get_obs(self) -> ObsType:
        """
        pos_error = cube_pos - drop_off_pos
        pos_error_xy = np.linalg.norm(pos_error[:2])
        pos_error_z = np.abs(pos_error[2])
        """

        # TODO: figure out frequency of sensor updates and cache

        # imu.update(state)
        # acc = imu.accelerometer.numpy()   # (n_sensors, 3) linear acceleration
        # gyro = imu.gyroscope.numpy()      # (n_sensors, 3) angular velocity

        # TODO: remove asserts? check performance implications
        assert self.state_0.body_q
        q = self.state_0.body_q.numpy()[self.chassis]
        pos = q[:3]
        _, _, yaw = quat_to_rpy(q[3:])
        heading_sincos = np.sin(yaw), np.cos(yaw)

        assert self.state_0.body_qd
        qd = self.state_0.body_qd.numpy()[self.chassis]
        # v_body_frame = v_com - omega x r_com_world
        # qd_rel = qd[:3] - np.cross(qd[3:], pos)
        qd_rel = world_to_body(yaw, qd[:2])
        omega = qd[5]

        progress = self.waypoint_index / len(self.waypoints)
        cross_track_error = self._cross_track_error(pos[:2])

        # Relative waypoints (body frame), padded if past end
        rel = np.zeros((self.N_LOOKAHEAD, 3), dtype=np.float32)
        for k in range(self.N_LOOKAHEAD):
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
        assert self.state_0.body_q
        q = self.state_0.body_q.numpy()[self.chassis]
        assert self.state_0.body_qd
        qd = self.state_0.body_qd.numpy()[self.chassis]
        return {
            "pos": q[:3],
            "yaw": quat_to_rpy(q[3:])[2],
            "vel": qd[:3],
        }

    # region Reset

    def reset(self, *, seed: int | None = None, options: OptType | None = None) -> tuple[ObsType, InfoType]:
        super().reset(seed=seed)
        self.sim_time = 0.0
        self.prev_dist = None
        self.waypoint_index = 0
        # TODO: outstanding issue for this
        # self.solver.reset()
        # TODO: handle vectorised envs and randomized initial conditions
        # TODO: handle environmental randomization

        # print("[INFO] Resetting example")
        # wp.copy(self.state_0.joint_q, self._initial_joint_q)
        # wp.copy(self.state_0.joint_qd, self._initial_joint_qd)
        # wp.copy(self.state_1.joint_q, self._initial_joint_q)
        # wp.copy(self.state_1.joint_qd, self._initial_joint_qd)
        # newton.eval_fk(self.model, self.state_0.joint_q, self.state_0.joint_qd, self.state_0)
        # newton.eval_fk(self.model, self.state_1.joint_q, self.state_1.joint_qd, self.state_1)
        # # Clear stale policy history so the first observation after reset is
        # # built purely from the restored kinematic state.
        # if self._prev_act_wp is not None:
        #     self._prev_act_wp.zero_()

        # """Reset the simulation."""
        # if self.reset_graph:
        #     wp.capture_launch(self.reset_graph)
        # else:
        #     self.sim.reset()
        # if not self.use_cuda_graph and self.logging:
        #     self.logger.reset()
        #     self.logger.log()

        # issac will do a complete tear down and reinitialization
        # cls.start_simulation()
        # cls.initialize_solver()

        # self._np_random, _ = gym.utils.seeding.np_random(seed)

        """
        Must reset
        - state, control, actuator, contacts, ?solver?
        """

        obs = self._get_obs()
        info = self._get_info()
        return obs, info

    # region step

    def _simulate(self) -> None:

        for _ in range(self.sim_steps_per_frame):
            # NOTE: this structure does not account for control delay
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

    N_LOOKAHEAD = 3
    HIT_RADIUS = 0.05  # distance threshold for waypoint "hit"
    WORLD_BOUND = 20.0  # crash if |pos| exceeds this
    # MAX_LIN_ACC = 4.0  # m/s^2 commanded by throttle = 1
    # MAX_YAW_RATE = 2.5  # rad/s commanded by steering = 1
    # LIN_DRAG = 0.5  # simple drag so velocity doesn't explode

    W_PROGRESS = 1.0
    W_HIT = 10.0
    W_CTE = 0.1
    W_HEADING = 0.05
    W_ACTION = 0.01
    W_TIME = 0.002
    W_CRASH = 50.0
    W_FINISH = 100.0

    def _compute_reward(self, action, crashed, pos, yaw, info):

        if crashed:
            info["event"] = "crash"
            return -self.W_CRASH, True
        if self.waypoint_index >= len(self.waypoints):
            info["event"] = "finish"
            return self.W_FINISH, True

        target = self.waypoints[self.waypoint_index]
        dist = float(np.linalg.norm(target - pos))

        # 1. Potential-based progress
        if self.prev_dist is None:
            progress_r = 0.0
        else:
            progress_r = self.W_PROGRESS * (self.prev_dist - dist)
        self.prev_dist = dist

        # 2. Waypoint hit
        hit_r = 0.0
        if dist < self.HIT_RADIUS:
            hit_r = self.W_HIT
            self.waypoint_index += 1
            self.prev_dist = None
            info["event"] = "waypoint_hit"

        # 3. Cross-track penalty
        cte_r = -self.W_CTE * abs(self._cross_track_error(pos))

        # 4. Heading alignment toward current target
        if self.waypoint_index < len(self.waypoints):
            tgt = self.waypoints[self.waypoint_index]
            desired = np.arctan2(tgt[1] - pos[1], tgt[0] - pos[0])
            err = np.arctan2(np.sin(desired - yaw), np.cos(desired - yaw))
            heading_r = self.W_HEADING * np.cos(err)
        else:
            heading_r = 0.0

        # 5. Action effort
        action_r = -self.W_ACTION * float(np.sum(np.square(action)))

        # 6. Time
        time_r = -self.W_TIME

        reward = progress_r + hit_r + cte_r + heading_r + action_r + time_r

        terminated = self.waypoint_index >= len(self.waypoints)
        if terminated:
            reward += self.W_FINISH
            info["event"] = "finish"

        info["dist"] = dist
        info["waypoint_index"] = self.waypoint_index

        return reward, terminated

    def step(self, action: NDArray) -> tuple[ObsType, float, bool, bool, InfoType]:

        # Clip action?
        # action = np.clip(action, self.action_space.low, self.action_space.high)

        self.actuation_values[self.actuator_indices] = np.tile(action, self.model.world_count)  # type: ignore
        wp.copy(self.control.joint_target_vel, wp.array(self.actuation_values))  # type: ignore

        if self.graph:
            wp.capture_launch(self.graph)
        else:
            self._simulate()

        self.sim_time += self.frame_dt

        obs = self._get_obs()
        info = self._get_info()

        # TODO: figure out crashing conditions (e.g., flipped over, out of bounds, etc)
        # crashed = np.any(np.abs(self.pos) > self.WORLD_BOUND)
        crashed = False

        q = self.state_0.body_q.numpy()[self.chassis]
        pos = q[:2]
        _, _, yaw = quat_to_rpy(q[3:])
        reward, terminated = self._compute_reward(action, crashed, pos, yaw, info)

        # TODO: figure out termination conditions
        truncated = False
        # truncated = self.steps >= self.MAX_STEPS and not terminated

        self.prev_action = action

        return obs, float(reward), bool(terminated), bool(truncated), info

        reward = 0.0
        terminated = False
        truncated = False
        return obs, reward, terminated, truncated, info

    # region render
    def render(self, mode="viser") -> RenderFrame | list[RenderFrame] | None:
        if self.viewer:
            self.viewer.begin_frame(self.sim_time)
            self.viewer.log_state(self.state_0)
            self.viewer.end_frame()

    # region close
    def close(self) -> None:
        # self.viewer.save_recording()
        # TODO: handle "(viser) Server stopped" message at start somehow
        if self.viewer:
            self.viewer.close()
        super().close()
