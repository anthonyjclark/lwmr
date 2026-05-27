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
        # TODO: this is only for compatibility with rllib
        config=None,
        *,
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
        self.robot_config = robot_config
        self.waypoints = waypoints
        self.solver_name = solver_name
        self.sim_freq = sim_freq
        self.control_freq = control_freq
        self.frame_freq = frame_freq
        self.num_worlds = num_worlds
        self.device = device
        self.quiet = quiet
        self.render_mode = render_mode
        self.max_viewer_worlds = max_viewer_worlds
        self.viewer_port = viewer_port
        self.viewer_spacing = viewer_spacing
        self.viewer_output_path = viewer_output_path
        self._start_simulation()

    def _start_simulation(self):

        # TODO:
        self.max_episode_steps = 256
        self.steps = 0

        # TODO: consider validating arguments

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
            self.robot_config,
            fixed_base=fixed_base,
        )

        assert not self.robot_config.add_imu, "IMU is currently not supported."
        if self.robot_config.add_imu:
            # Add an imu at the chassis center
            robot_builder.add_site(body=self.chassis, label="imu")

        assert self.num_worlds == 1, "Multiple worlds are currently not supported."
        for _ in range(self.num_worlds):
            builder.begin_world()

            # Add a step to the world for the robot to drive over
            # hz = (robot_config.wh_radius / 2.5) * np.random.uniform(0.8, 1.2)
            hz = (self.robot_config.wh_radius / 5.5) * np.random.uniform(0.8, 1.2)
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

        # # TODO: support different solvers and configurations (e.g., iterations, tolerance, etc)
        # assert solver_name == "MuJoCo", f"Unsupported solver: {solver_name}"
        # using_generalized_coordinates = solver_name in ["MuJoCo", "Featherstone"]

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
        # region Viewer
        #

        viewer = None

        assert self.render_mode in self.metadata["render_modes"], (
            f"Unsupported render mode: {self.render_mode}"
        )
        if self.render_mode == "viser":
            # TODO: warn if `viewer_output_path` already exists and will be overwritten
            recording_path = Path(self.viewer_output_path).resolve()
            recording_path.parent.mkdir(parents=True, exist_ok=True)

            viewer = create_viewer_viser(str(recording_path), quiet=self.quiet, port=self.viewer_port)

            max_viewer_worlds = min(self.model.world_count, self.max_viewer_worlds)
            viewer.set_model(self.model, max_worlds=max_viewer_worlds)
            viewer.set_world_offsets(spacing=(self.viewer_spacing, self.viewer_spacing, 0.0))

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
        # obs_dim = 8
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

        # TODO: figure out frequency of sensor updates and cache

        # imu.update(state)
        # acc = imu.accelerometer.numpy()   # (n_sensors, 3) linear acceleration
        # gyro = imu.gyroscope.numpy()      # (n_sensors, 3) angular velocity

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
        # TODO: don't reset sim_time if specified (useful for visualization and debugging)
        # self.sim_time = 0.0

        if self.steps != 0:
            self.prev_action = np.zeros(4, dtype=np.float32)
            self.waypoint_index = 0
            self.prev_dist = None
            self.prev_segment_progress = None
            self.steps = 0
            self._start_simulation()
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
    HIT_RADIUS = 0.01  # distance threshold for waypoint "hit"
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

    def _compute_reward(self, action, crashed, pos, yaw):
        info = {}

        MAX_DIST_TO_TARGET = 1.0

        target = self.waypoints[self.waypoint_index].astype(np.float32)
        dist_to_target = float(np.linalg.norm(target - pos))
        # dist_to_target = min(dist_to_target, MAX_DIST_TO_TARGET)
        # reward = (MAX_DIST_TO_TARGET - dist_to_target) / MAX_DIST_TO_TARGET
        # reward = float(np.mean(action))

        velocity = self.state_0.body_qd.numpy()[self.chassis][:2]
        target_velocity = np.array([0.25, 0.0], dtype=np.float32)
        velocity_error = velocity - target_velocity
        velocity_error_reward = 1 - np.tanh(5 * np.linalg.norm(velocity_error))
        reward = velocity_error_reward

        terminated = dist_to_target < self.HIT_RADIUS

        return reward, info, terminated

        # ----------------------------
        # Tunable reward constants
        # ----------------------------
        R_CRASH = -500.0

        R_PROGRESS_FWD = 20.0  # reward per meter of forward path progress
        R_PROGRESS_BACK = 40.0  # penalty per meter of backward path progress

        R_WAYPOINT = 150.0  # bonus per waypoint reached
        R_FINISH = 1000.0  # terminal completion bonus

        R_TIME = -0.01  # per-step penalty
        R_ACTION = -0.001  # small effort penalty

        R_CTE = -0.10  # cross-track penalty coefficient
        CTE_CLIP = 2.0  # cap cross-track penalty magnitude

        # Optional anti-stall term.
        R_STALL = -0.02
        MIN_SPEED = 0.05

        # ----------------------------
        # Terminal failure
        # ----------------------------
        if crashed:
            return R_CRASH, {"event": "crash"}, True

        # ----------------------------
        # Already finished
        # ----------------------------
        if self.waypoint_index >= len(self.waypoints):
            return R_FINISH, {"event": "finish"}, True

        # pos = self.pos.astype(np.float32)
        action = np.asarray(action, dtype=np.float32)

        target = self.waypoints[self.waypoint_index].astype(np.float32)
        dist_to_target = float(np.linalg.norm(target - pos))

        # ----------------------------
        # Segment start/end
        # ----------------------------
        # For the first segment, use the true episode start if available.
        # Otherwise default to origin.
        if self.waypoint_index == 0:
            if hasattr(self, "start_pos"):
                a = self.start_pos.astype(np.float32)
            else:
                a = np.zeros(2, dtype=np.float32)
        else:
            a = self.waypoints[self.waypoint_index - 1].astype(np.float32)

        b = target
        ab = b - a
        seg_len = float(np.linalg.norm(ab)) + 1e-8
        ab_hat = ab / seg_len

        # Progress along current segment, in meters.
        raw_segment_progress = float(np.dot(pos - a, ab_hat))
        segment_progress = float(np.clip(raw_segment_progress, 0.0, seg_len))

        # ----------------------------
        # 1. Forward path progress
        # ----------------------------
        if not hasattr(self, "prev_segment_progress") or self.prev_segment_progress is None:
            delta_progress = 0.0
        else:
            delta_progress = segment_progress - self.prev_segment_progress

        self.prev_segment_progress = segment_progress

        if delta_progress >= 0.0:
            progress_r = R_PROGRESS_FWD * delta_progress
        else:
            progress_r = R_PROGRESS_BACK * delta_progress

        # ----------------------------
        # 2. Waypoint hit / finish
        # ----------------------------
        hit_r = 0.0
        finish_r = 0.0
        terminated = False

        if dist_to_target < self.HIT_RADIUS:
            hit_r = R_WAYPOINT
            self.waypoint_index += 1

            # Reset progress baseline for the next segment.
            self.prev_segment_progress = None

            info["event"] = "waypoint_hit"

            if self.waypoint_index >= len(self.waypoints):
                finish_r = R_FINISH
                terminated = True
                info["event"] = "finish"

        # ----------------------------
        # 3. Time penalty
        # ----------------------------
        time_r = R_TIME

        # ----------------------------
        # 4. Action penalty
        # ----------------------------
        action_r = R_ACTION * float(np.sum(np.square(action)))

        # ----------------------------
        # 5. Cross-track penalty, clipped
        # ----------------------------
        # Important: clipped so that a successful but imperfect trajectory
        # cannot get destroyed by one large CTE term.
        cte = float(self._cross_track_error(pos))
        cte_abs_clipped = min(abs(cte), CTE_CLIP)
        cte_r = R_CTE * cte_abs_clipped

        # ----------------------------
        # 6. Small anti-stall penalty
        # ----------------------------
        if hasattr(self, "vel_world"):
            speed = float(np.linalg.norm(self.vel_world))
        else:
            speed = 0.0

        if not terminated and dist_to_target > self.HIT_RADIUS and speed < MIN_SPEED:
            stall_r = R_STALL
        else:
            stall_r = 0.0

        # ----------------------------
        # Total
        # ----------------------------
        reward = progress_r + hit_r + finish_r + time_r + action_r + cte_r + stall_r

        info.update(
            {
                "dist": dist_to_target,
                "waypoint_index": self.waypoint_index,
                "raw_segment_progress": raw_segment_progress,
                "segment_progress": segment_progress,
                "delta_progress": delta_progress,
                "progress_r": progress_r,
                "hit_r": hit_r,
                "finish_r": finish_r,
                "time_r": time_r,
                "action_r": action_r,
                "cte": cte,
                "cte_r": cte_r,
                "stall_r": stall_r,
                "speed": speed,
                "reward": float(reward),
            }
        )

        return float(reward), info, terminated

    def _compute_reward2(self, action, crashed, pos, yaw, info):

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

        action = np.asarray(action, dtype=np.float32)

        if not np.all(np.isfinite(action)):
            print("BAD incoming action:", action)
            raise RuntimeError("Non-finite action received by env")

        action = np.clip(action, self.action_space.low, self.action_space.high)

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

        # TODO: figure out crashing conditions (e.g., flipped over, out of bounds, etc)
        # crashed = np.any(np.abs(self.pos) > self.WORLD_BOUND)
        crashed = False

        q = self.state_0.body_q.numpy()[self.chassis]
        pos = q[:2]
        _, _, yaw = quat_to_rpy(q[3:])
        reward, reward_info, terminated = self._compute_reward(action, crashed, pos, yaw)
        info.update(reward_info)

        # TODO: figure out termination conditions
        truncated = self.steps >= self.max_episode_steps and not terminated

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
