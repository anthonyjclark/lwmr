from pathlib import Path
from typing import Any

import gymnasium as gym
import newton
import warp as wp
from gymnasium.core import RenderFrame
from newton.sensors import SensorIMU

from ..robot import add_lwmr_robot

# TODO: add more specificity
ObsType = gym.spaces.Box
InfoType = dict[str, Any]
OptType = dict[str, Any]


class LwmrPlaneEnv(gym.Env):
    # TODO: implement ViewerFile
    # TODO: consider ViewerUSD
    metadata = {"render_modes": ["viser", "file", "none"], "render_fps": 60}

    # TODO: list all the possible kwargs in the docstring, and their defaults (e.g., density, ch_width, ch_length, ch_height, wh_radius, wh_thickness, drop_height, solver, device, etc)
    # def __init__(self, **kwargs):
    def __init__(
        self,
        # TODO: extract to RobotConfig dataclass and pass as single argument
        ch_width: float = 0.3,
        ch_length: float = 0.15,
        ch_height: float = 0.02,
        wh_radius: float = 0.03,
        density: float = 1000.0,
        add_imu: bool = False,
        solver_name: str = "MuJoCo",
        sim_freq: int = 600,
        control_freq: int = 5,
        frame_freq: int | None = None,
        device: str = "cuda",
        quiet: bool = False,
        # TODO: actually use this to configure rendering
        render_mode: str = "none",
    ):
        super().__init__()

        # TODO: consider validating arguments

        self.quiet = quiet

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

        # TODO: leg params

        # TODO: only needed if supporting cylindrical wheels
        # wh_thickness = kwargs.get("wh_thickness", 0.01)

        drop_height = wh_radius + 0.05

        #
        # region World
        # (default world -1; collides with all worlds)
        #

        world = newton.ModelBuilder()

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
        world.add_ground_plane()

        #
        # region Env
        # Create shared world and ground plane
        #

        initial_xform = wp.transform(p=(0.0, 0.0, drop_height))

        # return builder, chassis_body, wheel_bodies, wheel_joints, wheel_qd_indices
        robot_builder, chassis, _, _, self.wheel_qd_indices = add_lwmr_robot(
            xform=initial_xform,
            ch_width=ch_width,
            ch_length=ch_length,
            ch_height=ch_height,
            ch_density=density,
            wh_radius=wh_radius,
            wh_density=density,
            lg_radius=wh_radius * 0.3,
            lg_offset=wh_radius,
            num_legs=3,
            fixed_base=False,
            # fixed_base=True,
        )

        # Add an imu at the chassis center
        if add_imu:
            robot_builder.add_site(body=chassis, label="imu")

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

        # TODO: replicate vs vec
        # Should this be world.replicate?
        # scene.replicate(arm, world_count=4, spacing=(2.0, 0.0, 0.0))

        world.add_world(robot_builder)
        self.model = world.finalize(device=device)

        if add_imu:
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
        self.actuation_values = self.control.joint_target_vel.numpy()

        #
        # region Viewer
        #

        # TODO: check render mode before creating viewer, and only create if needed
        recording_path = Path("./recordings/test4.viser").resolve()
        recording_path.parent.mkdir(parents=True, exist_ok=True)

        # TODO: configure the port
        if not quiet:
            self.viewer = newton.viewer.ViewerViser(record_to_viser=str(recording_path))
        else:
            import rich

            console = rich.get_console()
            with console.capture() as _:
                self.viewer = newton.viewer.ViewerViser(record_to_viser=str(recording_path), verbose=False)

        # NOTE: allows for multiple worlds (`max_worlds=`)
        self.viewer.set_model(self.model)

        starts = wp.array([wp.vec3(0, 0, 0.001)])
        ends = wp.array([wp.vec3(1, 0, 0.001)])
        self.viewer.log_arrows("x-axes", starts, ends, (1.0, 0.0, 0.0), width=0.04)

        starts = wp.array([wp.vec3(0, 0, 0.001)])
        ends = wp.array([wp.vec3(0, 1, 0.001)])
        self.viewer.log_arrows("y-axes", starts, ends, (0.0, 1.0, 0.0), width=0.04)

        starts = wp.array([wp.vec3(0, 0, 0.001)])
        ends = wp.array([wp.vec3(0, 0, 1)])
        self.viewer.log_arrows("z-axes", starts, ends, (0.0, 0.0, 1.0), width=0.04)

        # Set the initial camera pose (this is a bit of a workaround)
        # self.viewer.set_camera(pos=wp.vec3(-0.748, -0.626, 0.576), pitch=-0.5, yaw=0.0)
        self.viewer._server.initial_camera.position = (-0.748, -0.626, 0.576)
        self.viewer._server.initial_camera.look_at = (0.000, 0.000, 0.000)
        self.viewer._server.initial_camera.up = (0.000, 0.000, 1.000)
        self.viewer._server.initial_camera.fov = 1.3090
        self.viewer._server.initial_camera.near = 0.01
        self.viewer._server.initial_camera.far = 1000

        # Render initial state before the first step
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

        # NOTE: control the four wheel motors
        # TODO: make this depend on self.control.joint_target_vel shape
        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(len(self.actuators),))

        # TODO: replace with actual observation space from the robot
        self.observation_space = gym.spaces.Box(low=-float("inf"), high=float("inf"), shape=(7,))

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

    # region Obs
    def _get_obs(self) -> ObsType:

        # TODO: figure out frequency of sensor updates and cache

        # imu.update(state)
        # acc = imu.accelerometer.numpy()   # (n_sensors, 3) linear acceleration
        # gyro = imu.gyroscope.numpy()      # (n_sensors, 3) angular velocity

        # TODO: remove asserts? check performance implications
        assert self.state_0.body_q
        # TODO: currently returning the 7D transform of the sphere
        return self.state_0.body_q.numpy()[0]

    # region Info
    def _get_info(self) -> InfoType:
        return {}

    # region simulate
    def _simulate(self) -> None:

        for _ in range(self.sim_steps_per_frame):
            # Update control
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

    # region reset
    def reset(self, *, seed: int | None = None, options: OptType | None = None) -> tuple[ObsType, InfoType]:
        super().reset(seed=seed)
        self.sim_time = 0.0
        # TODO: handle vectorised envs and randomized initial conditions
        # TODO: handle environmental randomization
        # self.lwmr.reset()
        obs = self._get_obs()
        info = self._get_info()
        return obs, info

    # region step
    def step(self, action) -> tuple[ObsType, float, bool, bool, InfoType]:

        self.actuation_values[self.wheel_qd_indices] = action[:]
        wp.copy(self.control.joint_target_vel, wp.array(self.actuation_values))  # type: ignore

        if self.graph:
            wp.capture_launch(self.graph)
        else:
            self._simulate()

        self.sim_time += self.frame_dt

        obs = self._get_obs()
        info = self._get_info()
        # reward = self.lwmr.get_reward()
        # terminated = self.lwmr.is_terminated()
        # truncated = self.lwmr.is_truncated()
        reward = 0.0
        terminated = False
        truncated = False
        return obs, reward, terminated, truncated, info

    # region render
    def render(self, mode="viser") -> RenderFrame | list[RenderFrame] | None:
        if mode == "viser":
            self.viewer.begin_frame(self.sim_time)
            self.viewer.log_state(self.state_0)
            self.viewer.end_frame()

    # region close
    def close(self) -> None:
        # TODO: check if viewer exists and if recording should be saved
        # self.viewer.save_recording()
        # TODO: handle "(viser) Server stopped" message at start somehow
        self.viewer.close()
        if not self.quiet:
            print("Viewer closed and recording saved")
        super().close()
