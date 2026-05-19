from pathlib import Path
from typing import Any

import gymnasium as gym
import newton
import warp as wp
from gymnasium.core import RenderFrame

from ..robot import add_lwmr_robot

# TODO: add more specificity
ObsType = gym.spaces.Box
InfoType = dict[str, Any]
OptType = dict[str, Any]


# # TODO: remove
# max_delay = 5
# N = 2  # 2 % 5 != 0, N < buf_depth, N is even
# K = 2
# dt = 0.02
# warmup_target = 0.0
# cycle_targets = [2.0, -3.0, 5.0, -1.0]


class LwmrPlaneEnv(gym.Env):
    # TODO: implement ViewerFile
    # TODO: consider ViewerUSD
    # TODO: consider ViewerRerun instead of ViewerViser (both support notebooks)
    # TODO: define a render_fps
    metadata = {"render_modes": ["viser", "file", "none"], "render_fps": 60}

    # TODO: list all the possible kwargs in the docstring, and their defaults (e.g., density, ch_width, ch_length, ch_height, wh_radius, wh_thickness, drop_height, solver, device, etc)
    def __init__(self, **kwargs):
        super().__init__()

        quiet = kwargs.get("quiet", False)
        self.quiet = quiet

        solver_name = kwargs.get("solver", "MuJoCo")
        using_generalized_coordinates = solver_name in ["MuJoCo", "Featherstone"]

        #
        # region Gym
        # Gym parameters and setup
        #

        # NOTE: control the four wheel motors
        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(4,))

        # TODO: replace with actual observation space from the robot
        self.observation_space = gym.spaces.Box(low=-float("inf"), high=float("inf"), shape=(7,))

        #
        # region World
        # (default world -1; collides with all worlds)
        #

        world = newton.ModelBuilder()

        # Use body=-1 to attach shapes to the static world frame:

        # builder = newton.ModelBuilder()

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

        # # World-frame reference site
        # world_origin = world.add_site(
        #     body=-1,
        #     xform=wp.transform(wp.vec3(0, 0, 0), wp.quat_identity()),
        #     label="world_origin"
        # )
        # # 1. Create sensor and specify what to measure
        # imu = SensorIMU(model, sites="imu_*")
        # ...
        # # 2. Compute measurements from the current state
        # imu.update(state)

        # # 3. Results stored on sensor attributes
        # acc = imu.accelerometer.numpy()   # (n_sensors, 3) linear acceleration
        # gyro = imu.gyroscope.numpy()      # (n_sensors, 3) angular velocity

        #
        # region Dims
        # Physical dimensions and properties for the robot
        #

        density = kwargs.get("density", 1000.0)

        ch_width = kwargs.get("ch_width", 0.3)
        ch_length = kwargs.get("ch_length", 0.15)
        ch_height = kwargs.get("ch_height", 0.02)

        wh_radius = kwargs.get("wh_radius", 0.03)

        # TODO: only needed if supporting cylindrical wheels
        # wh_thickness = kwargs.get("wh_thickness", 0.01)

        drop_height = wh_radius + 0.02 + 0.1

        #
        # region Env
        # Create shared world and ground plane
        #

        initial_xform = wp.transform(p=(0.0, 0.0, drop_height))

        self.robot_builder, _, _, _ = add_lwmr_robot(
            xform=initial_xform,
            ch_width=ch_width,
            ch_length=ch_length,
            ch_height=ch_height,
            ch_density=density,
            wh_radius=wh_radius,
            wh_density=density,
            add_support=True,
            using_generalized_coordinates=using_generalized_coordinates,
        )

        # TODO: replicate vs vec
        # Should this be world.replicate?
        # scene.replicate(arm, world_count=4, spacing=(2.0, 0.0, 0.0))

        world.add_world(self.robot_builder)
        self.model = world.finalize(device=kwargs.get("device", None))

        # #
        # device = self.model.device
        self.ndof = self.model.joint_coord_count
        # ndof = self.model.joint_coord_count
        # solver, s0, s1, ctrl, act, act_a, act_b = _setup(self.model)
        # self.solver = solver
        # self.state_0 = s0
        # self.state_1 = s1
        # self.control = ctrl
        # self.actuator = act
        # self.actuator_state_0 = act_a
        # self.actuator_state_1 = act_b
        # wp.copy(ctrl.joint_target_pos, wp.full(ndof, warmup_target, dtype=wp.float32, device=device))
        # s0, s1, act_a, act_b = _loop(solver, s0, s1, ctrl, act, act_a, act_b, max_delay)
        # #

        #
        # region State
        #

        # TODO: rename state_0/state_1 to state_curr/state_next
        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()
        self.contacts = self.model.contacts()

        self.actuators = self.model.actuators
        # self.actuator_state_0 = self.actuator.state()
        # self.actuator_state_1 = self.actuator.state()

        # # rigid bodies
        # if self.body_count:
        #     s.body_q = wp.clone(self.body_q, requires_grad=requires_grad)
        #     s.body_qd = wp.clone(self.body_qd, requires_grad=requires_grad)
        #     s.body_f = wp.zeros_like(self.body_qd, requires_grad=requires_grad)

        # # joints
        # if self.joint_count:
        #     s.joint_q = wp.clone(self.joint_q, requires_grad=requires_grad)
        #     s.joint_qd = wp.clone(self.joint_qd, requires_grad=requires_grad)

        # Create the XPBD solver with 10 constraint iterations
        # TODO: make the type of solver configurable along with the solver options
        # self.solver = newton.solvers.SolverXPBD(self.model, iterations=10)
        # TODO: try other solvers
        # TODO: try other parameter values
        self.solver = newton.solvers.SolverMuJoCo(self.model, iterations=100, ls_iterations=50, njmax=100)

        # # TODO: just do this always?
        # if not using_generalized_coordinates:
        #     assert self.state_0.joint_q and self.state_0.joint_qd
        #     newton.eval_ik(self.model, self.state_0, self.state_0.joint_q, self.state_0.joint_qd)

        # TODO: only do this when appropriate?
        assert self.state_0.joint_q and self.state_0.joint_qd
        newton.eval_fk(self.model, self.state_0.joint_q, self.state_0.joint_qd, self.state_0)

        # Simulation parameters
        # TODO: control, render, and physics time steps
        fps = self.metadata["render_fps"]
        self.frame_dt = 1.0 / fps  # Time step per frame
        self.sim_substeps = 10  # Number of physics substeps per frame
        self.sim_dt = self.frame_dt / self.sim_substeps  # Physics time step
        self.sim_time = 0.0

        # TODO: check render mode before creating viewer, and only create if needed
        recording_path = Path("./recordings/test3.viser").resolve()
        # Create the Viser viewer with a path to save the recording
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

        # # TODO: log x, y, and z arrows
        # starts = wp.array([wp.vec3(0, 0, 0.001)])
        # ends = wp.array([wp.vec3(1, 0, 0.001)])
        # self.viewer.log_arrows("axes", starts, ends, None)

        # # Render initial state before the first step
        # # TODO: this doesn't appear to work...
        # self.render()

        # self.viewer.set_camera(pos=(0, -50, 50), pitch=-0.5, yaw=0.0)

        # self._simulate()

        # Capture the simulation as a CUDA graph (if running on GPU)
        if wp.get_device().is_cuda:
            with wp.ScopedCapture() as capture:
                self._simulate()
            self.graph = capture.graph
        else:
            self.graph = None

        # region Debug

        if not quiet:
            print("Number of worlds:", world.world_count)
            print(f"Model finalized (device={self.model.device})")
            print("  Num bodies:", self.model.body_count)
            print("  Num shapes:", self.model.shape_count)
            print("  Num joints:", self.model.joint_count)
            print("State, Contacts and Control objects created")
            print("  State body count:", self.state_0.body_count)
            print("  State joint dof count:", self.state_0.joint_dof_count)
            # print(f"  Control size: {self.control.joint_act.size}")
            print("Solver created:", type(self.solver).__name__)
            print("Simulation configured")
            print(f"  Frame rate: {fps} Hz")
            print(f"  Frame dt: {self.frame_dt:.4f} s")
            print(f"  Physics substeps: {self.sim_substeps}")
            print(f"  Physics dt: {self.sim_dt:.4f} s")

            if self.graph:
                print("CUDA graph captured for optimized execution")
            else:
                print("Running on CPU (no CUDA graph)")

        # # NOTE: generalized coordinate (MuJoCo) solvers use joint_q and joint_qd
        # #       maximal coordinate solvers (XPBD) use body_q and body_qd
        # print(f"self.model.joint_dof_count: {self.model.joint_dof_count}")
        # print(f"self.control.joint_f: {self.control.joint_f}")
        # print(f"self.control.joint_target_pos: {self.control.joint_target_pos}")
        # print(f"self.control.joint_target_vel: {self.control.joint_target_vel}")
        # print(f"self.control.joint_act: {self.control.joint_act}")
        # print(f"self.state_0.joint_q: {self.state_0.joint_q}")
        # print(f"self.state_0.joint_qd: {self.state_0.joint_qd}")
        # print(f"self.model.joint_q: {self.model.joint_q}")
        # print(f"self.model.joint_qd: {self.model.joint_qd}")
        # print(f"self.state_0.body_q: {self.state_0.body_q}")
        # print(f"self.state_0.body_qd: {self.state_0.body_qd}")
        # print(f"self.model.body_q: {self.model.body_q}")
        # print(f"self.model.body_qd: {self.model.body_qd}")
        # print(f"self.model.actuators: {self.model.actuators}")
        # print("Body worlds:", self.model.body_world.numpy().tolist())
        # print("Shape worlds:", self.model.shape_world.numpy().tolist())
        # print("Joint worlds:", self.model.joint_world.numpy().tolist())
        # print("Joint worlds:", self.model.particle_world.numpy().tolist())
        # print("Joint worlds:", self.model.articulation_world.numpy().tolist())
        # print("Joint worlds:", self.model.equality_constraint_world.numpy().tolist())

    # region Obs
    def _get_obs(self) -> ObsType:
        # return self.lwmr.get_obs()
        # TODO: remove asserts? check performance implications
        assert self.state_0.body_q
        # TODO: currently returning the 7D transform of the sphere
        return self.state_0.body_q.numpy()[0]

    # region Info
    def _get_info(self) -> InfoType:
        return {}

    def _simulate(self) -> None:

        assert self.control.joint_f
        self.control.joint_f.zero_()

        # self.control.joint_target_vel[]

        # print("=== control.joint_f          :", self.control.joint_f)
        # print("=== control.joint_target_vel :", self.control.joint_target_vel)
        # print("=== model.joint_qd           :", self.model.joint_qd)
        # print("=== state_0.joint_qd         :", self.state_0.joint_qd)

        # print("positions =", getattr(self.state_0, self.actuator.state_pos_attr))
        # print("velocities =", getattr(self.state_0, self.actuator.state_vel_attr))

        # print("orig_target_pos =", getattr(self.control, self.actuator.control_target_pos_attr))
        # print("orig_target_vel =", getattr(self.control, self.actuator.control_target_vel_attr))

        # orig_feedforward = None
        # if self.actuator.control_feedforward_attr is not None:
        #     orig_feedforward = getattr(self.control, self.actuator.control_feedforward_attr, None)

        # # print("target_pos =", orig_target_pos)
        # # print("target_vel =", orig_target_vel)
        # print("feedforward =", orig_feedforward)
        # print("target_pos_indices =", self.actuator.target_pos_indices)
        # print("target_vel_indices =", self.actuator.indices)

        # print("=== ndof    :", self.ndof)
        # print("=== full    :", wp.full(self.ndof, 1.0, dtype=wp.float32, device=self.model.device))

        # tgt = 0.0001
        # assert self.control.joint_target_vel
        # # wp.copy(
        # #     self.control.joint_target_vel, wp.full(self.ndof, tgt, dtype=wp.float32, device=self.model.device)
        # self.control.joint_target_vel.fill_(tgt)
        # # )

        # TODO: figure out relationship between sim_dt and control_dt
        # self.actuator.step(
        #     self.state_0, self.control, self.actuator_state_0, self.actuator_state_1, dt=self.frame_dt
        # )
        for actuator in self.actuators:
            actuator.step(self.state_0, self.control, dt=self.frame_dt)

        # self.actuator_state_0, self.actuator_state_1 = self.actuator_state_1, self.actuator_state_0

        for _ in range(self.sim_substeps):
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

        # print("=== control.joint_f          :", self.control.joint_f)
        # print("=== control.joint_target_vel :", self.control.joint_target_vel)
        # print("=== model.joint_qd           :", self.model.joint_qd)
        # print("=== state_0.joint_qd         :", self.state_0.joint_qd)

    def _simulate2(self) -> None:
        for _ in range(self.sim_substeps):
            # self.state_0.clear_forces()
            # TODO: update control
            # TODO: viewer.apply_forces(state_0)
            # TODO: apply external forces
            self.model.collide(self.state_0, self.contacts)
            self.solver.step(
                state_in=self.state_0,
                state_out=self.state_1,
                control=self.control,
                contacts=self.contacts,
                dt=self.sim_dt,
            )
            self.state_0, self.state_1 = self.state_1, self.state_0

    def reset(self, *, seed: int | None = None, options: OptType | None = None) -> tuple[ObsType, InfoType]:
        super().reset(seed=seed)
        self.sim_time = 0.0
        # TODO: handle vectorised envs and randomized initial conditions
        # TODO: handle environmental randomization
        # self.lwmr.reset()
        obs = self._get_obs()
        info = self._get_info()
        return obs, info

    def step(self, action) -> tuple[ObsType, float, bool, bool, InfoType]:
        # self.lwmr.step(action)

        # Execute the simulation (use CUDA graph if available)
        if self.graph:
            wp.capture_launch(self.graph)
        else:
            self._simulate()

        # Advance simulation time
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

    def render(self, mode="viser") -> RenderFrame | list[RenderFrame] | None:
        if mode == "viser":
            # Log the current state to the viewer
            self.viewer.begin_frame(self.sim_time)
            self.viewer.log_state(self.state_0)
            self.viewer.end_frame()

            # # Log contacts to the viewer (not supported by the Viser viewer)
            # self.viewer.log_contacts(self.contacts, self.state_0)

    def close(self) -> None:
        # TODO: check if viewer exists and if recording should be saved
        # self.viewer.save_recording()
        # TODO: handle "(viser) Server stopped" message at start somehow
        self.viewer.close()
        if not self.quiet:
            print("Viewer closed and recording saved")
        super().close()
