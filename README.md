# Legged-Wheeled Mobile Robot

## Initial Setup

```bash
# From inside your project directory
uv init --python 3.13 --bare
uv init --package packages/lwmr

# Install dependencies (if local, uv add "./path/to/newton[examples,notebook,torch-cu12]")
uv add "newton[examples,notebook,torch-cu12]"

# Minimual example
uv run -m newton.examples basic_pendulum --viewer null
# Extended example
uv run --extra examples -m newton.examples robot_humanoid --num-envs 16 --viewer null
# Torch example
uv run --extra examples --extra torch-cu12 -m newton.examples robot_anymal_c_walk --viewer null
# List all examples
uv run -m newton.examples

# Create packages and add them to the project
uv init --package packages/lwmr
uv add --editable packages/lwmr
```

## Newton Concepts

- Multi-world simulation (i.e., vectorised/parallel environments)

http://127.0.0.1:8047/viser-client/?playbackPath=http://127.0.0.1:8047/recordings/test.viser

http://localhost:8000/viser-client/?playbackPath=http://localhost:8000/recordings/recording.viser&logCamera

http://127.0.0.1:8047/viser-client/?playbackPath=http://127.0.0.1:8047/recordings/test3.viser&initialCameraPosition=-0.748,-0.626,0.576&initialCameraLookAt=0.000,0.000,0.000&initialCameraUp=0.000,0.000,1.000&initialCameraFov=1.3090&initialCameraNear=0.01&initialCameraFar=1000

- no articulation (The articulation is a set of joints that must be contiguous and monotonically increasing)

- "For heterogeneous worlds, use begin_world() and end_world()."
- "For large-scale parallel simulations (e.g., RL), replicate() stamps out many copies of a template environment builder into separate worlds in one call:" (`main.replicate(env_builder, world_count=1024)`)

- User add_link instead of add_shape

[Articulations](https://newton-physics.github.io/newton/stable/concepts/articulations.html)
- Generalized (sometimes also called “reduced”) coordinates describe an articulation in terms of its joint positions and velocities.
- Maximal coordinates describe the configuration of an articulation in terms of the body link positions and velocities.
- To convert between these two representations, we use forward and inverse kinematics: forward kinematics (newton.eval_fk()) converts generalized coordinates to maximal coordinates, and inverse kinematics (newton.eval_ik()) converts maximal coordinates to generalized coordinates.
- Newton supports both parameterizations, and each solver chooses which one it treats as the primary articulation state representation. For example, SolverMuJoCo and SolverFeatherstone use generalized coordinates, while SolverXPBD, SolverSemiImplicit, and SolverVBD use maximal coordinates. Note that collision detection, e.g., via newton.Model.collide() requires the maximal coordinates to be current in the state.

Notes
- `add_body()` calls `add_link()`, `add_joint_free()`, and `add_articulation()`
- `model.joint_q` stores the default positions and is used to initialize `state.joint_q`
- `model.joint_qd` stores the default velocities and is used to initialize `state.joint_qd`
- `model.joint_qd`, `state.joint_qd`, and `control.joint_f` share DOF order
- `joint_q` and `joint_qd` can use different indieces (and lengths)
- query `model.joint_dof_dim` for the number of DOFs for each joint
- use an `ArticulationView` to handle duplicate articulations for RL

Generalized
- write to `joint_q` and `joint_qd`
- call `eval_fk()` to update `joint_q` and `joint_qd` from `body_q` and `body_qd`

Maximal
- write to `body_q` and `body_qd` (pose is initialized with `xform` in `add_link()`)
- call `eval_ik()` to update `body_q` and `body_qd` from `joint_q` and `joint_qd`

To use either generalized or maximal coordinates

```python
builder = newton.ModelBuilder()

body = builder.add_link(xform=xform)
shape = builder.add_shape_box(body, ...) # Inertial

joint = builder.add_joint_free(body)

builder.add_articulation([joint])

model = builder.finalize()

...

state = model.state()

# The body poses (maximal coordinates) are initialized by the xform argument:
assert all(state.body_q.numpy()[0] == [*tf])

# Now, the generalized coordinates are initialized by the free joint:
assert len(state.joint_q) == 7
assert all(state.joint_q.numpy() == [*tf])
```


add_constraint_mimic(joint0, joint1, coef0=0.0, coef1=1.0, enabled=True, label=None, custom_attributes=None)
Adds a mimic constraint to the model.
    A mimic constraint enforces that joint0 = coef0 + coef1 * joint1




    # NOTE: setting initialize position of the joint
    # q_start = builder.joint_q_start[wheel_joint]
    # q_count = sum(builder.joint_dof_dim[wheel_joint])
    # builder.joint_q[q_start : q_start + q_count] = [0.5]

    # NOTE: setting initial velocity of the joint
    # qd_start = builder.joint_qd_start[wheel_joint]
    # qd_count = sum(builder.joint_dof_dim[wheel_joint])
    # builder.joint_qd[qd_start : qd_start + qd_count] = [5]

    # NOTE: setting initial effort of the joint
    # f = ma --> a = f/m (constant acceleration)
    # f_start = builder.joint_qd_start[wheel_joint]
    # f_count = sum(builder.joint_dof_dim[wheel_joint])
    # builder.joint_f[f_start : f_start + f_count] = [0.01]

Timings for
- simulation step (forward dynamics, collision detection, constraint solving)
- rendering step (CPU and GPU times)
- control step

- custom controller/actuator for tutorial?

`wp.config.quiet = True`


Requirements/recommendations for a mobile base
- Building
  - Create an articulation for the robot
  - Create sub-builders for obstacles
  - Create sub-builders for the terrain
- If setting initial conditions by `joint_q` and `joint_qd` -->
  - call `newton.eval_fk` to update `body_q` and `body_qd` for maximal coordinate solvers
  - calling `newton.eval_fk` is not necessary for generalized coordinate solvers
- If setting initial conditions by `xform=...` -->
  - maximal coordinate solvers are all set, but `joint_q` and `joint_qd` will be set be `None`
  - call `newton.eval_ik` to update `joint_q` and `joint_qd` for generalized coordinate solvers
- If using generalized coordinates
  - create a free joint attaching the robot base to the world
- Convert from generalized to maximal coordinates with `newton.eval_fk`
- Convert from maximal to generalized coordinates with `newton.eval_ik`
- ["MuJoCo Warp as the primary solver for rigid body dynamics"](https://github.com/newton-physics/newton/discussions/639)

```python
using_generalized_coordinates = solver_name in ["MuJoCo", "Featherstone"]

# Main builder for the shared ground plane
world = newton.ModelBuilder(...)
world.current_env_group = -1
world.add_ground_plane()

# Create an independent builder for the scene so that it can be duplicated
scene = newton.ModelBuilder(...)
scene.add_articulation(...)
root_body = scene.add_body(...)
scene.add_shape_*(root_body, ...)
# NOTE: it is okay to add a free joint even when using maximal coordinates
if using_generalized_coordinates: scene.add_joint_free(root_body, ...)

# Add multiple instances of the scene to the world
for i in range(num_envs):
  world.add_builder(scene, xform=..., ...)

model = world.finalize()

# isinstance(solver, SolverMuJoCo | SolverFeatherstone)

if not using_generalized_coordinates: newton.eval_ik(model, state_0, state_0.joint_q, state_0.joint_qd)
```


## Control

- Control fields
  - `self.model.joint_dof_count` is the number of degrees of freedom (DOF) for all joints in the model
  - `self.control.joint_f` is the force/torque applied to each joint
  - `self.control.joint_target` is the target position for each joint (position or velocity depending on mode)
- State fields (this might not work)
  - `self.state_0.joint_q` is the position of each joint
  - `self.state_0.joint_qd` is the velocity of each joint


## Bugs/fixes/features

- set width and length of plane in add_ground_plane
- builder.add_shape_box does not use is_static=True when mass is 0
- export compute_box_inertia in newton.geometry
- `world.plot_articulation()` should allow plotting to a file

## Timing

Axes
- steps
- frequency (Hz)
- time (s)

Timesteps for
- rendering (`frame_dt = 1.0 / render_fps`)

sim_freq
control_freq

elapsed_steps

elapsed_time


Here are the notable parts (I am controlling based on velocity, not position, you'll probably want position control instead):
1. Create a joint: builder.add_joint_*(...)
2. Create an actuator: builder.add_actuator(...)
3. Set the target: wp.copy(self.control.joint_target_vel, wp.array(self.actuation_values))
4. Step the actuator: actuator.step(self.state_0, self.control, dt=self.frame_dt)


# Always have the source handy
❯ git clone --depth 1 --branch develop https://github.com/isaac-sim/IsaacLab



`NewtonManager`
- `_solver_dt = 1.0 / 200.0`
- `_num_substeps = 1`

what information will the robot have? (proprioception? cameras? lidar? pose?)


Don't need `make_vec` when using newton
    # TODO: consider vectorised environments
    # vec_env = gym.make_vec(..., n_envs=...)

setup operations can be slow (non-gpu)
try to keep all sim step operations on the gpu

should `reset` take a batched input for vectorised environments?


Opinionated python
- `tyro`
- `tqdm` or
- `@dataclass` for config
- wandb
- tensorboard
- checkpoints

def _set_seed(self, seed: int | None) -> None:
    rng = torch.Generator()
    if seed is not None:
        rng.manual_seed(seed)
    self.rng = rng
