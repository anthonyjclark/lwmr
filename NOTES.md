# Notes

## Rendering Links

http://localhost:8000/viser-client/?playbackPath=http://localhost:8000/recordings/recording.viser&logCamera

http://127.0.0.1:8047/viser-client/?playbackPath=http://127.0.0.1:8047/recordings/test3.viser&initialCameraPosition=-0.748,-0.626,0.576&initialCameraLookAt=0.000,0.000,0.000&initialCameraUp=0.000,0.000,1.000&initialCameraFov=1.3090&initialCameraNear=0.01&initialCameraFar=1000

## Newton Concepts

From [Articulations](https://newton-physics.github.io/newton/stable/concepts/articulations.html):
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

## Mimic Joints

add_constraint_mimic(joint0, joint1, coef0=0.0, coef1=1.0, enabled=True, label=None, custom_attributes=None)
Adds a mimic constraint to the model.
    A mimic constraint enforces that joint0 = coef0 + coef1 * joint1

## Setting initial conditions for joints

```python
# NOTE: setting initialize position of the joint
q_start = builder.joint_q_start[wheel_joint]
q_count = sum(builder.joint_dof_dim[wheel_joint])
builder.joint_q[q_start : q_start + q_count] = [0.5]

# NOTE: setting initial velocity of the joint
qd_start = builder.joint_qd_start[wheel_joint]
qd_count = sum(builder.joint_dof_dim[wheel_joint])
builder.joint_qd[qd_start : qd_start + qd_count] = [5]

# NOTE: setting initial effort of the joint
# f = ma --> a = f/m (constant acceleration)
f_start = builder.joint_qd_start[wheel_joint]
f_count = sum(builder.joint_dof_dim[wheel_joint])
builder.joint_f[f_start : f_start + f_count] = [0.01]
```

## Timing

Timings for
- simulation step (forward dynamics, collision detection, constraint solving)
- rendering (must be multiple of simulation steps)
- control (must be multiple of simulation steps)
- sensor updates (must be multiple of simulation steps)

Axes
- steps
- frequency (Hz)
- time (s)

## Actuators

- custom controller/actuator for tutorial?

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

Actuators
1. Create a joint: builder.add_joint_*(...)
2. Create an actuator: builder.add_actuator(...)
3. Set the target: wp.copy(self.control.joint_target_vel, wp.array(self.actuation_values))
4. Step the actuator: actuator.step(self.state_0, self.control, dt=self.frame_dt)

## Control

- Control fields
  - `self.model.joint_dof_count` is the number of degrees of freedom (DOF) for all joints in the model
  - `self.control.joint_f` is the force/torque applied to each joint
  - `self.control.joint_target` is the target position for each joint (position or velocity depending on mode)
- State fields
  - `self.state_0.joint_q` is the position of each joint
  - `self.state_0.joint_qd` is the velocity of each joint

## Bugs/fixes/features

- builder.add_shape_box does not use is_static=True when mass is 0
- export compute_box_inertia in newton.geometry
- `world.plot_articulation()` should allow plotting to a file



what information will the robot have? (proprioception? cameras? lidar? pose?)


torchrl ParallelEnv vs newton num_worlds/world_count

Don't need `make_vec` when using newton
    # TODO: consider vectorised environments
    # vec_env = gym.make_vec(..., n_envs=...)

setup operations can be slow (non-gpu)
try to keep all sim step operations on the gpu

should `reset` take a batched input for vectorised environments?


- checkpoints

def _set_seed(self, seed: int | None) -> None:
    rng = torch.Generator()
    if seed is not None:
        rng.manual_seed(seed)
    self.rng = rng
    # torch.manual_seed(seed)

maybe mask the different worlds?
if done, then either reset or don't process input?


use info to plot path

create notebook to test sb3 model

input/output scaling



discuss
        self,
        # TODO: this is only for compatibility with rllib
        config=None,
        *,
        robot_config: LwmrRobotConfig = LwmrRobotConfig(),
        waypoints: list[WaypointType] | None = None,
        solver_name: str = "MuJoCo",
        max_steps: int = 256,
        sim_freq: int = 600,
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



@wp.kernel
def drone_cost(
    body_q: wp.array[wp.transform],
    body_qd: wp.array[wp.spatial_vector],
    targets: wp.array[wp.vec3],
    prop_control: wp.array[float],
    step: int,
    horizon_length: int,
    weighting: float,
    cost: wp.array[wp.float32],
):
    world_id = wp.tid()
    tf = body_q[world_id]
    target = targets[0]

    pos_drone = wp.transform_get_translation(tf)
    pos_cost = wp.length_sq(pos_drone - target)
    altitude_cost = wp.max(pos_drone[2] - 0.75, 0.0) + wp.max(0.25 - pos_drone[2], 0.0)
    upvector = wp.vec3(0.0, 0.0, 1.0)
    drone_up = wp.transform_vector(tf, upvector)
    upright_cost = 1.0 - wp.dot(drone_up, upvector)

    vel_drone = body_qd[world_id]

    # Encourage zero velocity.
    vel_cost = wp.length_sq(vel_drone)

    control = wp.vec4(
        prop_control[world_id * 4 + 0],
        prop_control[world_id * 4 + 1],
        prop_control[world_id * 4 + 2],
        prop_control[world_id * 4 + 3],
    )
    control_cost = wp.dot(control, control)

    discount = 0.8 ** wp.float(horizon_length - step - 1) / wp.float(horizon_length) ** 2.0

    pos_weight = 1000.0
    altitude_weight = 100.0
    control_weight = 0.05
    vel_weight = 0.1
    upright_weight = 10.0
    total_weight = pos_weight + altitude_weight + control_weight + vel_weight + upright_weight

    wp.atomic_add(
        cost,
        world_id,
        (
            pos_cost * pos_weight
            + altitude_cost * altitude_weight
            + control_cost * control_weight
            + vel_cost * vel_weight
            + upright_cost * upright_weight
        )
        * (weighting / total_weight)
        * discount,
    )



## Multiple Worlds

- Multi-world simulation (i.e., vectorised/parallel environments)

- "For heterogeneous worlds, use begin_world() and end_world()."
- "For large-scale parallel simulations (e.g., RL), replicate() stamps out many copies of a template environment builder into separate worlds in one call:" (`main.replicate(env_builder, world_count=1024)`)

## Initial Setup Using Uv

```bash
# From inside your project directory
uv init --python 3.12 --bare
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
