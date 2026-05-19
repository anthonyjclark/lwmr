# Legged-Wheeled Mobile Robot

## Initial Setup

```bash
uv init --python 3.13 --bare
uv init --package packages/lwmr

# Install dependencies
uv add "newton[examples,notebook,torch-cu12]"

# Test newton
python -m newton.examples robot_anymal_c_walk --viewer null

# Create packages and add them to the project
uv init --package packages/lwmr
uv add --editable packages/lwmr
```

## Newton Concepts

- Multi-world simulation (i.e., vectorised/parallel environments)

http://127.0.0.1:8047/viser-client/?playbackPath=http://127.0.0.1:8047/recordings/test.viser

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
