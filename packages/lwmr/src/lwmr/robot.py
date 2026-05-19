import newton
import warp as wp
from newton._src.core.types import Transform
from newton.actuators import ClampingDCMotor, ControllerPD
from newton.geometry import compute_inertia_shape


def add_lwmr_robot(
    xform: Transform,
    ch_width: float,
    ch_length: float,
    ch_height: float,
    ch_density: float,
    wh_radius: float,
    # TODO: only specify wh_thickness for cylindrical wheels
    # wh_thickness: float,
    wh_density: float,
    add_support: bool,
    using_generalized_coordinates: bool,
    # TODO: default to false
    fixed_base: bool = True,
) -> tuple[newton.ModelBuilder, int, list[int], list[int]]:

    # max_delay = 5
    # builder = newton.ModelBuilder()
    # builder.default_shape_cfg.density = 1000.0
    # link = builder.add_link()
    # joint = builder.add_joint_revolute(parent=-1, child=link, axis=newton.Axis.Z)
    # builder.add_shape_sphere(body=link, radius=0.1)
    # builder.add_articulation([joint])
    # dof = builder.joint_qd_start[joint]
    # builder.add_actuator(
    #     ControllerPD,
    #     index=dof,
    #     kp=200.0,
    #     kd=10.0,
    #     delay_steps=max_delay,
    #     # clamping=[(ClampingMaxEffort, {"max_effort": 500.0})],
    # )
    # return builder, link, None, [joint]

    builder = newton.ModelBuilder()
    # TODO: look into `validate_inertia_detailed` for debugging inertia issues
    # builder.validate_inertia_detailed = True

    '''
    density: float = 1000.0
    """The density of the shape material."""
    ke: float = 2.5e3
    """The contact elastic stiffness. Used by SemiImplicit, Featherstone, MuJoCo."""
    kd: float = 100.0
    """The contact damping coefficient. Used by SemiImplicit, Featherstone, MuJoCo."""
    kf: float = 1000.0
    """The friction damping coefficient. Used by SemiImplicit, Featherstone."""
    ka: float = 0.0
    """The contact adhesion distance. Used by SemiImplicit, Featherstone."""
    mu: float = 1.0
    """The coefficient of friction. Used by all solvers."""
    restitution: float = 0.0
    """The coefficient of restitution. Used by XPBD. To take effect, enable restitution in solver constructor via ``enable_restitution=True``."""
    mu_torsional: float = 0.005
    """The coefficient of torsional friction (resistance to spinning at contact point). Used by XPBD, MuJoCo."""
    mu_rolling: float = 0.0001
    """The coefficient of rolling friction (resistance to rolling motion). Used by XPBD, MuJoCo."""
    margin: float = 0.0
    """Outward offset from the shape's surface [m] for collision detection.
    Extends the effective collision surface outward by this amount. When two shapes collide,
    their margins are summed (margin_a + margin_b) to determine the total separation [m].
    This value is also used when computing inertia for hollow shapes (``is_solid=False``)."""
    gap: float | None = None
    """Additional contact detection gap [m]. If None, uses builder.rigid_gap as default.
    Broad phase uses (margin + gap) [m] for AABB expansion and pair filtering."""
    is_solid: bool = True
    """Indicates whether the shape is solid or hollow. Defaults to True."""
    collision_group: int = 1
    """The collision group ID for the shape. Defaults to 1 (default group). Set to 0 to disable collisions for this shape."""
    collision_filter_parent: bool = True
    """Whether to inherit collision filtering from the parent. Defaults to True."""
    has_shape_collision: bool = True
    """Whether the shape can collide with other shapes. Defaults to True."""
    has_particle_collision: bool = True
    """Whether the shape can collide with particles. Defaults to True."""
    is_visible: bool = True
    """Indicates whether the shape is visible in the simulation. Defaults to True."""
    is_site: bool = False
    """Indicates whether the shape is a site (non-colliding reference point). Directly setting this to True will NOT enforce site invariants. Use `mark_as_site()` or set via the `flags` property to ensure invariants. Defaults to False."""
    sdf_narrow_band_range: tuple[float, float] = (-0.1, 0.1)
    """The narrow band distance range (inner, outer) for primitive SDF computation."""
    sdf_target_voxel_size: float | None = None
    """Target voxel size for sparse SDF grid.
    If provided, enables primitive SDF generation and takes precedence over
    sdf_max_resolution. Requires GPU since wp.Volume only supports CUDA."""
    sdf_max_resolution: int | None = None
    """Maximum dimension for sparse SDF grid (must be divisible by 8).
    If provided (and sdf_target_voxel_size is None), enables primitive SDF
    generation. Requires GPU since wp.Volume only supports CUDA."""
    sdf_texture_format: str = "uint16"
    """Subgrid texture storage format for the SDF. ``"uint16"``
    (default) stores subgrid voxels as 16-bit normalized textures (half
    the memory of ``"float32"``). ``"float32"`` stores full-precision
    values. ``"uint8"`` uses 8-bit textures for minimum memory."""
    is_hydroelastic: bool = False
    """Whether the shape collides using SDF-based hydroelastics. For hydroelastic collisions, both participating shapes must have is_hydroelastic set to True. Defaults to False.

    .. note::
        Hydroelastic collision handling only works with volumetric shapes and in particular will not work for shapes like flat meshes or cloth.
        This flag will be automatically set to False for planes and heightfields in :meth:`ModelBuilder.add_shape`.
    """
    kh: float = 1.0e10
    """Contact stiffness coefficient for hydroelastic collisions. Used by MuJoCo, Featherstone, SemiImplicit when is_hydroelastic is True.

    .. note::
        For MuJoCo, stiffness values will internally be scaled by masses.
        Users should choose kh to match their desired force-to-penetration ratio.
    """
    '''

    # TODO: sift through the shape config options and determine which ones we want to set
    # cfg = newton.ModelBuilder.ShapeConfig()

    chassis_body = builder.add_link()
    builder.add_shape_box(chassis_body, hx=0.05, hy=0.1, hz=0.01, xform=xform, color=wp.vec3(0.8, 0.1, 0.1))
    joint = builder.add_joint_free(parent=-1, child=chassis_body)
    # joint = builder.add_joint_fixed(parent=-1, child=chassis_body)

    wheels = [
        (wp.vec3(0.025, -0.12, xform[2]), "a"),
        (wp.vec3(0.025, 0.12, xform[2]), "b"),
        (wp.vec3(-0.025, -0.12, xform[2]), "c"),
        (wp.vec3(-0.025, 0.12, xform[2]), "d"),
    ]

    wheel_bodies = []
    wheel_joints = []

    for wh_xform, wh_label in wheels:
        wheel_body = builder.add_link()

        builder.add_shape_sphere(wheel_body, radius=0.02)
        wheel_bodies.append(wheel_body)

        # has_drive = dim.target_ke != 0.0 or dim.target_kd != 0.0
        # if not has_drive: return JointTargetMode.NONE
        # if force_position_velocity and (target_ke != 0.0 and target_kd != 0.0): return JointTargetMode.POSITION_VELOCITY
        # elif target_ke != 0.0: return JointTargetMode.POSITION
        # elif target_kd != 0.0: return JointTargetMode.VELOCITY
        # else: return JointTargetMode.EFFORT

        # ke is the joint stiffness
        # kd is the joint damping

        wheel_joint = builder.add_joint_revolute(
            parent=chassis_body,
            child=wheel_body,
            parent_xform=wp.transform(p=wh_xform),
            axis=newton.Axis.Y,
            actuator_mode=newton.JointTargetMode.VELOCITY,
            target_vel=10.47 * wp.sign(wh_xform[1]),
            target_kd=0.1,
            # label=wh_label,
        )
        wheel_joints.append(wheel_joint)

        qd_index = builder.joint_qd_start[wheel_joint]

        # max_e, sat, v_lim = 100.0, 80.0, 20.0
        # max_e, sat, v_lim = 0.1, 0.1, 0.1
        max_e, sat, v_lim = 200.0, 80.0, 15.0
        dc_args = {"saturation_effort": sat, "velocity_limit": v_lim, "max_motor_effort": max_e}

        # builder.add_actuator(ControllerPD, index=qd_index, kp=10, kd=0.1, delay_steps=None, clamping=None)
        # builder.add_actuator(ControllerPD, index=qd_index, kd=0.01, delay_steps=None, clamping=None)
        builder.add_actuator(
            ControllerPD,
            index=qd_index,
            kd=0.01,
            delay_steps=None,
            clamping=[(ClampingDCMotor, dc_args)],
        )

    builder.add_articulation([joint] + wheel_joints)

    # TODO: get index of wheel_joint and set qd

    # q_start = builder.joint_q_start[wheel_joint]
    # q_dim = builder.joint_dof_dim[wheel_joint]
    # q_count = sum(q_dim)
    # q = [0.5]
    # builder.joint_q[q_start : q_start + q_count] = q

    # qd_start = builder.joint_qd_start[wheel_joint]
    # qd_dim = builder.joint_dof_dim[wheel_joint]
    # qd_count = sum(qd_dim)
    # qd = [5]
    # builder.joint_qd[qd_start : qd_start + qd_count] = qd

    # # f = ma --> a = f/m (constant acceleration)
    # for wh_joint in wheel_joints:
    #     f_start = builder.joint_qd_start[wh_joint]
    #     f_dim = builder.joint_dof_dim[wh_joint]
    #     f_count = sum(f_dim)
    #     f = [0.001]
    #     builder.joint_f[f_start : f_start + f_count] = f

    # builder.joint_q[-1] = 0.5
    # builder.joint_qd[-1] = 0.1

    # # Add a site at body origin
    # imu_site = builder.add_site(
    #     body=body,
    #     label="imu"
    # )

    # # Add a site with offset and rotation
    # camera_site = builder.add_site(
    #     body=body,
    #     xform=wp.transform(
    #         wp.vec3(0.5, 0, 0.2),  # Position
    #         wp.quat_from_axis_angle(wp.vec3(0, 1, 0), 3.14159/4)  # Orientation
    #     ),
    #     type=newton.GeoType.BOX,
    #     scale=(0.05, 0.05, 0.02),
    #     visible=True,
    #     label="camera"
    # )

    # TODO: Can probably just return the builder
    return builder, chassis_body, wheel_bodies, wheel_joints

    builder = newton.ModelBuilder()

    #
    # region chassis
    #

    ch_scale = wp.vec3(ch_length / 2, ch_width / 2, ch_height / 2)
    ch_mass, _, ch_inertia = compute_inertia_shape(newton.GeoType.BOX, ch_scale, None, ch_density)

    # TODO: consider label, lock_inertia, is_kinematic, and custom_attributes
    chassis = builder.add_body(xform=xform, inertia=ch_inertia, mass=ch_mass)

    # NOTE: set `density=0.0` so that the mass from above is used
    # TODO: look at other shape config options (friction, restitution, etc)
    ch_cfg = newton.ModelBuilder.ShapeConfig(density=0.0)
    builder.add_shape_box(body=chassis, hx=ch_length / 2, hy=ch_width / 2, hz=ch_height / 2, cfg=ch_cfg)

    # TODO: only add the free-joint when using a solver with generalized coordinates
    # This issues a warning when using MuJoCo and I did not expect that
    # builder.add_joint_free(chassis)

    #
    # Build four wheels
    #

    wh_scale = wp.vec3(wh_radius, 0.0, 0.0)
    wh_mass, _, wh_inertia = compute_inertia_shape(newton.GeoType.SPHERE, wh_scale, None, wh_density)

    # TODO: look at other shape config options (friction, restitution, etc)
    wh_cfg = newton.ModelBuilder.ShapeConfig(density=0.0)

    wh_vertical_offset = xform[2]
    print(f"==>> wh_vertical_offset: {wh_vertical_offset}")
    print(xform)
    print(xform[0])
    print(xform[1])

    wh_offsets = [
        ((ch_length / 2, ch_width / 2, wh_vertical_offset), "FrontRight"),
        ((-ch_length / 2, ch_width / 2, wh_vertical_offset), "FrontLeft"),
        ((ch_length / 2, -ch_width / 2, wh_vertical_offset), "RearRight"),
        ((-ch_length / 2, -ch_width / 2, wh_vertical_offset), "RearLeft"),
    ]

    # # NOTE: newton defaults to z-up for cylinders
    # wh_rotation = wp.quat_from_axis_angle(wp.vec3(1.0, 0.0, 0.0), pi / 2)

    wheels = []
    wheel_joints = []

    for wh_offset, wh_name in wh_offsets:
        # TODO: add support for cylindrical wheels when newton supports it
        # wh_xform = wp.transform(p=wh_offset, q=wh_rotation)
        # builder.add_shape_cylinder(wheel, radius=wh_radius, half_height=wh_thickness / 2)

        wh_xform = wp.transform(p=wh_offset)
        wheel = builder.add_body(xform=wh_xform, inertia=wh_inertia, mass=wh_mass)
        # wheel = builder.add_body(inertia=wh_inertia, mass=wh_mass)
        builder.add_shape_sphere(wheel, radius=wh_radius, cfg=wh_cfg)

        joint = builder.add_joint_revolute(
            parent=chassis,
            child=wheel,
            # parent_xform=wp.transform(p=wh_offset, q=wh_rotation),
            parent_xform=wp.transform(p=wh_offset),
            # parent_xform=wp.transform(p=(0.2, 0.2, 0)),
            # child_xform=wp.transform(p=(-0.2, 0, 0)),
            # axis=newton.Axis.Y,
            actuator_mode=newton.JointTargetMode.VELOCITY,
            # armature=0.0,
            collision_filter_parent=True,
            label=f"wheel_{wh_name}",
        )

        wheels.append(wheel)
        wheel_joints.append(joint)

    return builder, chassis, wheels, wheel_joints


"""
    parent: The index of the parent body.
    child: The index of the child body.
    parent_xform: The transform from the parent body frame to the joint parent anchor frame.
    child_xform: The transform from the child body frame to the joint child anchor frame.
    axis: The axis of rotation in the joint parent anchor frame, which is
        the parent body's local frame transformed by `parent_xform`. It can be a :class:`JointDofConfig` object
        whose settings will be used instead of the other arguments.
    target_pos: The target position of the joint.
    target_vel: The target velocity of the joint.
    target_ke: The stiffness of the joint target.
    target_kd: The damping of the joint target.
    limit_lower: The lower limit of the joint. If None, the default value from ``ModelBuilder.default_joint_cfg.limit_lower`` is used.
    limit_upper: The upper limit of the joint. If None, the default value from ``ModelBuilder.default_joint_cfg.limit_upper`` is used.
    limit_ke: The stiffness of the joint limit. If None, the default value from ``ModelBuilder.default_joint_cfg.limit_ke`` is used.
    limit_kd: The damping of the joint limit. If None, the default value from ``ModelBuilder.default_joint_cfg.limit_kd`` is used.
    armature: Artificial inertia added around the joint axis. If None, the default value from ``ModelBuilder.default_joint_cfg.armature`` is used.
    effort_limit: Maximum effort (force/torque) the joint axis can exert. If None, the default value from ``ModelBuilder.default_joint_cfg.effort_limit`` is used.
    velocity_limit: Maximum velocity the joint axis can achieve. If None, the default value from ``ModelBuilder.default_joint_cfg.velocity_limit`` is used.
    friction: Friction coefficient for the joint axis. If None, the default value from ``ModelBuilder.default_joint_cfg.friction`` is used.
    label: The label of the joint.
    collision_filter_parent: Whether to filter collisions between shapes of the parent and child bodies.
    enabled: Whether the joint is enabled.
    custom_attributes: Dictionary of custom attribute values for JOINT, JOINT_DOF, or JOINT_COORD frequency attributes.

    parent: int,
    child: int,
    parent_xform: Transform | None = None,
    child_xform: Transform | None = None,
    axis: AxisType | Vec3 | JointDofConfig | None = None,
    target_pos: float | None = None,
    target_vel: float | None = None,
    target_ke: float | None = None,
    target_kd: float | None = None,
    limit_lower: float | None = None,
    limit_upper: float | None = None,
    limit_ke: float | None = None,
    limit_kd: float | None = None,
    armature: float | None = None,
    effort_limit: float | None = None,
    velocity_limit: float | None = None,
    friction: float | None = None,
    actuator_mode: JointTargetMode | None = None,
    label: str | None = None,
    collision_filter_parent: bool = True,
    enabled: bool = True,
    custom_attributes: dict[str, Any] | None = None,
    **kwargs,
"""
