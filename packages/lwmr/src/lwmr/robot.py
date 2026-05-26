from dataclasses import dataclass

import newton
import warp as wp
from newton._src.core.types import Transform
from newton.actuators import ClampingDCMotor, ControllerPD


@dataclass
class LwmrRobotConfig:
    ch_width: float = 0.3
    ch_length: float = 0.15
    ch_height: float = 0.02
    ch_density: float = 1000.0
    wh_radius: float = 0.03
    wh_density: float = 1000.0
    lg_radius: float = 0.01
    lg_offset: float = 0.03
    num_legs: int = 0
    add_imu: bool = False

    # TODO: only specify wh_thickness for cylindrical wheels
    # wh_thickness: float,
    # TODO: leg params


def add_lwmr_robot(
    builder: newton.ModelBuilder,
    xform: Transform,
    config: LwmrRobotConfig,
    fixed_base: bool = False,
    ch_color: wp.vec3 = wp.vec3(0.8, 0.1, 0.1),
    wh_color: wp.vec3 = wp.vec3(0.1, 0.1, 0.8),
    lg_color: wp.vec3 = wp.vec3(0.1, 0.8, 0.1),
) -> tuple[int, list[int], list[int], list[int]]:
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

    #
    # region chassis
    #

    hx = config.ch_length / 2
    hy = config.ch_width / 2
    hz = config.ch_height / 2

    # TODO: set chassis density (and other properties)

    chassis_body = builder.add_link()
    builder.add_shape_box(chassis_body, hx=hx, hy=hy, hz=hz, xform=xform, color=ch_color)

    if fixed_base:
        joint = builder.add_joint_fixed(parent=-1, child=chassis_body)
    else:
        joint = builder.add_joint_free(parent=-1, child=chassis_body)

    #
    # region wheels
    #

    drop_z: float = xform[2]  # type: ignore
    wh_x_offset = config.ch_length / 2
    wh_y_offset = config.ch_width / 2 + config.wh_radius

    wheels = [
        (wh_x_offset, wh_y_offset, "wheel_front_left"),
        (wh_x_offset, -wh_y_offset, "wheel_front_right"),
        (-wh_x_offset, wh_y_offset, "wheel_rear_left"),
        (-wh_x_offset, -wh_y_offset, "wheel_rear_right"),
    ]

    wheel_bodies = []
    wheel_joints = []
    wheel_qd_indices = []

    for x, y, wh_label in wheels:
        # # NOTE: newton defaults to z-up for cylinders
        # wh_rotation = wp.quat_from_axis_angle(wp.vec3(1.0, 0.0, 0.0), pi / 2)

        wheel_body = builder.add_link()
        builder.add_shape_sphere(wheel_body, radius=config.wh_radius, color=wh_color)
        wheel_bodies.append(wheel_body)

        # Add legs at fixed positions around the wheel
        for i in range(config.num_legs):
            angle = 2 * wp.pi * i / config.num_legs
            lg_x = config.lg_offset * wp.cos(angle)
            lg_z = config.lg_offset * wp.sin(angle)
            builder.add_shape_sphere(
                wheel_body,
                radius=config.lg_radius,
                color=lg_color,
                xform=wp.transform(p=wp.vec3(lg_x, 0, lg_z)),
            )

        # has_drive = dim.target_ke != 0.0 or dim.target_kd != 0.0
        # if not has_drive: return JointTargetMode.NONE
        # if force_position_velocity and (target_ke != 0.0 and target_kd != 0.0): return JointTargetMode.POSITION_VELOCITY
        # elif target_ke != 0.0: return JointTargetMode.POSITION
        # elif target_kd != 0.0: return JointTargetMode.VELOCITY
        # else: return JointTargetMode.EFFORT

        # ke is the joint stiffness
        # kd is the joint damping

        #
        # region joints
        #

        """TODO: figure out these parameters
        effort_limit: Maximum effort (force/torque) the joint axis can exert. If None, the default value from ``ModelBuilder.default_joint_cfg.effort_limit`` is used.
        velocity_limit: Maximum velocity the joint axis can achieve. If None, the default value from ``ModelBuilder.default_joint_cfg.velocity_limit`` is used.
        friction: Friction coefficient for the joint axis. If None, the default value from ``ModelBuilder.default_joint_cfg.friction`` is used.
        custom_attributes: Dictionary of custom attribute values for JOINT, JOINT_DOF, or JOINT_COORD frequency attributes.
        """

        wheel_joint = builder.add_joint_revolute(
            parent=chassis_body,
            child=wheel_body,
            parent_xform=wp.transform(p=wp.vec3(x, y, drop_z)),
            axis=newton.Axis.Y,
            actuator_mode=newton.JointTargetMode.VELOCITY,
            # TODO: initialize to zero (don't set the value) and use control?
            # target_vel=10.47 * wp.sign(y),
            # target_vel=10.47,
            # TODO: add an argument for target_kd (do we need it?)
            target_kd=0.1,
            label=wh_label,
        )
        wheel_joints.append(wheel_joint)

        # TODO: add parameters for these values
        dc_args = {"saturation_effort": 80.0, "velocity_limit": 15.0, "max_motor_effort": 200.0}

        qd_index = builder.joint_qd_start[wheel_joint]
        builder.add_actuator(
            ControllerPD,
            # MyController,
            index=qd_index,
            # TODO: add an argument for kp
            kd=0.01,
            # TODO: add parameters for delay steps
            # delay_steps=None,
            clamping=[(ClampingDCMotor, dc_args)],
        )
        wheel_qd_indices.append(qd_index)

    builder.add_articulation([joint] + wheel_joints)

    return chassis_body, wheel_bodies, wheel_joints, wheel_qd_indices
