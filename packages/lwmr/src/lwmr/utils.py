from collections.abc import Generator
from dataclasses import dataclass
from math import pi
from pathlib import Path
from socketserver import TCPServer

from newton import Model, ModelBuilder
from newton.viewer import ViewerViser
from numpy import ndarray


# https://github.com/NVIDIA/warp/blob/542ef1733c125378b58bf84aab4045fbd19c6de5/warp/_src/math.py#L190
def quat_to_rpy(q: ndarray) -> tuple[float, float, float]:
    from numpy import asin, atan2, clip

    x = q[0]
    y = q[1]
    z = q[2]
    w = q[3]
    t0 = 2.0 * (w * x + y * z)
    t1 = 1.0 - 2.0 * (x * x + y * y)
    roll_x = atan2(t0, t1)

    t2 = 2.0 * (w * y - z * x)
    t2 = clip(t2, -1.0, 1.0)
    pitch_y = asin(t2)

    t3 = 2.0 * (w * z + x * y)
    t4 = 1.0 - 2.0 * (y * y + z * z)
    yaw_z = atan2(t3, t4)

    return (roll_x, pitch_y, yaw_z)


def world_to_body(yaw: float, vec_world: ndarray) -> ndarray:
    from numpy import array, cos, float32, sin

    c, s = cos(-yaw), sin(-yaw)
    R = array([[c, -s], [s, c]], dtype=float32)
    return R @ vec_world


# region Control

MotorTuple = tuple[float, float, float, float]
GeneratorType = Generator[MotorTuple, None, None]


def control_sequence_generator(
    cmds: list[MotorTuple],
    steps: list[int] | int,
    loop: bool = True,
) -> GeneratorType:

    if isinstance(steps, int):
        steps = [steps] * len(cmds)

    control_iter = zip(steps, cmds)
    current_control_steps, current_control = next(control_iter)
    while True:
        yield current_control

        current_control_steps -= 1
        if current_control_steps <= 0:
            try:
                current_control_steps, current_control = next(control_iter)
            except StopIteration:
                if not loop:
                    break
                control_iter = zip(steps, cmds)
                current_control_steps, current_control = next(control_iter)


# region Franka


@dataclass
class FrankaEmikaPandaConfig:
    FINGER_CLOSED = 0.0
    FINGER_OPEN = 0.04

    HOME_Q_ARM = [0.0, 0.02, 0.0, -2.37, 0.0, 2.39, pi / 4]
    HOME_Q_GRIPPER = [FINGER_OPEN, FINGER_OPEN]
    HOME_Q = HOME_Q_ARM + HOME_Q_GRIPPER

    DOF_COUNT = len(HOME_Q)
    ARM_DOF_COUNT = len(HOME_Q_ARM)

    EFFORT_LIMITS = [87, 87, 87, 87, 12, 12, 12, 100, 100]
    MUJOCO_ARMATURE = [0.195] * 4 + [0.074] * 3 + [0.1] * 2

    TARGET_KE = [900, 900, 700, 700, 400, 400, 400, 100, 100]
    TARGET_KD = [90, 90, 70, 70, 40, 40, 40, 10, 10]
    BODY_GRAVCOMP = 1.0


_config = FrankaEmikaPandaConfig()


def enable_target_control(builder: ModelBuilder):
    """Enable position targets and MuJoCo gravity compensation for Franka."""
    builder.joint_target_pos[: _config.DOF_COUNT] = _config.HOME_Q
    builder.joint_target_ke[: _config.DOF_COUNT] = _config.TARGET_KE
    builder.joint_target_kd[: _config.DOF_COUNT] = _config.TARGET_KD

    joint_actgravcomp = builder.custom_attributes["mujoco:jnt_actgravcomp"].values
    for dof_index in range(_config.DOF_COUNT):
        joint_actgravcomp[dof_index] = True  # type: ignore

    body_gravcomp = builder.custom_attributes["mujoco:gravcomp"].values
    for body_index, label in enumerate(builder.body_label):
        if label.startswith("fr3/") and label not in {"fr3/base", "fr3/fr3_link0"}:
            body_gravcomp[body_index] = _config.BODY_GRAVCOMP  # type: ignore


def set_initial_state(builder: ModelBuilder):
    """Set the initial joint positions and armature."""
    builder.joint_q[: _config.DOF_COUNT] = _config.HOME_Q
    builder.joint_effort_limit[: _config.DOF_COUNT] = _config.EFFORT_LIMITS
    builder.joint_armature[: _config.DOF_COUNT] = _config.MUJOCO_ARMATURE


# region Viewer

# Path is hardcoded in Newton's viewer for generating docs
RECORDING_BASE_PATH = Path("docs/_static/")


def create_viewer_viser(
    file_stem: str,
    model: Model,
    quiet: bool = True,
    port: int | None = None,
    overwrite: bool = True,
    max_worlds: int | None = None,
) -> ViewerViser:

    import newton
    import warp as wp

    rec_path = str(RECORDING_BASE_PATH / f"{file_stem}.viser")

    if not overwrite:
        i = 1
        while Path(rec_path).exists():
            rec_path = str(RECORDING_BASE_PATH / f"{file_stem}_{i:02d}.viser")
            i += 1

    # Default port for ViewerViser is 8080
    port = port if port is not None else 8080

    if quiet:
        import rich

        console = rich.get_console()
        with console.capture() as _:
            viewer = newton.viewer.ViewerViser(record_to_viser=rec_path, verbose=False, port=port)
    else:
        print(f"Recording to {rec_path}...")
        viewer = newton.viewer.ViewerViser(record_to_viser=rec_path, verbose=False, port=port)

    viewer.set_model(model, max_worlds=max_worlds)

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

    return viewer


# region Colab

# Let the OS assign a free port
_port_to_serve = 0


class ColabViserHTTPServer(TCPServer):
    allow_reuse_address = True


def _start_server_colab():
    from http.server import SimpleHTTPRequestHandler

    global _port_to_serve

    viser_client = "./lwmr/viser-client/index.html"

    if not Path(viser_client).is_file():
        raise FileNotFoundError(f"File '{viser_client}' not found.")

    Handler = SimpleHTTPRequestHandler
    with ColabViserHTTPServer(("", _port_to_serve), Handler) as httpd:
        _port_to_serve = httpd.socket.getsockname()[1]
        print(f"Starting viser-client on port {_port_to_serve}")
        httpd.serve_forever()


def start_server_thread_colab() -> int:
    import threading
    from time import sleep

    global _port_to_serve

    server_thread = threading.Thread(target=_start_server_colab)
    server_thread.daemon = True
    server_thread.start()

    # Wait for the server to start and get the assigned port
    while _port_to_serve == 0:
        sleep(0.1)

    print(f"Server started on port: {_port_to_serve}")
    return _port_to_serve
