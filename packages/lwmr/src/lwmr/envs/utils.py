import newton
import numpy as np


def create_viewer_viser(rec_path: str, quiet: bool = True, port: int = 8080) -> newton.viewer.ViewerViser:
    import warp as wp

    if quiet:
        import rich

        console = rich.get_console()
        with console.capture() as _:
            viewer = newton.viewer.ViewerViser(record_to_viser=rec_path, verbose=False, port=port)
    else:
        viewer = newton.viewer.ViewerViser(record_to_viser=rec_path, port=port)

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


def quat_to_rpy(q: np.ndarray) -> tuple[float, float, float]:
    # https://github.com/NVIDIA/warp/blob/542ef1733c125378b58bf84aab4045fbd19c6de5/warp/_src/math.py#L190
    x = q[0]
    y = q[1]
    z = q[2]
    w = q[3]
    t0 = 2.0 * (w * x + y * z)
    t1 = 1.0 - 2.0 * (x * x + y * y)
    roll_x = np.atan2(t0, t1)

    t2 = 2.0 * (w * y - z * x)
    t2 = np.clip(t2, -1.0, 1.0)
    pitch_y = np.asin(t2)

    t3 = 2.0 * (w * z + x * y)
    t4 = 1.0 - 2.0 * (y * y + z * z)
    yaw_z = np.atan2(t3, t4)

    return (roll_x, pitch_y, yaw_z)


def world_to_body(yaw, vec_world):
    c, s = np.cos(-yaw), np.sin(-yaw)
    R = np.array([[c, -s], [s, c]], dtype=np.float32)
    return R @ vec_world
