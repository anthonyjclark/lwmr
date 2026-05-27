from collections.abc import Generator

MotorCommand = tuple[float, float, float, float]


def looping_control_sequence_generator(
    cmds: list[MotorCommand], steps: list[int] | int
) -> Generator[MotorCommand, None, None]:

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
                control_iter = zip(steps, cmds)
                current_control_steps, current_control = next(control_iter)
