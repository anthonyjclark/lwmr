from collections.abc import Generator

MotorTuple = tuple[float, float, float, float]
GeneratorType = Generator[MotorTuple, None, None]


def control_sequence_generator(
    cmds: list[MotorTuple], steps: list[int] | int, loop: bool = True
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
