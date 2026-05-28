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
