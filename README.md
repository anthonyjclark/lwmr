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
