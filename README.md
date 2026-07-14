# Legged-Wheeled Mobile Robot

Note: we are switching to `pixi` for managing packages since our HPC is out of date and incompatible with PYPI packages.

## Using Pixi

### One-Time Setup

```bash
# Clone this repository
gh repo clone anthonyjclark/lwmr

# Install dependencies (requires pixi >= 0.72.0)
pixi install

# Testing the installation
pixi run python -m lwmr
```

### Running the Tutorials

```bash
cd tutorial

# Run the basic example
python basic_gym.py --quiet --steps 200

# Serve the visualization
# (Consider doing so in a separate terminal so that you can rerun the simulation without restarting the server.)
# (Use port forwarding when running on the HPC: `ssh -L PORT:localhost:PORT USER@SERVER`)
python -m http.server
```

### Initial Project Setup

<details>
<summary>Click to see initial setup instructions.</summary>

```bash
# `uv` is useful for creating Python packages
uv init --python 3.12 --bare
# You'll need to manually add packages to pixi.toml in [pypi-dependencies]
uv init --package packages/lwmr

# We'll use pixi to manage dependencies and run the examples
pixi init . --format pixi

# See pixi.toml for manually added platform ([workspace.platforms])

# Manually install newton and its extras (newton-all contains most)
pixi add python=3.12
pixi add pytorch-gpu
pixi add newton-all gymnasium loguru tyro
```

</details>

### TODO

- Remove `viser-client/` from the project root?
- Cleanup the `lwmr` package
  - Rename to `lwmr-newton`? (or `lwmr-newton-gym`?)
  - Should lwmr be top-level instead of in `packages/`?
