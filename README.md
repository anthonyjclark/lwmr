# Legged-Wheeled Mobile Robot

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

> [!NOTE]
> The conda-forge build of Viser may not include its compiled browser client.
> The first tutorial run can therefore use `nodeenv` to download Node.js and
> build the client. Run it once with network access before using the project on
> an offline system such as a remote server.

### Running the Tutorials

```bash
cd tutorial

# Run the basic example
pixi run python basic_gym.py --quiet --steps 200

# Or watch it live in Newton's native OpenGL viewer (close the window to exit)
pixi run python basic_gym.py --viewer gl --steps 200

# Serve the visualization
# (Consider doing so in a separate terminal so that you can rerun the simulation without restarting the server.)
# (Use port forwarding when running on a remote server: `ssh -L PORT:localhost:PORT USER@SERVER`)
pixi run python -m http.server

# Other things to try
pixi run python basic_gym.py --help
pixi run python basic_gym.py --quiet --steps 200
pixi run python basic_gym.py --quiet --steps 200 --num-legs 3
pixi run python basic_gym.py --quiet --steps 200 --num-legs 3 --num-worlds 2
pixi run python basic_gym.py --quiet --steps 200 --num-legs 3 --num-worlds 16 --add-step
pixi run python basic_gym.py --quiet --steps 200 --num-legs 3 --num-worlds 16 --add-step --fixed_base
pixi run python basic_gym.py --quiet --steps 200 --num-legs 3 --num-worlds 16 --add-step --wh-radius 0.02
```

After starting the server, open the
[LWMR plane recording](http://localhost:8000/?playbackPath=recordings/lwmr_plane.viser)
in your browser. The OpenGL option displays the simulation live and does not
create or require a Viser recording. Its **Reset** button rebuilds the
simulation and restarts the tutorial from the beginning while keeping the
viewer window open. **Pause** and **Step** are part of Newton's default OpenGL
interface; the tutorial respects both controls.

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
