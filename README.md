# Rust SHA1 Cracker (CUDA + Rust)

This project brute-forces a SHA-1 hash using a CUDA kernel and a Rust host program.

## Prerequisites

1. NVIDIA GPU with CUDA support
2. NVIDIA driver installed and working
3. CUDA Toolkit installed (must include `nvcc`)
4. Rust toolchain installed (`cargo`, `rustc`)
5. Linux shell environment (the provided script targets Ubuntu-style environments)

## Quick Start

1. Go to the project folder.

  `cd rust_sha1_cracker`

2. Make the build script executable (first time only).

  `chmod +x build.sh`

3. Build everything.

  `./build.sh`

  This does two things:
  - Compiles `sha1_kernel.cu` to `sha1_kernel.ptx`
  - Builds the Rust binary in release mode

4. Run from the release folder (so the binary can find `sha1_kernel.ptx` and
   `known_passwords.txt`). `--hash` is required.

  `cd target/release`

  `./rust_sha1_cracker --hash <YOUR_TARGET_HASH>`

  Running `./rust_sha1_cracker` from anywhere else (e.g. the project root) fails with
  `No such file or directory` — the binary only exists under `target/release/` after `./build.sh`.

## Command-Line Options

`--hash` is required on every run. The other options have defaults baked into
[src/main.rs](src/main.rs) (edit and rebuild to change the defaults), but you can override
them per run without rebuilding:

| Flag | Short | Required | Description | Example |
|---|---|---|---|---|
| `--hash` | `-H` | Yes | Target SHA-1 hash (40 hex chars, `sha1(utf16le($pass))`) | `--hash B4CFC8DC918B7CBF9F7653B1DDB0540D7748C086` |
| `--length-min` | | No | Minimum password length to try | `--length-min 6` |
| `--length-max` | | No | Maximum password length to try | `--length-max 8` |
| `--charset` | `-c` | No | Character set: `letters`, `numbers`, `alphanumeric`, `all` | `--charset alphanumeric` |

Examples:

- Try known Dell BIOS-style alphanumeric passwords, length 6-8:

  `./rust_sha1_cracker --hash B4CFC8DC918B7CBF9F7653B1DDB0540D7748C086 --charset alphanumeric --length-min 6 --length-max 8`

- See all options:

  `./rust_sha1_cracker --help`

## Run In Background With nohup

Use this when you want the cracker to keep running after you close the terminal.

1. Go to the release folder (so the binary can find `sha1_kernel.ptx` and `known_passwords.txt`).

  `cd rust_sha1_cracker/target/release`

2. Start the process with `nohup`, passing your crack parameters, and redirect output to a
   log file. `--hash` is required; see [Command-Line Options](#command-line-options) for the rest.

  Example:

  `nohup ./rust_sha1_cracker --hash FA3569135BCE3660ED2C3CB9E977790BED926E9D --charset alphanumeric --length-min 9 --length-max 10 > output-9to10char-alphanumeric-FA3569135BCE3660ED2C3CB9E977790BED926E9D.log 2>&1 &`

  Name the log file after the run's parameters (hash/charset/length range) so it stays
  identifiable later, and add new ones to `.gitignore` (already covers `*.log`) instead of
  committing them.

3. Save the process ID (PID) shown by the shell (for example: `[1] 12345`).

## Check Progress

1. Follow live logs:

  `tail -f output-9to10char-alphanumeric-FA3569135BCE3660ED2C3CB9E977790BED926E9D.log`

2. Confirm process is still running:

  `ps -fp <PID>`

3. Find the process by name if you do not have the PID:

  `pgrep -af rust_sha1_cracker`

4. Check GPU utilization while it runs:

  `nvidia-smi`

## Cancel / Stop A Running Job

1. Graceful stop with PID:

  `kill <PID>`

2. Force stop if it does not exit:

  `kill -9 <PID>`

3. Stop by process name (when PID is unknown):

  `pkill -f rust_sha1_cracker`

## Configure What To Crack

Most options are now CLI flags (see [Command-Line Options](#command-line-options)) and do not
require a rebuild: `--hash`, `--charset`, `--length-min`, `--length-max`.

To change other defaults, edit [src/main.rs](src/main.rs) before building:

1. `TARGET_HASH`, `PWD_LEN_MIN`, `PWD_LEN_MAX`: default values used when the matching CLI flag is omitted
2. `CharsetOption::chars()`: edit the character lists behind `letters` / `numbers` / `alphanumeric` / `all`

After any config change, run `./build.sh` again.

## GPU Occupancy And Launch Configuration

The grid/block size is no longer hard-coded. At startup the program queries the GPU
(`cuOccupancyMaxPotentialBlockSize` / `cuOccupancyMaxActiveBlocksPerMultiprocessor` via `cust`)
and prints a line like:

```
[*] GPU occupancy: 20 SMs | 32 regs/thread | 4 active block(s)/SM (100% theoretical occupancy)
```

This adapts automatically to whatever GPU the binary runs on — no manual tuning needed when
moving between machines. To independently verify/profile on the server:

- Register usage per thread. Must compile to a cubin (`-cubin`), not `-ptx` — `-ptx` stops
  nvcc at the `cicc` stage and never invokes `ptxas`, so `-Xptxas -v` prints nothing with it:

  `nvcc -Xptxas -v -arch=sm_121 --use_fast_math -O3 -cubin sha1_kernel.cu -o /tmp/sha1_kernel.cubin`

  Look for the `Used XX registers` line and `N bytes spill stores/loads` (spills should be 0).

- GPU topology / SM count / compute capability:

  `nvidia-smi --query-gpu=name,compute_cap,clocks.max.sm --format=csv`

- Achieved occupancy and stall reasons while the binary is running (requires Nsight Compute,
  install via CUDA toolkit or `sudo apt install nsight-compute` if packaged for your distro):

  `sudo ncu --metrics sm__warps_active.avg.pct_of_peak_sustained_active ./rust_sha1_cracker --length-max 5`

  (use a small `--length-max` here just to get one short profiling run instead of a multi-day one)

- Live SM/memory utilization while a full run is in progress:

  `nvidia-smi dmon -s u`

  This streams one line per second and only produces meaningful numbers while
  `rust_sha1_cracker` is actually running — start it in one terminal, then run `dmon` in a
  second terminal alongside it. Columns to watch:

  ```
  # gpu    sm   mem   enc   dec  jpg   ofa
  # Idx     %     %     %     %    %     %
      0    99    46     -     -    -     -
  ```

  - `sm`: streaming multiprocessor utilization — should sit near 100 during a brute-force run;
    if it's noticeably lower, the GPU is stalling (e.g. on memory or launch overhead) instead of
    computing.
  - `mem`: memory controller utilization — this kernel is compute-bound, so this can safely be
    much lower than `sm`.
  - Press `Ctrl+C` to stop it, or run `nvidia-smi dmon -s u -c 30` to auto-stop after 30 samples
    instead of streaming indefinitely.

## Try Known Passwords First

Before starting the GPU brute-force search, the program checks a plain-text
wordlist on the CPU: [known_passwords.txt](known_passwords.txt).

- One candidate password per line.
- Lines starting with `#` and empty lines are ignored.
- If a match is found, the program prints it and exits without touching the GPU.
- If the file is missing or no match is found, it falls back to the normal brute-force search.

Run the binary from the same directory as `known_passwords.txt` (e.g. `target/release`)
so it can find the file, or copy it next to the binary like `sha1_kernel.ptx`.

## Verify Environment

Useful checks:

- `nvidia-smi`
- `nvcc --version`
- `cargo --version`

## Troubleshooting

1. `nvcc: command not found`
  - CUDA Toolkit is missing or not in PATH.

2. Runtime cannot find `sha1_kernel.ptx`
  - Run the binary from `target/release`, or copy the PTX next to the binary.

3. Build fails on CUDA architecture
  - Ensure your installed CUDA version supports your GPU architecture.

4. Very low speed
  - Confirm release build was used (`./build.sh`).
  - Confirm GPU is visible in `nvidia-smi`.
