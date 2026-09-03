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

4. Run from the release folder (so the binary can find `sha1_kernel.ptx`).

  `cd target/release`

  `./rust_sha1_cracker`

## Run In Background With nohup

Use this when you want the cracker to keep running after you close the terminal.

1. Start from the project root.

  `cd rust_sha1_cracker`

2. Start the process with `nohup` and redirect output to a log file.

  Example:

  `nohup cargo run --release > output-7char-all-B4CFC8DC918B7CBF9F7653B1DDB0540D7748C086.log 2>&1 &`

  You can change the log file name to match your run settings.

3. Save the process ID (PID) shown by the shell (for example: `[1] 12345`).

## Check Progress

1. Follow live logs:

  `tail -f output-7char-all-B4CFC8DC918B7CBF9F7653B1DDB0540D7748C086.log`

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

Edit [src/main.rs](src/main.rs) before building/running:

1. `TARGET_HASH`: SHA-1 hash to crack (40 hex chars)
2. `CHARSET`: candidate characters
3. `PWD_LEN_MIN` and `PWD_LEN_MAX`: password length range

Example:

- Try only length 5:
  - `PWD_LEN_MIN = 5`
  - `PWD_LEN_MAX = 5`
- Try 6 through 8:
  - `PWD_LEN_MIN = 6`
  - `PWD_LEN_MAX = 8`

After any config change, run `./build.sh` again.

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
