# Rust SHA1 Cracker

This project is a simple SHA1 hash cracker written in Rust.

## Requirements
- Ubuntu Server (any recent version)
- Rust toolchain (cargo, rustc)
- Git (optional, for cloning the repository)

## Step-by-Step Instructions

### 1. Install Rust
If Rust is not installed, run the following commands:

```
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source $HOME/.cargo/env
```

You may need to restart your terminal or run `source $HOME/.cargo/env` to update your environment variables.

### 2. Clone or Transfer the Project
- **If using Git:**
  ```
  git clone <your-repo-url>
  cd rust_sha1_cracker
  ```
- **If transferring manually:**
  - Use `scp` or SFTP to copy the project folder to your server.
  - Example:
    ```
    scp -r /path/to/rust_sha1_cracker user@your-server:/home/user/
    cd rust_sha1_cracker
    ```

### 3. Build the Project
Run:
```
cargo build --release
```
The compiled binary will be in the `target/release/` directory.

### 4. Run the Program
Run the program with:
```
cargo run --release
```
Or, to run the compiled binary directly:
```
./target/release/rust_sha1_cracker
```

### 5. (Optional) Passing Arguments
If your program expects arguments (e.g., a hash to crack), run:
```
cargo run --release -- <arguments>
```
Or:
```
./target/release/rust_sha1_cracker <arguments>
```

---

## Troubleshooting
- If you get a "command not found" error for `cargo`, ensure Rust is installed and your `$PATH` is set up.
- For permission errors, you may need to run `chmod +x ./target/release/rust_sha1_cracker`.

## References
- [Rust Installation Guide](https://www.rust-lang.org/tools/install)
- [Cargo Book](https://doc.rust-lang.org/cargo/)
