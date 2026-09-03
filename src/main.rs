use cust::prelude::*;
use cust::memory::DeviceBox;
use cust::device::DeviceAttribute;
use cust::function::{BlockSize, FunctionAttribute};
use clap::{Parser, ValueEnum};
use std::ffi::CStr;
use std::error::Error;
use std::time::Instant;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::fs;
use std::path::Path;

// -----------------------------------------------------------------------
// Configuration defaults – override at runtime via CLI flags (see `Args`)
// -----------------------------------------------------------------------

// Plain-text file, one candidate password per line. Lines starting with
// '#' and empty lines are skipped. Checked on CPU before the GPU
// brute-force search starts.
const KNOWN_PASSWORDS_FILE: &str = "known_passwords.txt";

const PWD_LEN_MIN: usize = 1;
const PWD_LEN_MAX: usize = 8;

// Must match the kernel's __launch_bounds__(256) in sha1_kernel.cu
const THREADS_PER_BLOCK_LIMIT: u32 = 256;

// How often the progress heartbeat prints while a length is running
const PROGRESS_INTERVAL_SECS: u64 = 30;

// -----------------------------------------------------------------------
// Selectable charsets for the brute-force search
// -----------------------------------------------------------------------
#[derive(Copy, Clone, Debug, ValueEnum)]
#[value(rename_all = "kebab-case")]
enum CharsetOption {
    /// Letters only: a-z, A-Z (52 chars)
    Letters,
    /// Digits only: 0-9 (10 chars)
    Numbers,
    /// Letters + digits: a-z, A-Z, 0-9 (62 chars)
    Alphanumeric,
    /// All printable ASCII: letters, digits, symbols, space (95 chars)
    All,
}

impl CharsetOption {
    fn chars(self) -> &'static [u8] {
        match self {
            CharsetOption::Letters => b"abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ",
            CharsetOption::Numbers => b"0123456789",
            CharsetOption::Alphanumeric => {
                b"abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
            }
            CharsetOption::All => {
                b"abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~ "
            }
        }
    }
}

/// GPU-accelerated SHA-1(UTF-16LE(password)) brute-force cracker
#[derive(Parser, Debug)]
#[command(name = "rust_sha1_cracker")]
struct Args {
    /// Target SHA-1 hash to crack (40 hex chars; hashcat mode 170: sha1(utf16le($pass)))
    #[arg(short = 'H', long = "hash")]
    hash: String,

    /// Minimum password length to try
    #[arg(long = "length-min", default_value_t = PWD_LEN_MIN)]
    length_min: usize,

    /// Maximum password length to try
    #[arg(long = "length-max", default_value_t = PWD_LEN_MAX)]
    length_max: usize,

    /// Character set to brute-force with
    #[arg(short = 'c', long = "charset", value_enum, default_value_t = CharsetOption::All)]
    charset: CharsetOption,
}

// -----------------------------------------------------------------------
// Hash a candidate password the same way the target hash was produced:
// SHA-1(UTF-16LE(password)) → 5 big-endian uint32 words
// -----------------------------------------------------------------------
fn hash_words(password: &str) -> [u32; 5] {
    let utf16: Vec<u8> = password.encode_utf16().flat_map(|u| u.to_le_bytes()).collect();
    let digest = sha1_smol::Sha1::from(&utf16).digest().bytes();
    let mut words = [0u32; 5];
    for i in 0..5 {
        words[i] = u32::from_be_bytes(digest[i * 4..i * 4 + 4].try_into().unwrap());
    }
    words
}

// -----------------------------------------------------------------------
// Try a list of known/likely passwords on the CPU before touching the GPU.
// Returns the matching password, if any.
// -----------------------------------------------------------------------
fn try_known_passwords(path: &str, target_u32: &[u32; 5]) -> Option<String> {
    if !Path::new(path).exists() {
        println!("[*] No known-password list found at '{}', skipping straight to brute force.", path);
        return None;
    }

    let contents = fs::read_to_string(path).ok()?;
    let candidates: Vec<&str> = contents
        .lines()
        .map(|l| l.trim())
        .filter(|l| !l.is_empty() && !l.starts_with('#'))
        .collect();

    println!("[*] Trying {} known password(s) from '{}'...", candidates.len(), path);
    for candidate in candidates {
        if &hash_words(candidate) == target_u32 {
            return Some(candidate.to_string());
        }
    }
    println!("[!] No match in known-password list. Falling back to brute force.");
    None
}

// -----------------------------------------------------------------------
// Decode a base-N index back to a password string
// -----------------------------------------------------------------------
fn decode_idx(mut n: u64, len: usize, charset: &[u8]) -> String {
    let cs = charset.len() as u64;
    let mut chars = Vec::with_capacity(len);
    for _ in 0..len {
        chars.push(charset[(n % cs) as usize] as char);
        n /= cs;
    }
    chars.reverse();
    chars.into_iter().collect()
}

fn main() -> Result<(), Box<dyn Error>> {
    let args = Args::parse();
    let charset_bytes = args.charset.chars();
    let charset_len = charset_bytes.len();
    assert!(charset_len <= 96, "charset too large (max 96)");
    assert!(
        args.length_min >= 1 && args.length_min <= args.length_max,
        "length-min must be >= 1 and <= length-max"
    );
    assert!(
        args.length_max <= 16,
        "length-max > 16 exceeds the kernel's fixed-size char buffer"
    );

    // ------------------------------------------------------------------- //
    // 1.  Parse target hash → 5 big-endian uint32 words
    // ------------------------------------------------------------------- //
    let target_raw = hex::decode(&args.hash)?;
    assert_eq!(target_raw.len(), 20, "SHA-1 hash must be 20 bytes");
    let mut target_u32 = [0u32; 5];
    for i in 0..5 {
        target_u32[i] = u32::from_be_bytes(
            target_raw[i*4..i*4+4].try_into().unwrap()
        );
    }

    // ------------------------------------------------------------------- //
    // 1b. Try known/likely passwords first (fast, CPU-only, no GPU needed)
    // ------------------------------------------------------------------- //
    if let Some(password) = try_known_passwords(KNOWN_PASSWORDS_FILE, &target_u32) {
        println!("\n[FOUND] Password: '{}' (from known-password list)", password);
        return Ok(());
    }

    // ------------------------------------------------------------------- //
    // 2.  CUDA initialisation
    // ------------------------------------------------------------------- //
    cust::init(cust::CudaFlags::empty())?;
    let device = Device::get_device(0)?;
    let _ctx = Context::new(device)?;

    let module = Module::from_file("sha1_kernel.ptx")?;
    let func = module.get_function("sha1_kernel")?;

    // ------------------------------------------------------------------- //
    // 3.  Query GPU occupancy and derive a launch configuration from it,
    //     instead of hard-coded grid/block sizes. This adapts automatically
    //     to whatever GPU the binary runs on.
    // ------------------------------------------------------------------- //
    let sm_count = device.get_attribute(DeviceAttribute::MultiprocessorCount)? as u32;
    let max_threads_per_sm = device.get_attribute(DeviceAttribute::MaxThreadsPerMultiprocessor)? as u32;
    let regs_per_thread = func.get_attribute(FunctionAttribute::NumRegisters)?;

    // CUDA's own occupancy calculator: the smallest block size that achieves
    // maximum occupancy, plus the minimum grid size needed to saturate the GPU.
    let (suggested_grid, threads_per_block) =
        func.suggested_launch_configuration(0, BlockSize::x(THREADS_PER_BLOCK_LIMIT))?;

    let active_blocks_per_sm =
        func.max_active_blocks_per_multiprocessor(BlockSize::x(threads_per_block), 0)?;
    let occupancy_pct =
        (active_blocks_per_sm * threads_per_block) as f64 / max_threads_per_sm as f64 * 100.0;

    // Grid-stride kernel: never launch fewer blocks than needed to keep
    // every SM fully occupied.
    let blocks = suggested_grid.max(sm_count * active_blocks_per_sm);

    println!(
        "[*] GPU occupancy: {} SMs | {} regs/thread | {} active block(s)/SM ({:.0}% theoretical occupancy)",
        sm_count, regs_per_thread, active_blocks_per_sm, occupancy_pct
    );

    // ------------------------------------------------------------------- //
    // 4.  Copy constant data to device globals
    // ------------------------------------------------------------------- //
    // Charset – pad to 96 bytes
    let mut charset_padded = [0u8; 96];
    charset_padded[..charset_len].copy_from_slice(charset_bytes);
    let mut d_charset = module.get_global::<[u8; 96]>(
        CStr::from_bytes_with_nul(b"d_charset\0").unwrap()
    )?;
    d_charset.copy_from(&charset_padded)?;

    // Target hash
    let mut d_target = module.get_global::<[u32; 5]>(
        CStr::from_bytes_with_nul(b"d_target\0").unwrap()
    )?;
    d_target.copy_from(&target_u32)?;

    // ------------------------------------------------------------------- //
    // 5.  Device buffers
    // ------------------------------------------------------------------- //
    let mut found_flag = DeviceBox::new(&0i32)?;
    let mut found_idx  = DeviceBox::new(&0u64)?;

    // ------------------------------------------------------------------- //
    // 6.  Search loop (iterates over each length from MIN to MAX)
    // ------------------------------------------------------------------- //
    let start_time = Instant::now();

    let stream = Stream::new(StreamFlags::NON_BLOCKING, None)?;

    println!("[*] GPU SHA-1 brute-force starting…");
    println!("    charset={:?} (size={}) | pwd_len range=[{}..={}]",
        args.charset, charset_len, args.length_min, args.length_max);
    println!("    grid={} blocks x {} threads | single launch per length",
        blocks, threads_per_block);

    // Track measured speed so heartbeat estimate improves after each length
    let mut last_speed_bps: f64 = 1_830_000_000.0;

    for pwd_len in args.length_min..=args.length_max {
        let total_combinations = (charset_len as u64).pow(pwd_len as u32);

        found_flag.copy_from(&0i32)?;
        found_idx.copy_from(&0u64)?;

        println!("\n[*] Trying length {} | search space={:e}",
            pwd_len, total_combinations as f64);

        // Heartbeat thread: polls every 200 ms, prints every PROGRESS_INTERVAL_SECS
        let running    = Arc::new(AtomicBool::new(true));
        let run_clone  = Arc::clone(&running);
        let length_start = Instant::now();
        let total_f64    = total_combinations as f64;
        let speed_hint   = last_speed_bps;

        let timer = std::thread::spawn(move || {
            let poll    = std::time::Duration::from_millis(200);
            let every   = std::time::Duration::from_secs(PROGRESS_INTERVAL_SECS);
            let mut next = Instant::now() + every;
            while run_clone.load(Ordering::Relaxed) {
                std::thread::sleep(poll);
                if !run_clone.load(Ordering::Relaxed) { break; }
                if Instant::now() >= next {
                    let elapsed = length_start.elapsed().as_secs_f64();
                    let pct = (elapsed * speed_hint / total_f64 * 100.0).min(99.9);
                    println!("    ... ~{:.1}% done | elapsed: {:.0}s", pct, elapsed);
                    next += every;
                }
            }
        });

        // Single kernel launch covering ALL candidates for this length
        unsafe {
            launch!(func<<<blocks, threads_per_block, 0, stream>>>(
                0u64,
                total_combinations,
                pwd_len as i32,
                charset_len as i32,
                found_flag.as_device_ptr(),
                found_idx.as_device_ptr()
            ))?;
        }

        stream.synchronize()?;

        // Stop heartbeat (returns within 200 ms)
        running.store(false, Ordering::Relaxed);
        let _ = timer.join();

        let elapsed = length_start.elapsed();
        let speed = total_combinations as f64 / elapsed.as_secs_f64();
        last_speed_bps = speed;

        let mut host_flag = 0i32;
        found_flag.copy_to(&mut host_flag)?;
        if host_flag == 1 {
            let mut win_idx = 0u64;
            found_idx.copy_to(&mut win_idx)?;
            let password = decode_idx(win_idx, pwd_len, charset_bytes);
            println!(
                "\n[FOUND] Password: '{}' (length {}) | Time: {:?}",
                password, pwd_len, start_time.elapsed()
            );
            return Ok(());
        }

        println!(
            "[!] Length {} done | {:.1}s | {:.2}B pwd/s | Password not found.",
            pwd_len, elapsed.as_secs_f64(), speed / 1e9
        );
    }

    println!("[!] Search complete. Password not found in range [{}..={}].", args.length_min, args.length_max);
    Ok(())
}