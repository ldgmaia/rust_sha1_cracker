use cust::prelude::*;
use cust::memory::DeviceBox;
use std::ffi::CStr;
use std::error::Error;
use std::time::Instant;

// -----------------------------------------------------------------------
// Configuration – change CHARSET and PWD_LEN to match your target
// -----------------------------------------------------------------------
const TARGET_HASH: &str = "B4CFC8DC918B7CBF9F7653B1DDB0540D7748C086";

// All printable ASCII: lowercase, uppercase, digits, symbols (95 chars)
const CHARSET: &[u8] = b"abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~ ";
// const CHARSET: &[u8] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz";

const PWD_LEN_MIN: usize = 5;
const PWD_LEN_MAX: usize = 5;

// Tuning – saturate the GB10 (2048 CUDA cores)
const THREADS_PER_BLOCK: u32 = 512;
const BLOCKS: u32 = 8192;

// -----------------------------------------------------------------------
// Decode a base-N index back to a password string
// -----------------------------------------------------------------------
fn decode_idx(mut n: u64, len: usize) -> String {
    let cs = CHARSET.len() as u64;
    let mut chars = Vec::with_capacity(len);
    for _ in 0..len {
        chars.push(CHARSET[(n % cs) as usize] as char);
        n /= cs;
    }
    chars.reverse();
    chars.into_iter().collect()
}

fn main() -> Result<(), Box<dyn Error>> {
    let charset_len = CHARSET.len();
    assert!(charset_len <= 96, "CHARSET too large (max 96)");

    // ------------------------------------------------------------------- //
    // 1.  Parse target hash → 5 big-endian uint32 words
    // ------------------------------------------------------------------- //
    let target_raw = hex::decode(TARGET_HASH)?;
    assert_eq!(target_raw.len(), 20, "SHA-1 hash must be 20 bytes");
    let mut target_u32 = [0u32; 5];
    for i in 0..5 {
        target_u32[i] = u32::from_be_bytes(
            target_raw[i*4..i*4+4].try_into().unwrap()
        );
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
    // 3.  Copy constant data to device globals
    // ------------------------------------------------------------------- //
    // Charset – pad to 96 bytes
    let mut charset_padded = [0u8; 96];
    charset_padded[..charset_len].copy_from_slice(CHARSET);
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
    // 4.  Device buffers
    // ------------------------------------------------------------------- //
    let mut found_flag = DeviceBox::new(&0i32)?;
    let mut found_idx  = DeviceBox::new(&0u64)?;

    // ------------------------------------------------------------------- //
    // 5.  Search loop (iterates over each length from MIN to MAX)
    // ------------------------------------------------------------------- //
    let batch_size = (THREADS_PER_BLOCK as u64) * (BLOCKS as u64);
    let start_time = Instant::now();

    let stream = Stream::new(StreamFlags::NON_BLOCKING, None)?;

    println!("[*] GPU SHA-1 brute-force starting…");
    println!("    charset size={} | pwd_len range=[{}..{}]",
        charset_len, PWD_LEN_MIN, PWD_LEN_MAX);
    println!("    batch={} ({} blocks × {} threads)",
        batch_size, BLOCKS, THREADS_PER_BLOCK);

    for pwd_len in PWD_LEN_MIN..=PWD_LEN_MAX {
        let total_combinations = (charset_len as u64).pow(pwd_len as u32);
        let mut current_start = 0u64;

        // Reset device flags for this length
        found_flag.copy_from(&0i32)?;
        found_idx.copy_from(&0u64)?;

        println!("\n[*] Trying length {} | search space={}", pwd_len, total_combinations);

        while current_start < total_combinations {
            let count = batch_size.min(total_combinations - current_start);

            unsafe {
                launch!(func<<<BLOCKS, THREADS_PER_BLOCK, 0, stream>>>(
                    current_start,
                    pwd_len as i32,
                    charset_len as i32,
                    count as i32,
                    found_flag.as_device_ptr(),
                    found_idx.as_device_ptr()
                ))?;
            }

            stream.synchronize()?;

            let mut host_flag = 0i32;
            found_flag.copy_to(&mut host_flag)?;
            if host_flag == 1 {
                let mut win_idx = 0u64;
                found_idx.copy_to(&mut win_idx)?;
                let password = decode_idx(win_idx, pwd_len);
                println!(
                    "\n[FOUND] Password: '{}' (length {}) | Time: {:?}",
                    password, pwd_len, start_time.elapsed()
                );
                return Ok(());
            }

            current_start += count;

            let elapsed = start_time.elapsed().as_secs_f64().max(1e-9);
            let speed   = current_start as f64 / elapsed;
            let pct     = current_start as f64 / total_combinations as f64 * 100.0;
            print!(
                "\r  len={} [{:6.2}%] {:6.2}B pwd/s  elapsed: {:.1}s   ",
                pwd_len, pct, speed / 1e9, elapsed
            );
            use std::io::Write;
            let _ = std::io::stdout().flush();
        }

        println!("\n[!] Length {} exhausted. Password not found at this length.", pwd_len);
    }

    println!("[!] Search complete. Password not found in range [{}..(=){}].", PWD_LEN_MIN, PWD_LEN_MAX);
    Ok(())
}