use cust::prelude::*;
use cust::memory::DeviceBox;
use std::ffi::CStr;
use std::error::Error;
use std::time::Instant;

// -----------------------------------------------------------------------
// Configuration
// -----------------------------------------------------------------------
const TARGET_HASH: &str = "2eec0a11782dd531aa9d0fcac4bbdef1af711c84";
const CHARSET: &[u8] = b"abcdefghijklmnopqrstuvwxyz";
const PWD_LEN: usize = 6;

// Tuning knobs – saturate the GB10 (2048 CUDA cores)
// 512 threads/block × 8192 blocks = 4 194 304 candidates per launch
const THREADS_PER_BLOCK: u32 = 512;
const BLOCKS: u32 = 8192;

// -----------------------------------------------------------------------
// Decode a base-26 index back to a password string
// -----------------------------------------------------------------------
fn decode_idx(mut n: u64, len: usize) -> String {
    let mut chars = Vec::with_capacity(len);
    for _ in 0..len {
        chars.push(CHARSET[(n % 26) as usize] as char);
        n /= 26;
    }
    chars.reverse();
    chars.into_iter().collect()
}

fn main() -> Result<(), Box<dyn Error>> {
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

    // Load pre-compiled PTX
    let module = Module::from_file("sha1_kernel.ptx")?;
    let func = module.get_function("sha1_kernel")?;

    // ------------------------------------------------------------------- //
    // 3.  Copy constant data to device globals
    // ------------------------------------------------------------------- //
    // Charset
    let mut d_charset = module.get_global::<[u8; 26]>(
        CStr::from_bytes_with_nul(b"d_charset\0").unwrap()
    )?;
    d_charset.copy_from(&CHARSET.try_into().unwrap())?;

    // Target hash
    let mut d_target = module.get_global::<[u32; 5]>(
        CStr::from_bytes_with_nul(b"d_target\0").unwrap()
    )?;
    d_target.copy_from(&target_u32)?;

    // ------------------------------------------------------------------- //
    // 4.  Device buffers for the "found" flag and winning index
    // ------------------------------------------------------------------- //
    let found_flag = DeviceBox::new(&0i32)?;
    let found_idx  = DeviceBox::new(&0u64)?;

    // ------------------------------------------------------------------- //
    // 5.  Search loop
    // ------------------------------------------------------------------- //
    let batch_size = (THREADS_PER_BLOCK as u64) * (BLOCKS as u64);
    let total_combinations = (CHARSET.len() as u64).pow(PWD_LEN as u32);
    let mut current_start = 0u64;
    let start_time = Instant::now();

    let stream = Stream::new(StreamFlags::NON_BLOCKING, None)?;

    println!("[*] GPU SHA-1 brute-force starting…");
    println!("    charset={} | len={} | space={}",
        std::str::from_utf8(CHARSET).unwrap(), PWD_LEN, total_combinations);
    println!("    batch={} ({} blocks × {} threads)",
        batch_size, BLOCKS, THREADS_PER_BLOCK);

    while current_start < total_combinations {
        let count = batch_size.min(total_combinations - current_start);

        unsafe {
            launch!(func<<<BLOCKS, THREADS_PER_BLOCK, 0, stream>>>(
                current_start,
                PWD_LEN as i32,
                count as i32,
                found_flag.as_device_ptr(),
                found_idx.as_device_ptr()
            ))?;
        }

        // Wait for the GPU to finish this batch before reading results
        stream.synchronize()?;

        // Check the flag
        let mut host_flag = 0i32;
        found_flag.copy_to(&mut host_flag)?;
        if host_flag == 1 {
            let mut win_idx = 0u64;
            found_idx.copy_to(&mut win_idx)?;
            let password = decode_idx(win_idx, PWD_LEN);
            println!(
                "\n[FOUND] Password: '{}' | Time: {:?}",
                password, start_time.elapsed()
            );
            return Ok(());
        }

        current_start += count;

        // Progress report
        let elapsed = start_time.elapsed().as_secs_f64().max(1e-9);
        let speed   = current_start as f64 / elapsed;
        let pct     = current_start as f64 / total_combinations as f64 * 100.0;
        print!(
            "\r[{:6.2}%] {:6.2}B pwd/s  elapsed: {:.1}s   ",
            pct, speed / 1e9, elapsed
        );
        use std::io::Write;
        let _ = std::io::stdout().flush();
    }

    println!("\n[!] Search complete. Password not found.");
    Ok(())
}