// CUDA-accelerated brute-force SHA1 cracker
use cust::prelude::*;
use std::error::Error;
use hex;
use std::time::Instant;

const TARGET_HASH: &str = "3d3ce61821b97b65f249d219a062cae11395bd11";
const MAX_LENGTH: usize = 6;
const UTF16LE_BYTES_PER_CHAR: usize = 2;
const CHARSET: &[u8] = b"abcdefghijklmnopqrstuvwxyz";
const BATCH_SIZE: usize = 1_048_576; // Larger batch for maximum GPU throughput (reduce if OOM)

fn main() -> Result<(), Box<dyn Error>> {
    let start_time = Instant::now();
    cust::init(cust::CudaFlags::empty())?;
    let device = Device::get_device(0)?;
    let _ctx = Context::new(device)?;
    let module = Module::from_file("sha1_kernel.ptx")?;
    let func = module.get_function("sha1_kernel")?;
    let target_hash = hex::decode(TARGET_HASH).expect("Invalid hex in TARGET_HASH");

    let charset_len = CHARSET.len();
    let total: u128 = charset_len.pow(MAX_LENGTH as u32) as u128;
    let mut found = false;
    let mut found_pwd = vec![];


    let input_bytes_per_pwd = MAX_LENGTH * UTF16LE_BYTES_PER_CHAR;

    // Double buffering setup
    let mut batches = [vec![0u8; BATCH_SIZE * input_bytes_per_pwd], vec![0u8; BATCH_SIZE * input_bytes_per_pwd]];
    let mut hashes_bufs = [vec![0u8; BATCH_SIZE * 20], vec![0u8; BATCH_SIZE * 20]];
    let mut d_inputs = [DeviceBuffer::zeroed(BATCH_SIZE * input_bytes_per_pwd)?, DeviceBuffer::zeroed(BATCH_SIZE * input_bytes_per_pwd)?];
    let d_hashes = [DeviceBuffer::<u8>::zeroed(BATCH_SIZE * 20)?, DeviceBuffer::<u8>::zeroed(BATCH_SIZE * 20)?];
    let stream0 = Stream::new(StreamFlags::NON_BLOCKING, None)?;
    let stream1 = Stream::new(StreamFlags::NON_BLOCKING, None)?;
    // Use array of Stream, not references
    let mut streams = [stream0, stream1];

    let threads_per_block = 1024u32;
    let blocks = ((BATCH_SIZE as u32) + threads_per_block - 1) / threads_per_block;

    let mut batch_start = 0u128;
    let mut buf_idx = 0;
    let mut prev_count = 0;
    let mut prev_launched = false;

    while batch_start < total {
        // Prepare batch
        let batch = &mut batches[buf_idx];
        let mut count = 0;
        let mut chars = [0u8; MAX_LENGTH];
        let mut utf16 = [0u16; MAX_LENGTH];
        for i in 0..BATCH_SIZE {
            let idx = batch_start + i as u128;
            if idx >= total {
                break;
            }
            let mut n = idx;
            for j in 0..MAX_LENGTH {
                chars[j] = CHARSET[(n % charset_len as u128) as usize];
                n /= charset_len as u128;
            }
            // Encode as UTF-16LE without heap allocation
            let s = &chars;
            let mut utf16_len = 0;
            for (j, c) in String::from_utf8_lossy(s).chars().enumerate() {
                utf16[j] = c as u16;
                utf16_len += 1;
            }
            for j in 0..utf16_len {
                let bytes = utf16[j].to_le_bytes();
                batch[i * input_bytes_per_pwd + j * 2] = bytes[0];
                batch[i * input_bytes_per_pwd + j * 2 + 1] = bytes[1];
            }
            count += 1;
        }

        // Async copy to device
        d_inputs[buf_idx].copy_from(&batch[..])?;

        // Launch kernel
        let stream = &mut streams[buf_idx];
        unsafe {
            launch!(func<<<blocks, threads_per_block, 0, stream>>>(
                d_inputs[buf_idx].as_device_ptr(),
                input_bytes_per_pwd as i32,
                d_hashes[buf_idx].as_device_ptr(),
                BATCH_SIZE as i32
            ))?;
        }

        // If not the first batch, copy results from previous buffer while this one runs
        if prev_launched {
            streams[1 - buf_idx].synchronize()?;
            d_hashes[1 - buf_idx].copy_to(&mut hashes_bufs[1 - buf_idx])?;
            for i in 0..prev_count {
                if hashes_bufs[1 - buf_idx][i * 20..(i + 1) * 20] == target_hash[..] {
                    found = true;
                    let utf16: Vec<u16> = (0..MAX_LENGTH)
                        .map(|j| u16::from_le_bytes([
                            batches[1 - buf_idx][i * input_bytes_per_pwd + j * 2],
                            batches[1 - buf_idx][i * input_bytes_per_pwd + j * 2 + 1],
                        ]))
                        .collect();
                    found_pwd = String::from_utf16_lossy(&utf16).into_bytes();
                    break;
                }
            }
            if found {
                break;
            }
        }

        prev_count = count;
        prev_launched = true;
        batch_start += BATCH_SIZE as u128;
        buf_idx = 1 - buf_idx;
    }

    // Final synchronize and check last buffer
    if prev_launched && !found {
        streams[1 - buf_idx].synchronize()?;
        d_hashes[1 - buf_idx].copy_to(&mut hashes_bufs[1 - buf_idx])?;
        for i in 0..prev_count {
            if hashes_bufs[1 - buf_idx][i * 20..(i + 1) * 20] == target_hash[..] {
                found = true;
                let utf16: Vec<u16> = (0..MAX_LENGTH)
                    .map(|j| u16::from_le_bytes([
                        batches[1 - buf_idx][i * input_bytes_per_pwd + j * 2],
                        batches[1 - buf_idx][i * input_bytes_per_pwd + j * 2 + 1],
                    ]))
                    .collect();
                found_pwd = String::from_utf16_lossy(&utf16).into_bytes();
                break;
            }
        }
    }

    if found {
        let pwd_str = String::from_utf8_lossy(&found_pwd);
        println!("[FOUND] Password: {} in {:.2} seconds", pwd_str, start_time.elapsed().as_secs_f64());
    } else {
        println!("Password not found.");
    }
    Ok(())

    // PERFORMANCE NOTE:
    // If you run out of memory, reduce BATCH_SIZE (try 262144 or 131072).
    // For even more speed, run multiple processes (if you have multiple GPUs).
}
