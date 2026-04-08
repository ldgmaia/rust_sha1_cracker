use sha1::{Digest, Sha1};
use std::{
    fs::write,
    str,
    sync::{
        atomic::{AtomicBool, AtomicU64, Ordering},
        Arc,
    },
    thread,
    time::{Duration, Instant},
};

const TARGET_HASH: &str = "3d3ce61821b97b65f249d219a062cae11395bd11";
const MAX_LENGTH: usize = 6;
const CHARSET: &[u8] =
    b"0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~ ";

fn to_utf16le_bytes(s: &str) -> Vec<u8> {
    s.encode_utf16()
        .flat_map(|u| u.to_le_bytes())
        .collect()
}

fn main() {
    let target_hash = hex::decode(TARGET_HASH).expect("Invalid hex in TARGET_HASH");
    let found = Arc::new(AtomicBool::new(false));
    let tested = Arc::new(AtomicU64::new(0));

    let num_threads = num_cpus::get();
    println!("[*] Using {} threads", num_threads);

    let start_time = Instant::now();

    let threads: Vec<_> = (0..num_threads)
        .map(|thread_id| {
            let found = Arc::clone(&found);
            let tested = Arc::clone(&tested);
            let target_hash = target_hash.clone();

            thread::spawn(move || {
                let mut pwd = vec![0u8; MAX_LENGTH];
                let mut stack = vec![];

                for i in (thread_id..CHARSET.len()).step_by(num_threads) {
                    stack.push((0, i));
                }

                while let Some((pos, char_idx)) = stack.pop() {
                    if found.load(Ordering::Relaxed) {
                        break;
                    }

                    pwd[pos] = CHARSET[char_idx];

                    if pos + 1 == MAX_LENGTH {
                        let s = String::from_utf8_lossy(&pwd);
                        let utf16le_bytes = to_utf16le_bytes(&s);
                        let mut hasher = Sha1::new();
                        hasher.update(&utf16le_bytes);
                        let hash = hasher.finalize();

                        tested.fetch_add(1, Ordering::Relaxed);

                        if hash.as_slice() == target_hash {
                            found.store(true, Ordering::Relaxed);
                            let password = s.trim_end_matches(char::from(0)).to_string();
                            let _ = write("found_password.txt", &password);
                            println!(
                                "[FOUND] Password: '{}' in {:.1} seconds after {} attempts",
                                password,
                                start_time.elapsed().as_secs_f64(),
                                tested.load(Ordering::Relaxed)
                            );
                            return;
                        }
                    } else {
                        for i in (0..CHARSET.len()).rev() {
                            stack.push((pos + 1, i));
                        }
                    }
                }
            })
        })
        .collect();

    // Progress reporting thread
    {
        let tested = Arc::clone(&tested);
        let found = Arc::clone(&found);
        let start_time = start_time.clone();

        thread::spawn(move || {
            let mut last_checked = 0u64;
            let mut last_reported = 0u64;
            let mut last_time = Instant::now();

            loop {
                thread::sleep(Duration::from_millis(200));

                if found.load(Ordering::Relaxed) {
                    break;
                }

                let checked = tested.load(Ordering::Relaxed);
                let delta = checked - last_checked;
                let elapsed = last_time.elapsed().as_secs_f64();

                if checked - last_reported >= 100_000_000 {
                    let total_elapsed = start_time.elapsed().as_secs_f64();
                    let speed = (delta as f64) / elapsed;
                    println!(
                        "[INFO] Checked {:>11} passwords in {:>5.1}s ({:>10.0} passwords/sec)",
                        checked,
                        total_elapsed,
                        speed
                    );
                    last_checked = checked;
                    last_reported = checked;
                    last_time = Instant::now();
                }
            }
        });
    }

    for t in threads {
        let _ = t.join();
    }

    if !found.load(Ordering::Relaxed) {
        println!("Password not found.");
    }
}
