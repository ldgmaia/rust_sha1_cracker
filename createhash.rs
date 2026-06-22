use sha1_smol::Sha1;

fn main() {
    let s = "abcdef";
    let utf16: Vec<u8> = s.encode_utf16().flat_map(|u| u.to_le_bytes()).collect();
    let hash_bytes = Sha1::from(&utf16).digest().bytes();
    for b in hash_bytes {
        print!("{:02x}", b);
    }
    println!();
}