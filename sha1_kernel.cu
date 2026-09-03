#include <stdint.h>

extern "C" {

#define ROTL32(x, n) (((x) << (n)) | ((x) >> (32 - (n))))

// Target hash (5 x uint32, big-endian words) – written by host via module symbol
__constant__ uint32_t d_target[5];

// Character set – written by host via module symbol (up to 96 printable ASCII chars)
__constant__ uint8_t d_charset[96];

// max 256 threads/block; hint to compiler to minimise register spill
__launch_bounds__(256)
__global__ void sha1_kernel(
    uint64_t  start_idx,
    uint64_t  count,        // number of candidates in this batch (uint64 for large batches)
    int       pwd_len,
    int       charset_len,
    int*      found_flag,
    uint64_t* found_idx)
{
    // Load charset into shared memory once per block
    __shared__ uint8_t s_charset[96];
    if (threadIdx.x < 96) {
        s_charset[threadIdx.x] = d_charset[threadIdx.x];
    }
    __syncthreads();

    const uint64_t gid    = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    const uint64_t stride = (uint64_t)gridDim.x  * blockDim.x;
    const uint64_t end    = start_idx + count;

    // Grid-stride loop: each thread processes multiple candidates per launch
    for (uint64_t idx = start_idx + gid; idx < end; idx += stride) {

        // Read found_flag through L2 cache (avoids L1 thrash from frequent polling)
        if (__ldcg(found_flag)) return;

        // ---------------------------------------------------------------- //
        // 1. Decode index → character array (most-significant first)
        //
        //    64-bit div/mod costs several times more than 32-bit on GPU ALUs.
        //    idx needs the full 64 bits only for the first few digits; once
        //    the remaining value fits in 32 bits, switch to native 32-bit
        //    div/mod for the rest (e.g. the whole loop is 32-bit when
        //    charset_len^pwd_len fits in 32 bits, as with digit-only charsets).
        // ---------------------------------------------------------------- //
        uint64_t n64 = idx;
        uint8_t  chars[16];
        int j = pwd_len - 1;
        for (; j >= 0 && n64 > 0xFFFFFFFFULL; j--) {
            chars[j] = s_charset[(uint32_t)(n64 % (uint32_t)charset_len)];
            n64 /= (uint32_t)charset_len;
        }
        uint32_t n32 = (uint32_t)n64;
        for (; j >= 0; j--) {
            chars[j] = s_charset[n32 % (uint32_t)charset_len];
            n32      /= (uint32_t)charset_len;
        }

        // ---------------------------------------------------------------- //
        // 2. Build W[] directly from UTF-16LE layout — no msg[] intermediary
        //
        //    UTF-16LE of ASCII char c → bytes [c, 0x00]
        //    SHA-1 W[i] is 4 bytes read big-endian, so:
        //      W[i] = chars[2i]<<24 | chars[2i+1]<<8   (two chars per word)
        //
        //    Padding byte 0x80 follows the last data byte, then zeros,
        //    then the 64-bit big-endian bit-length in W[14..15].
        // ---------------------------------------------------------------- //
        uint32_t w[16];

        for (int i = 0; i < pwd_len / 2; i++) {
            w[i] = ((uint32_t)chars[i * 2] << 24) | ((uint32_t)chars[i * 2 + 1] << 8);
        }
        if (pwd_len & 1) {
            // Odd length: last char + 0x00 + 0x80 + 0x00 fit in one word
            w[pwd_len / 2] = ((uint32_t)chars[pwd_len - 1] << 24) | 0x00008000u;
        } else {
            // Even length: 0x80 is the sole byte of the next word (high byte)
            w[pwd_len / 2] = 0x80000000u;
        }
        for (int i = pwd_len / 2 + 1; i < 15; i++) w[i] = 0u;
        w[14] = 0u;                                  // bit_len < 2^32 for pwd_len <= 8
        w[15] = (uint32_t)(pwd_len << 4);            // pwd_len * 2 bytes * 8 bits

        // ---------------------------------------------------------------- //
        // 3. SHA-1 compression — 4 separate unrolled rounds, zero branches
        // ---------------------------------------------------------------- //
        uint32_t a = 0x67452301u;
        uint32_t b = 0xEFCDAB89u;
        uint32_t c = 0x98BADCFEu;
        uint32_t d = 0x10325476u;
        uint32_t e = 0xC3D2E1F0u;

#define SHA1_STEP(f_, k_) { \
    uint32_t _t = ROTL32(a,5) + (f_) + e + (k_) + w[i & 15]; \
    e = d; d = c; c = ROTL32(b,30); b = a; a = _t; \
}

        // Round 1a: i = 0..15  (W already initialised, no expansion)
        #pragma unroll
        for (int i = 0; i < 16; i++) {
            SHA1_STEP((b & c) | (~b & d), 0x5A827999u)
        }
        // Round 1b: i = 16..19  (same f/k, expansion starts)
        #pragma unroll
        for (int i = 16; i < 20; i++) {
            w[i&15] = ROTL32(w[(i-3)&15]^w[(i-8)&15]^w[(i-14)&15]^w[i&15], 1);
            SHA1_STEP((b & c) | (~b & d), 0x5A827999u)
        }
        // Round 2: i = 20..39
        #pragma unroll
        for (int i = 20; i < 40; i++) {
            w[i&15] = ROTL32(w[(i-3)&15]^w[(i-8)&15]^w[(i-14)&15]^w[i&15], 1);
            SHA1_STEP(b ^ c ^ d, 0x6ED9EBA1u)
        }
        // Round 3: i = 40..59
        #pragma unroll
        for (int i = 40; i < 60; i++) {
            w[i&15] = ROTL32(w[(i-3)&15]^w[(i-8)&15]^w[(i-14)&15]^w[i&15], 1);
            SHA1_STEP((b & c) | (b & d) | (c & d), 0x8F1BBCDCu)
        }
        // Round 4: i = 60..79
        #pragma unroll
        for (int i = 60; i < 80; i++) {
            w[i&15] = ROTL32(w[(i-3)&15]^w[(i-8)&15]^w[(i-14)&15]^w[i&15], 1);
            SHA1_STEP(b ^ c ^ d, 0xCA62C1D6u)
        }

#undef SHA1_STEP

        // ---------------------------------------------------------------- //
        // 4. Compare — early exit on first word mismatch (skips 4 additions)
        // ---------------------------------------------------------------- //
        a += 0x67452301u;
        if (a != d_target[0]) continue;

        b += 0xEFCDAB89u;
        c += 0x98BADCFEu;
        d += 0x10325476u;
        e += 0xC3D2E1F0u;

        if (b == d_target[1] && c == d_target[2] && d == d_target[3] && e == d_target[4]) {
            if (atomicCAS(found_flag, 0, 1) == 0) {
                *found_idx = idx;
            }
            return;
        }
    }
}

} // extern "C"
