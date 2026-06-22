#include <stdint.h>

extern "C" {

#define ROTL32(x, n) (((x) << (n)) | ((x) >> (32 - (n))))

// Target hash (5 x uint32, big-endian words) – written by host via module symbol
__constant__ uint32_t d_target[5];

// Character set – written by host via module symbol (up to 96 printable ASCII chars)
__constant__ uint8_t d_charset[96];

__global__ void sha1_kernel(
    uint64_t  start_idx,
    int       pwd_len,
    int       charset_len,
    int       batch_size,
    int*      found_flag,
    uint64_t* found_idx)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= batch_size) return;

    // Early-exit if another thread already found it
    if (*found_flag) return;

    // ------------------------------------------------------------------ //
    // 1.  Decode candidate index → character array (most-significant first)
    // ------------------------------------------------------------------ //
    uint64_t n = start_idx + (uint64_t)tid;

    uint8_t chars[16];
    for (int j = pwd_len - 1; j >= 0; j--) {
        chars[j] = d_charset[n % charset_len];
        n /= charset_len;
    }

    // ------------------------------------------------------------------ //
    // 2.  Encode as UTF-16LE and lay out the SHA-1 message block
    // ------------------------------------------------------------------ //
    uint8_t msg[64] = {0};
    int byte_len = pwd_len * 2;
    for (int j = 0; j < pwd_len; j++) {
        msg[j * 2]     = chars[j];
        msg[j * 2 + 1] = 0x00;
    }

    // SHA-1 padding
    msg[byte_len] = 0x80;
    uint64_t bit_len = (uint64_t)byte_len * 8;
    msg[56] = (uint8_t)(bit_len >> 56);
    msg[57] = (uint8_t)(bit_len >> 48);
    msg[58] = (uint8_t)(bit_len >> 40);
    msg[59] = (uint8_t)(bit_len >> 32);
    msg[60] = (uint8_t)(bit_len >> 24);
    msg[61] = (uint8_t)(bit_len >> 16);
    msg[62] = (uint8_t)(bit_len >>  8);
    msg[63] = (uint8_t)(bit_len      );

    // Convert byte array → 16 big-endian uint32 words
    uint32_t w[16];
    #pragma unroll
    for (int i = 0; i < 16; i++) {
        w[i] = ((uint32_t)msg[i*4  ] << 24)
             | ((uint32_t)msg[i*4+1] << 16)
             | ((uint32_t)msg[i*4+2] <<  8)
             | ((uint32_t)msg[i*4+3]      );
    }

    // ------------------------------------------------------------------ //
    // 3.  SHA-1 compression
    // ------------------------------------------------------------------ //
    uint32_t a = 0x67452301u;
    uint32_t b = 0xEFCDAB89u;
    uint32_t c = 0x98BADCFEu;
    uint32_t d = 0x10325476u;
    uint32_t e = 0xC3D2E1F0u;

    #pragma unroll
    for (int i = 0; i < 80; i++) {
        if (i >= 16) {
            w[i & 15] = ROTL32(
                w[(i-3)  & 15] ^
                w[(i-8)  & 15] ^
                w[(i-14) & 15] ^
                w[ i     & 15], 1);
        }

        uint32_t f, k;
        if      (i < 20) { f = (b & c) | (~b & d);           k = 0x5A827999u; }
        else if (i < 40) { f =  b ^ c ^ d;                   k = 0x6ED9EBA1u; }
        else if (i < 60) { f = (b & c) | (b & d) | (c & d); k = 0x8F1BBCDCu; }
        else             { f =  b ^ c ^ d;                   k = 0xCA62C1D6u; }

        uint32_t temp = ROTL32(a, 5) + f + e + k + w[i & 15];
        e = d; d = c; c = ROTL32(b, 30); b = a; a = temp;
    }

    a += 0x67452301u;
    b += 0xEFCDAB89u;
    c += 0x98BADCFEu;
    d += 0x10325476u;
    e += 0xC3D2E1F0u;

    // ------------------------------------------------------------------ //
    // 4.  Compare all 5 words against the target
    // ------------------------------------------------------------------ //
    if (a == d_target[0] && b == d_target[1] &&
        c == d_target[2] && d == d_target[3] && e == d_target[4])
    {
        if (atomicCAS(found_flag, 0, 1) == 0) {
            *found_idx = start_idx + (uint64_t)tid;
        }
    }
}

} // extern "C"
