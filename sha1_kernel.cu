// CUDA SHA1 kernel for batch password hashing (lowercase a-z, 6 chars)
// Each thread processes one input string of length input_len

__device__ void sha1_transform(const unsigned char* data, unsigned int* state);

extern "C" __global__ void sha1_kernel(const unsigned char* inputs, int input_len, unsigned char* hashes, int num_inputs) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_inputs) return;

    // Each input is input_len bytes
    const unsigned char* input = inputs + idx * input_len;
    unsigned int state[5] = {
        0x67452301,
        0xEFCDAB89,
        0x98BADCFE,
        0x10325476,
        0xC3D2E1F0
    };
    unsigned char block[64] = {0};
    int i;
    for (i = 0; i < input_len && i < 64; ++i) block[i] = input[i];
    block[input_len] = 0x80;
    unsigned long long bit_len = ((unsigned long long)input_len) * 8ULL;
    block[56] = (bit_len >> 56) & 0xff;
    block[57] = (bit_len >> 48) & 0xff;
    block[58] = (bit_len >> 40) & 0xff;
    block[59] = (bit_len >> 32) & 0xff;
    block[60] = (bit_len >> 24) & 0xff;
    block[61] = (bit_len >> 16) & 0xff;
    block[62] = (bit_len >> 8) & 0xff;
    block[63] = (bit_len) & 0xff;
    sha1_transform(block, state);
    for (i = 0; i < 5; ++i) {
        hashes[idx * 20 + i * 4 + 0] = (state[i] >> 24) & 0xff;
        hashes[idx * 20 + i * 4 + 1] = (state[i] >> 16) & 0xff;
        hashes[idx * 20 + i * 4 + 2] = (state[i] >> 8) & 0xff;
        hashes[idx * 20 + i * 4 + 3] = (state[i]) & 0xff;
    }
}

// Minimal SHA1 transform implementation for a single 64-byte block
__device__ void sha1_transform(const unsigned char* data, unsigned int* state) {
    unsigned int a, b, c, d, e, f, k, temp;
    unsigned int w[80];
    int i;
    for (i = 0; i < 16; ++i) {
        w[i] = (data[i * 4 + 0] << 24) |
               (data[i * 4 + 1] << 16) |
               (data[i * 4 + 2] << 8) |
               (data[i * 4 + 3]);
    }
    for (i = 16; i < 80; ++i) {
        w[i] = (w[i-3] ^ w[i-8] ^ w[i-14] ^ w[i-16]);
        w[i] = (w[i] << 1) | (w[i] >> 31);
    }
    a = state[0]; b = state[1]; c = state[2]; d = state[3]; e = state[4];
    for (i = 0; i < 80; ++i) {
        if (i < 20)      { f = (b & c) | ((~b) & d); k = 0x5A827999; }
        else if (i < 40) { f = b ^ c ^ d;            k = 0x6ED9EBA1; }
        else if (i < 60) { f = (b & c) | (b & d) | (c & d); k = 0x8F1BBCDC; }
        else             { f = b ^ c ^ d;            k = 0xCA62C1D6; }
        temp = ((a << 5) | (a >> 27)) + f + e + k + w[i];
        e = d;
        d = c;
        c = (b << 30) | (b >> 2);
        b = a;
        a = temp;
    }
    state[0] += a;
    state[1] += b;
    state[2] += c;
    state[3] += d;
    state[4] += e;
}
