#include <metal_stdlib>
using namespace metal;

struct Msg { char text[64]; };

kernel void hello_from_gpu(device Msg* out [[buffer(0)]],
                           uint tid [[thread_position_in_grid]])
{
    const char prefix[] = "Hello from GPU! block=0 thread=";

    uint i = 0;
    for (; i < sizeof(prefix) - 1; ++i) out[tid].text[i] = prefix[i];

    out[tid].text[i++] = char('0' + (tid % 10));
    out[tid].text[i++] = '\n';
    out[tid].text[i++] = '\0';
}
