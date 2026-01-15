#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <stdio.h>
#include <string.h>

typedef struct { char text[64]; } Msg;

int main(void) {
    printf("Hello from CPU!\n");

    @autoreleasepool {
        id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
        if (!dev) { printf("Metal not supported.\n"); return 1; }

        NSError *err = nil;
        NSString *src = [NSString stringWithContentsOfFile:@"HelloWorld.metal"
                                                  encoding:NSUTF8StringEncoding
                                                     error:&err];
        if (!src) { NSLog(@"Read .metal failed: %@", err); return 1; }

        id<MTLLibrary> lib = [dev newLibraryWithSource:src options:nil error:&err];
        if (!lib) { NSLog(@"Metal compile failed: %@", err); return 1; }

        id<MTLFunction> fn = [lib newFunctionWithName:@"hello_from_gpu"];
        id<MTLComputePipelineState> pso = [dev newComputePipelineStateWithFunction:fn error:&err];
        if (!pso) { NSLog(@"Pipeline failed: %@", err); return 1; }

        const int threads = 4; // like CUDA <<<1,4>>>
        id<MTLBuffer> out = [dev newBufferWithLength:sizeof(Msg) * threads
                                            options:MTLResourceStorageModeShared];
        memset(out.contents, 0, sizeof(Msg) * threads);

        id<MTLCommandQueue> q = [dev newCommandQueue];
        id<MTLCommandBuffer> cb = [q commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];

        [enc setComputePipelineState:pso];
        [enc setBuffer:out offset:0 atIndex:0];
        [enc dispatchThreads:MTLSizeMake(threads, 1, 1)
      threadsPerThreadgroup:MTLSizeMake(threads, 1, 1)];
        [enc endEncoding];

        [cb commit];
        [cb waitUntilCompleted];

        Msg* msgs = (Msg*)out.contents;
        for (int t = 0; t < threads; ++t) printf("%s", msgs[t].text);
    }
    return 0;
}
