#include <iostream>
#include <cuda_runtime.h>

#include <thrust/device_ptr.h>
#include <thrust/reduce.h>
#include <thrust/execution_policy.h>

#include <npp.h>
#include <nppcore.h> 
#include <nppi.h>
#include <nppdefs.h>

#include <filesystem>
#include <string>

#include <opencv2/opencv.hpp>
#include <opencv2/core/cuda.hpp>

void checkCudaError(cudaError_t err, const char *msg) {
    if (err != cudaSuccess) {
        std::cerr << "CUDA Error: " << msg << " - " << cudaGetErrorString(err) << std::endl;
        exit(EXIT_FAILURE);
    }
}

__global__ void threshold_kernel(
    unsigned char* d_input, 
    unsigned char* d_output, 
    int width, 
    int height, 
    unsigned char threshold, 
    int bw_mode
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x < width && y < height) {
        int idx = y * width + x;
        if (bw_mode == 0) { // target black on white
            d_output[idx] = (d_input[idx] > threshold) ? 0 : 255;
        }
        else { // target white on black
            d_output[idx] = (d_input[idx] > threshold) ? 255 : 0;
        }
    }
}

__device__ inline int isPixelBlack(
    const unsigned char* img , 
    int x, 
    int y, 
    int w, 
    int h
) {
    if (x < 0 || x >= w || y<0 || y>=h) return 0;
    return img[y*w+x] > 0 ? 1 : 0;
}

__global__
void ZhangSuenMark(
    const unsigned char* src, 
    unsigned char* mark, 
    int w, 
    int h, 
    int pass
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h) return;

    int p1 = isPixelBlack(src, x, y, w, h);
    // ! the pixel is black (cond0)
    if (!p1) {
        mark[y*w + x] = 0; 
        return;
    }

    int p2 = isPixelBlack(src, x-1, y-1, w, h); // top left
    int p3 = isPixelBlack(src, x-1, y, w, h);   // left
    int p4 = isPixelBlack(src, x-1, y+1, w, h); // bottom left
    int p5 = isPixelBlack(src, x, y+1, w, h);   // bottom
    int p6 = isPixelBlack(src, x+1, y+1, w, h); // bottom right
    int p7 = isPixelBlack(src, x+1, y, w, h);   // right 
    int p8 = isPixelBlack(src, x+1, y-1, w, h); // top right
    int p9 = isPixelBlack(src, x, y-1, w, h);   // top


    // A(i,j) = number of transitions from white to black 
    //          in sequence of 8 neighbors making complete circle
    int A = 0;
    int sequence[8] = {p2, p3, p4, p5, p6, p7, p8, p9};
    for (int i=0; i<8; i++) {
        if (sequence[i] == 0 && sequence[(i+1)%8]==1) {
            A++; // white --> black
        }
    }

    // B(i,j) = number of black pixels among 8 neighbors
    int B = p2+p3+p4+p5+p6+p7+p8+p9;

    // 1. 2 <= B(i,j)
    bool cond1 = (B>=2 && B<=6);
    // 2. A(i,j) = 1
    bool cond2 = (A==1); 
    
    // 3.
    bool cond3, cond4; 
    if (pass == 0) { 
        // 3. at least one of north, east, south neighbors is white
        cond3 = (!p9 || !p7 || !p5);
        // 4. at least one of east, south, west neightbors is white
        cond4 = (!p7 || !p5 || !p3);
    } else {
        // 3. at least one of north, east, west neigbors is white
        cond3 = (!p9 || !p7 || !p3);
        // 4. at least one of north, south, west neighbors is white
        cond4 = (!p9 || !p5 || !p3);
    }

    if (cond1 && cond2 && cond3 && cond4) {
        mark[y*w + x] = 1; // mark to remove
    }
    else {
        mark[y*w + x] = 0;
    }
}

__global__ void removeMarked(unsigned char* img, const unsigned char* mark, int w, int h) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h) return;
    if (mark[y*w + x]) img[y*w + x] = 0;
}

void setupContext(NppStreamContext& ctx) {
    int device = 0;
    cudaGetDevice(&device);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);
    ctx.nCudaDeviceId = device;
    ctx.nMultiProcessorCount = prop.multiProcessorCount;
    ctx.nMaxThreadsPerMultiProcessor = prop.maxThreadsPerMultiProcessor;
    ctx.nMaxThreadsPerBlock = prop.maxThreadsPerBlock;
    ctx.nSharedMemPerBlock = prop.sharedMemPerBlock;
    ctx.nCudaDevAttrComputeCapabilityMajor = prop.major;
    ctx.nCudaDevAttrComputeCapabilityMinor = prop.minor;
}

void processFrame(
    unsigned char* d_gray, 
    char gray_threshold,
    int bw_mode, 
    unsigned char* d_bw,
    unsigned char* d_mark,
    int w,
    int h, 
    cudaStream_t stream     // the cuda stream to use for this frame
) {
    size_t size = (size_t) w*h;
    
    int BLOCK = 16;
    dim3 threads(BLOCK,BLOCK);
    dim3 blocks((w+BLOCK-1)/BLOCK,(h+BLOCK-1)/BLOCK);

    // threshold pre-processing
    threshold_kernel<<<blocks,threads>>>(d_gray, d_bw, w, h, gray_threshold, bw_mode);
    cudaDeviceSynchronize();

    // skeletonization (Zhang Suen) 

    int max_itr = 1000; // in case no success
    bool changed;
    for (int itr=0; itr<max_itr; itr++) {
        changed = false;
        for (int pass=0; pass<2; pass++) { // pass 1, 2
            ZhangSuenMark<<<blocks,threads>>>(d_bw, d_mark, w, h, pass);
            cudaDeviceSynchronize();

            // Thrust: count marked pixels
            thrust::device_ptr<unsigned char> dev_mark(d_mark);
            int numMarked = thrust::reduce(thrust::device, 
                                            dev_mark, dev_mark + w*h, 0);
            if(numMarked > 0) changed = true;

            removeMarked<<<blocks,threads>>>(d_bw, d_mark, w, h);
            cudaDeviceSynchronize();
        }
        if (!changed) { // finish
            break;
        }
    }
    if (changed) {
        std::cout << "did not finish zhang suen" << std::endl;
    }
}

unsigned char getViableThreshold(const cv::Mat& gray_frame) {
    if (gray_frame.empty()) return 128; // Safety fallback
    cv::Scalar mean_val = cv::mean(gray_frame);
    return 0.9*static_cast<unsigned char>(mean_val.val[0]);
}


int main(int argc , char** argv) {
    std::string image_path;
    unsigned char threshold_value = 128; // Default value
    int bw_mode = 1; // white on black

    // --- Command Line Argument Parsing ---
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " -i <image_file_path> -t <threshold_value (0-255)> -m <b/w mode(0 = target black on white)>" << std::endl;
        std::cerr << "Example: " << argv[0] << " -i inputs/cat.png -t 220 -m 0" << std::endl;
        return 1;
    }

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "-i" && i + 1 < argc) {
            image_path = argv[++i];
        } else if (arg == "-t" && i + 1 < argc) {
            int t = std::atoi(argv[++i]);
            if (t >= 0 && t <= 255) {
                threshold_value = static_cast<unsigned char>(t);
            } else {
                std::cerr << "Error: Threshold value must be between 0 and 255. Using default (128)." << std::endl;
                // Continue using default or exit based on preference, here we exit
                return 1;
            }
        } else if (arg == "-m" && i + 1 < argc) {
            int m = std::atoi(argv[++i]);
            bw_mode = m > 0 ? 1 : 0;
        }
    }
    
    if (image_path.empty()) {
        std::cerr << "Error: Input image path (-i) is required." << std::endl;
        return 1;
    }
    
    cv::VideoCapture cap(image_path);

    
    int w = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
    int h = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));
    double fps = cap.get(cv::CAP_PROP_FPS);
    int fourcc = cv::VideoWriter::fourcc('M', 'J', 'P', 'G');
    int total_frames = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_COUNT));

    std::cout << "Input Video: " << image_path << " (" << w << "x" << h << ", " << fps << " FPS, " << total_frames << " frames)" << std::endl;
    std::cout << "Running skeletonization targeting " << (bw_mode ? "white object on black background" : "black object on white background") << " and bw threshold value " << std::to_string(threshold_value) << std::endl;

    // set up output dir + writer
    std::filesystem::path input_path(image_path);
    std::string base_name = input_path.stem().string();
    std::filesystem::path output_path = std::filesystem::path("outputs") / 
                                            (base_name + "_skeleton_t" + std::to_string(threshold_value) + "_" + (bw_mode ? "w" : "b") + ".avi");
    std::filesystem::create_directories("outputs");
    std::string output_filename = output_path.string();

    cv::VideoWriter writer(output_filename, fourcc, fps, cv::Size(w, h), false); // 'false' for single channel (grayscale) output
    if (!writer.isOpened()) {
        std::cerr << "Error: Could not open the output video writer." << std::endl;
        return -1;
    }


    // allocate all needed memory for each stream
    const int NUM_STREAMS = 5;

    size_t gray_size = (size_t)w * h * sizeof(unsigned char);
    cudaStream_t streams[NUM_STREAMS];

    unsigned char* d_gray[NUM_STREAMS];
    unsigned char* d_bw[NUM_STREAMS];
    unsigned char *d_mark[NUM_STREAMS];
    unsigned char *h_pinned_buffer[NUM_STREAMS];
    cv::Mat h_result_mats[NUM_STREAMS];

    for (int i=0; i<NUM_STREAMS; i++) {
        cudaStreamCreate(&streams[i]);
        // allocate device memory
        cudaMalloc(&d_gray[i], w*h);
        cudaMalloc(&d_bw[i], w*h);
        cudaMalloc(&d_mark[i], w*h);

        // allocate host (pinned)
        checkCudaError(cudaHostAlloc(&h_pinned_buffer[i], gray_size, cudaHostAllocPortable), 
                       "Host pinned allocation failed");

        // Initialize output Mat to use the pinned memory buffer
        h_result_mats[i] = cv::Mat(h, w, CV_8UC1, h_pinned_buffer[i]);
    }

    // kick off initial streams
    cv::Mat h_color_frame, h_gray_frame;
    int frames_processed = 0;

    for (int i=0; i<NUM_STREAMS; i++) {
        cap >> h_color_frame;
        if (h_color_frame.empty()) break;

        // convert RGB --> Grayscale
        cv::cvtColor(h_color_frame, h_gray_frame, cv::COLOR_BGR2GRAY);

        // copy data and launch initial kernels
        checkCudaError(cudaMemcpyAsync(d_gray[i], h_gray_frame.data, gray_size, 
                                        cudaMemcpyHostToDevice, streams[i]), 
                       "H2D copy failed");

        processFrame(d_gray[i], threshold_value, bw_mode, d_bw[i], d_mark[i], w, h, streams[i]);
    }

    int current_stream = 0;
    // process frames (in order)
    while (true) {
        // wait for current oldest frame so it can be written in order
        // other streams continue to process
        checkCudaError(cudaStreamSynchronize(streams[current_stream]), "Stream sync failed");

        checkCudaError(cudaMemcpyAsync(h_pinned_buffer[current_stream], d_bw[current_stream], 
                                        gray_size, cudaMemcpyDeviceToHost, streams[current_stream]), 
                       "D2H copy failed");

        checkCudaError(cudaStreamSynchronize(streams[current_stream]), "D2H sync failed");
        writer.write(h_result_mats[current_stream]);
        frames_processed++;
        std::cout << "finished " << frames_processed << "th frame" << std::endl;
        
        // kick off next frame
        cap >> h_color_frame;
        if (h_color_frame.empty()) break; // End of video

        cv::cvtColor(h_color_frame, h_gray_frame, cv::COLOR_BGR2GRAY);
        checkCudaError(cudaMemcpyAsync(d_gray[current_stream], h_gray_frame.data, gray_size, 
                                        cudaMemcpyHostToDevice, streams[current_stream]), 
                       "H2D copy failed");
        
        processFrame(d_gray[current_stream], threshold_value, bw_mode, d_bw[current_stream], d_mark[current_stream], w, h, streams[current_stream]);
        
        // to next stream
        current_stream = (current_stream + 1) % NUM_STREAMS;
    }

    // finish out streams
    for (int i = 0; i < NUM_STREAMS; ++i) {
        int stream_idx = (current_stream + i) % NUM_STREAMS;
        
        checkCudaError(cudaStreamSynchronize(streams[stream_idx]), "Flush sync failed");
        
        // Copy processed data from Device (d_bw) to Host Pinned Memory (h_pinned_buffer)
        checkCudaError(cudaMemcpyAsync(h_pinned_buffer[stream_idx], d_bw[stream_idx], 
                                        gray_size, cudaMemcpyDeviceToHost, streams[stream_idx]), 
                       "D2H copy failed during flush");

        checkCudaError(cudaStreamSynchronize(streams[stream_idx]), "D2H sync failed during flush");
        
        // Write the remaining frame
        writer.write(h_result_mats[stream_idx]);
        frames_processed++;
    }

    // clean up
    std::cout << "Finished processing " << frames_processed << " frames in video." << std::endl;

    writer.release();
    cap.release();

    for (int i=0; i<NUM_STREAMS; i++) {
        cudaFree(d_gray[i]);
        cudaFree(d_bw[i]);
        cudaFree(d_mark[i]);
        cudaFreeHost(h_pinned_buffer[i]);
        cudaStreamDestroy(streams[i]);
    }

    return 0;
}