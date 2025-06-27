#include "Kitokarosu.hpp"
#include <iomanip>
#include <chrono>
#include <atomic>
#include <thread>
#include <mutex>
#include <vector>
#include <atomic>

using Kito::QAM16;
using Kito::QAM64;
using Kito::QAM256;

static constexpr size_t TxAntNum = 32;
static constexpr size_t RxAntNum = 32;
static constexpr size_t K = 32;  // K-best size

using QAM = QAM64<float>;
using Kito::Detection;
using Kito::Mod;
using Kito::Rx;
using Kito::Tx;

struct ThreadResult {
    int err_frames = 0;
    int err_bits = 0;
    uint64_t total_size = 0;
    int processed = 0;
};
int main(int argc, char* argv[]) {
    int max_sample = 10000000;  // 最大样本数防止无限循环
    int err_frame_threshold = 1000;  // 错误帧阈值
    int snr_start = 24;  // Default start SNR
    int snr_end = 24;    // Default end SNR
    int snr_step = 1;    // Default SNR step size
    unsigned int seed = 114514;

    if (argc > 1) max_sample = atoi(argv[1]);
    if (argc > 2) err_frame_threshold = atoi(argv[2]);
    if (argc > 3) snr_start = atoi(argv[3]);
    if (argc > 4) snr_end = atoi(argv[4]);
    if (argc > 5) snr_step = atoi(argv[5]);
    if (argc > 6) seed = atoi(argv[6]);

    const bool show_progress = true;
    constexpr int update_interval = 10;
    
    // Vectors to store results for each SNR
    std::vector<int> snr_values;
    std::vector<float> ber_values;
    std::vector<float> avg_list_size_values;

    // Iterate over SNR range
    for (int snr = snr_start; snr <= snr_end; snr += snr_step) {
        
        // Atomic counters for tracking progress
        std::atomic<int> global_progress(0);
        std::atomic<int> global_err_frames(0);
        std::atomic<int> global_err_bits(0);
        std::atomic<uint64_t> global_total_size(0);
        std::atomic<bool> should_stop(false);

        const unsigned int num_threads = std::thread::hardware_concurrency();

        auto worker = [&](unsigned int thread_seed) {
            Kito::set_random_seed(thread_seed);
            auto det = Detection<Rx<RxAntNum>, Tx<TxAntNum>, Mod<QAM>>();
            det.setSNR(snr);  // Use current SNR value
            auto tree = Kito::KBest<decltype(det),K>();
            
            ThreadResult local;
            int local_count = 0;

            while (!should_stop.load(std::memory_order_relaxed) && 
                   global_progress.load(std::memory_order_relaxed) < max_sample) {
                det.generate();
                auto list = tree.run(det);
                auto err = det.judge(list);

                local.err_frames += (err > 0);
                local.err_bits += err;
                // local.total_size += size;
                local.processed++;
                local_count++;

                if (local_count % update_interval == 0) {
                    global_progress.fetch_add(local.processed, std::memory_order_relaxed);
                    int new_err_frames = global_err_frames.fetch_add(local.err_frames, std::memory_order_relaxed) + local.err_frames;
                    global_err_bits.fetch_add(local.err_bits, std::memory_order_relaxed);
                    global_total_size.fetch_add(local.total_size, std::memory_order_relaxed);
                    local = ThreadResult();
                    local_count = 0;

                    // 检查是否达到错误帧阈值
                    if (new_err_frames >= err_frame_threshold) {
                        should_stop.store(true, std::memory_order_relaxed);
                        break;
                    }
                }
            }

            // 最后一次更新全局计数器
            global_progress.fetch_add(local.processed, std::memory_order_relaxed);
            global_err_frames.fetch_add(local.err_frames, std::memory_order_relaxed);
            global_err_bits.fetch_add(local.err_bits, std::memory_order_relaxed);
            global_total_size.fetch_add(local.total_size, std::memory_order_relaxed);
        };

        auto start = std::chrono::high_resolution_clock::now();
        std::vector<std::thread> threads;
        
        // Start worker threads
        for (unsigned int t = 0; t < num_threads; ++t) {
            threads.emplace_back(worker, seed + t);
        }

        // Progress display thread
        std::thread display_thread([&]() {
            while (!should_stop.load(std::memory_order_relaxed) && 
                   global_err_frames.load(std::memory_order_relaxed) < err_frame_threshold && 
                   global_progress.load(std::memory_order_relaxed) < max_sample) {
                if (show_progress) {
                    const int progress = global_progress.load(std::memory_order_relaxed);
                    const int ef = global_err_frames.load(std::memory_order_relaxed);
                    const int eb = global_err_bits.load(std::memory_order_relaxed);
                    const uint64_t ts = global_total_size.load(std::memory_order_relaxed);
                    
                    if (progress > 0) {
                        std::cout << "SNR " << snr << "dB - Samples: " << progress 
                                << "  ErrFrames: " << ef << "/" << err_frame_threshold
                                << "  BER: " << static_cast<float>(eb) / (progress * TxAntNum * QAM::bitLength)
                                << "  AvgListSize: " << static_cast<float>(ts) / progress  / TxAntNum / 2 / QAM::symbolsRD.size()
                                << "\r";
                        std::cout.flush();
                    }
                }
                std::this_thread::sleep_for(std::chrono::milliseconds(100));
            }
        });

        // Wait for all worker threads to complete
        for (auto& t : threads) {
            t.join();
        }
        
        // 最终结果输出
        const int progress = global_progress.load(std::memory_order_relaxed);
        const int ef = global_err_frames.load(std::memory_order_relaxed);
        const int eb = global_err_bits.load(std::memory_order_relaxed);
        const uint64_t ts = global_total_size.load(std::memory_order_relaxed);
        
        // Calculate and store current SNR results
        float ber = static_cast<float>(eb) / (progress * TxAntNum * QAM::bitLength);
        float avgListSize = static_cast<float>(ts) / progress / TxAntNum / 2 / QAM::symbolsRD.size();
        
        snr_values.push_back(snr);
        ber_values.push_back(ber);
        avg_list_size_values.push_back(avgListSize);
        
        std::cout << "SNR " << snr << "dB - Samples: " << progress 
                << "  ErrFrames: " << ef << "/" << err_frame_threshold
                << "  BER: " << ber
                << "  AvgListSize: " << avgListSize
                << std::endl;

        display_thread.join();

        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = end - start;
    }

    // Print results as vectors
    std::cout << "\n=== Summary Results ===\n";
    
    std::cout << "SNR values: [";
    for (size_t i = 0; i < snr_values.size(); i++) {
        std::cout << snr_values[i];
        if (i < snr_values.size() - 1) std::cout << ", ";
    }
    std::cout << "]\n";
    
    std::cout << "BER values: [";
    for (size_t i = 0; i < ber_values.size(); i++) {
        std::cout << std::scientific << std::setprecision(6) << ber_values[i];
        if (i < ber_values.size() - 1) std::cout << ", ";
    }
    std::cout << "]\n";
    
    std::cout << "AvgListSize values: [";
    for (size_t i = 0; i < avg_list_size_values.size(); i++) {
        std::cout << std::fixed << std::setprecision(4) << avg_list_size_values[i];
        if (i < avg_list_size_values.size() - 1) std::cout << ", ";
    }
    std::cout << "]\n";

    return 0;
}
