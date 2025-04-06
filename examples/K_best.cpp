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

static constexpr size_t TxAntNum = 128;
static constexpr size_t RxAntNum = 128;
using QAM = QAM256<float>;
using Kito::Detection;
using Kito::Mod;
using Kito::Rx;
using Kito::Tx;

struct ThreadResult {
    int err_frames = 0;
    int err_bits = 0;
    int total_size = 0;
    int processed = 0;
};

int main(int argc, char* argv[]) {
    Eigen::setNbThreads(1); 
    int sample = 1000000;
    int snr = 28;
    int K = 6400;
    unsigned int seed = 114514;

    if (argc > 1) sample = atoi(argv[1]);
    if (argc > 2) snr = atoi(argv[2]);
    if (argc > 3) K = atoi(argv[3]);
    if (argc > 4) seed = atoi(argv[4]);

    const bool show_progress = (argc <= 1);
    constexpr int update_interval = 1;  // 每100个样本更新一次显示

    // 原子计数器用于进度跟踪
    std::atomic<int> global_progress(0);
    std::atomic<int> global_err_frames(0);
    std::atomic<int> global_err_bits(0);
    std::atomic<int> global_total_size(0);

    const unsigned int num_threads = std::thread::hardware_concurrency();
    const int samples_per_thread = sample / num_threads;
    const int remaining_samples = sample % num_threads;

    auto worker = [&](int thread_samples, unsigned int thread_seed) {
        Kito::set_random_seed(thread_seed);
        auto det = Detection<Rx<RxAntNum>, Tx<TxAntNum>, Mod<QAM>>();
        det.setSNR(snr);
        auto tree = Kito::KBest<decltype(det),1024>();
        
        ThreadResult local;

        for (int i = 0; i < thread_samples; ++i) {
            det.generate();
            auto list = tree.run(det);
            auto err = det.judge(list);

            local.err_frames += (err > 0);
            local.err_bits += err;
            local.processed++;

            // 定期更新全局计数器
            if (local.processed % update_interval == 0) {
                global_progress.fetch_add(local.processed, std::memory_order_relaxed);
                global_err_frames.fetch_add(local.err_frames, std::memory_order_relaxed);
                global_err_bits.fetch_add(local.err_bits, std::memory_order_relaxed);
                local = ThreadResult();  // 重置本地计数器
            }
        }

        // 处理剩余未提交的样本
        global_progress.fetch_add(local.processed, std::memory_order_relaxed);
        global_err_frames.fetch_add(local.err_frames, std::memory_order_relaxed);
        global_err_bits.fetch_add(local.err_bits, std::memory_order_relaxed);
    };

    auto start = std::chrono::high_resolution_clock::now();
    std::vector<std::thread> threads;
    
    // 启动工作线程
    for (unsigned int t = 0; t < num_threads; ++t) {
        const int actual_samples = samples_per_thread + (t == num_threads - 1 ? remaining_samples : 0);
        threads.emplace_back(worker, actual_samples, seed + t);
    }

    // 进度显示线程
    std::thread display_thread([&]() {
        while (global_progress.load() < sample) {
            if (show_progress) {
                const int progress = global_progress.load(std::memory_order_relaxed);
                const int ef = global_err_frames.load(std::memory_order_relaxed);
                const int eb = global_err_bits.load(std::memory_order_relaxed);
                const int ts = global_total_size.load(std::memory_order_relaxed);
                
                if (progress > 0) {
                    std::cout << "Progress: " << progress << "/" << sample
                              << "  ErrFrames: " << ef
                              << "  BER: " << static_cast<float>(eb) / (progress * TxAntNum * QAM::bitLength)
                              << "  AvgListSize: " << static_cast<float>(ts) / progress / TxAntNum / 2 / QAM::symbolsRD.size()
                              << "  Progress Per Second: " << static_cast<float>(progress) / (std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - start).count())
                              << "\r";
                    std::cout.flush();
                }
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(100));  // 每100ms更新一次
        }
    });

    // 等待所有工作线程完成
    for (auto& t : threads) {
        t.join();
    }
    
    // 确保最后一次更新显示
    if (show_progress) {
        const int ef = global_err_frames.load();
        const int eb = global_err_bits.load();
        const int ts = global_total_size.load();
        std::cout << "Progress: " << sample << "/" << sample
                  << "  ErrFrames: " << ef
                  << "  BER: " << static_cast<float>(eb) / (sample * TxAntNum * QAM::bitLength)
                  << "  AvgListSize: " << static_cast<float>(ts) / sample / TxAntNum / 2 / QAM::symbolsRD.size()
                  << std::endl;
    }

    display_thread.join();

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;

    // 最终结果输出保持不变
    std::cout << "Elapsed time: " << elapsed.count() << " seconds" << std::endl;
    std::cout << "Total Err Bits: " << global_err_bits.load() << std::endl;
    std::cout << "Total Err Frames: " << global_err_frames.load() << std::endl;
    std::cout << "BER: " << static_cast<float>(global_err_bits) / (sample * TxAntNum * QAM::bitLength) << std::endl;
    std::cout << "FER: " << static_cast<float>(global_err_frames) / sample << std::endl;
    std::cout << "List Size: " << global_total_size.load() << std::endl;

    return 0;
}