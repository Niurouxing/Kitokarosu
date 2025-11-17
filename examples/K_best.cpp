#include "Kitokarosu.hpp"
#include <iomanip>
#include <chrono>
#include <atomic>
#include <thread>
#include <mutex>
#include <vector>
#include <atomic>
#include <cstdint>

using Kito::QAM16;
using Kito::QAM64;
using Kito::QAM256;

static constexpr size_t TxAntNum = 8;
static constexpr size_t RxAntNum = 8;
static constexpr size_t K = 32;  // K-best size

using QAM = QAM16<float>;
using Kito::Detection;
using Kito::Mod;
using Kito::Rx;
using Kito::Tx;
using Kito::SER;
using Kito::BER;

struct ThreadResult {
    long long err_frames = 0; 
    long long err_bits = 0;
    long long err_symbols = 0;
    uint64_t total_size = 0;
    long long processed = 0;    
};

int main(int argc, char* argv[]) {
    long long max_sample = 100000000000;
    long long err_frame_threshold = 1000; 
    int snr_start = 9;
    int snr_end = 18;
    int snr_step = 1;
    unsigned int seed = 114514;

    // 变化 6: 使用 atoll 解析 long long 类型的参数
    if (argc > 1) max_sample = atoll(argv[1]);
    if (argc > 2) err_frame_threshold = atoll(argv[2]);
    if (argc > 3) snr_start = atoi(argv[3]);
    if (argc > 4) snr_end = atoi(argv[4]);
    if (argc > 5) snr_step = atoi(argv[5]);
    if (argc > 6) seed = atoi(argv[6]);

    const bool show_progress = true;
    constexpr int update_interval = 10;
    
    std::vector<int> snr_values;
    std::vector<double> ber_values;
    std::vector<double> ser_values;
    std::vector<double> avg_list_size_values;

    for (int snr = snr_start; snr <= snr_end; snr += snr_step) {
        
        std::atomic<long long> global_progress(0);
        std::atomic<long long> global_err_frames(0);
        std::atomic<long long> global_err_bits(0);
        std::atomic<long long> global_err_symbols(0);
        std::atomic<uint64_t> global_total_size(0);
        std::atomic<bool> should_stop(false);

        const unsigned int num_threads = std::thread::hardware_concurrency();

        auto worker = [&](unsigned int thread_seed) {
            Kito::set_random_seed(thread_seed);
            auto det = Detection<Rx<RxAntNum>, Tx<TxAntNum>, Mod<QAM>>();
            det.setSNR(snr);
            auto tree = Kito::SphereDecoder<decltype(det)>();
            
            ThreadResult local;
            int local_count = 0;

            while (!should_stop.load(std::memory_order_relaxed) && 
                   global_progress.load(std::memory_order_relaxed) < max_sample) {
                det.generate();
                auto list = tree.run(det);
                auto [ser_cnt, ber_cnt] = det.judge<SER, BER>(list);

                local.err_frames += ((ser_cnt > 0) || (ber_cnt > 0));
                local.err_bits += ber_cnt;
                local.err_symbols += ser_cnt;
                local.total_size += tree.nodes;
                local.processed++;
                local_count++;

                if (local_count % update_interval == 0) {
                    global_progress.fetch_add(local.processed, std::memory_order_relaxed);
                    long long new_err_frames = global_err_frames.fetch_add(local.err_frames, std::memory_order_relaxed) + local.err_frames;
                    global_err_bits.fetch_add(local.err_bits, std::memory_order_relaxed);
                    global_total_size.fetch_add(local.total_size, std::memory_order_relaxed);
                    global_err_symbols.fetch_add(local.err_symbols, std::memory_order_relaxed);
                    local = ThreadResult();
                    local_count = 0;

                    if (new_err_frames >= err_frame_threshold) {
                        should_stop.store(true, std::memory_order_relaxed);
                        break;
                    }
                }
            }

            global_progress.fetch_add(local.processed, std::memory_order_relaxed);
            global_err_frames.fetch_add(local.err_frames, std::memory_order_relaxed);
            global_err_bits.fetch_add(local.err_bits, std::memory_order_relaxed);
            global_total_size.fetch_add(local.total_size, std::memory_order_relaxed);
            global_err_symbols.fetch_add(local.err_symbols, std::memory_order_relaxed);
        };

        auto start = std::chrono::high_resolution_clock::now();
        std::vector<std::thread> threads;
        
        for (unsigned int t = 0; t < num_threads; ++t) {
            threads.emplace_back(worker, seed + t);
        }

        std::thread display_thread([&]() {
            while (!should_stop.load(std::memory_order_relaxed) && 
                   global_err_frames.load(std::memory_order_relaxed) < err_frame_threshold && 
                   global_progress.load(std::memory_order_relaxed) < max_sample) {
                if (show_progress) {
                    const long long progress = global_progress.load(std::memory_order_relaxed);
                    const long long ef = global_err_frames.load(std::memory_order_relaxed);
                    const long long eb = global_err_bits.load(std::memory_order_relaxed);
                        const uint64_t ts = global_total_size.load(std::memory_order_relaxed);
                        const long long es = global_err_symbols.load(std::memory_order_relaxed);
                    
                    if (progress > 0) {
                        double current_ber = static_cast<double>(eb) / (progress * TxAntNum * QAM::bitLength);
                        double current_ser = static_cast<double>(es) / (progress * 2 * TxAntNum);
                        double current_avg_list_size = static_cast<double>(ts) / progress / TxAntNum / 2 / QAM::symbolsRD.size();
                        std::cout << "SNR " << snr << "dB - Samples: " << progress 
                                << "  ErrFrames: " << ef << "/" << err_frame_threshold
                            << "  BER: " << std::scientific << current_ber // 使用科学计数法显示BER
                            << "  SER: " << std::scientific << current_ser
                                << "  AvgListSize: " << std::fixed << current_avg_list_size
                                << "\r";
                        std::cout.flush();
                    }
                }
                std::this_thread::sleep_for(std::chrono::milliseconds(100));
            }
        });

        for (auto& t : threads) {
            t.join();
        }
        display_thread.join(); // 等待显示线程结束
        
        const long long progress = global_progress.load(std::memory_order_relaxed);
        const long long ef = global_err_frames.load(std::memory_order_relaxed);
        const long long eb = global_err_bits.load(std::memory_order_relaxed);
        const long long es = global_err_symbols.load(std::memory_order_relaxed);
        const uint64_t ts = global_total_size.load(std::memory_order_relaxed);
        
        double ber = (progress > 0) ? static_cast<double>(eb) / (progress * TxAntNum * QAM::bitLength) : 0.0;
        double ser = (progress > 0) ? static_cast<double>(es) / (progress * 2 * TxAntNum) : 0.0;
        double avgListSize = (progress > 0) ? static_cast<double>(ts) / progress / TxAntNum / 2 / QAM::symbolsRD.size() : 0.0;
        
        snr_values.push_back(snr);
        ber_values.push_back(ber);
        ser_values.push_back(ser);
        avg_list_size_values.push_back(avgListSize);
        
        // 清除进度行，然后打印最终结果
        std::cout << std::string(120, ' ') << "\r"; // 用空格覆盖之前的内容
        std::cout << "SNR " << snr << "dB - Samples: " << progress 
                << "  ErrFrames: " << ef << "/" << err_frame_threshold
                << "  BER: " << std::scientific << ber
            << "  SER: " << std::scientific << ser
                << "  AvgListSize: " << std::fixed << avgListSize
                << std::endl;


        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = end - start;
        // 可以选择性地打印每个SNR点的耗时
        std::cout << "Time elapsed for SNR " << snr << "dB: " << elapsed.count() << " seconds\n";
    }

    std::cout << "\n=== Summary Results ===\n";
    
    std::cout << "SNR values: [";
    for (size_t i = 0; i < snr_values.size(); i++) {
        std::cout << snr_values[i] << (i < snr_values.size() - 1 ? ", " : "");
    }
    std::cout << "]\n";
    
    std::cout << "BER values: [";
    for (size_t i = 0; i < ber_values.size(); i++) {
        std::cout << std::scientific << std::setprecision(6) << ber_values[i] << (i < ber_values.size() - 1 ? ", " : "");
    }
    std::cout << "]\n";

    std::cout << "SER values: [";
    for (size_t i = 0; i < ser_values.size(); i++) {
        std::cout << std::scientific << std::setprecision(6) << ser_values[i] << (i < ser_values.size() - 1 ? ", " : "");
    }
    std::cout << "]\n";
    
    std::cout << "AvgListSize values: [";
    for (size_t i = 0; i < avg_list_size_values.size(); i++) {
        std::cout << std::fixed << std::setprecision(4) << avg_list_size_values[i] << (i < avg_list_size_values.size() - 1 ? ", " : "");
    }
    std::cout << "]\n";

    return 0;
}