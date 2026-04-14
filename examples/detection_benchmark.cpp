#include "Kitokarosu.hpp"
#include <iomanip>
#include <chrono>
#include <atomic>
#include <thread>
#include <mutex>
#include <vector>
#include <cstdint>
#include <sstream>
#include <string>
#include <functional>

using Kito::QAM16;
using Kito::QAM64;
using Kito::QAM256;

// ===================== 仿真参数配置 =====================
static constexpr size_t TxAntNum = 32;
static constexpr size_t RxAntNum = 32;
static constexpr size_t K_BEST_K = 16;     // K-Best 的 K 值
static constexpr size_t EP_ITER  = 10;     // EP 迭代次数

using QAM = QAM64<float>;
using Kito::Detection;
using Kito::Mod;
using Kito::Rx;
using Kito::Tx;
using Kito::SER;
using Kito::BER;
using Kito::FER;

using Det = Detection<Rx<RxAntNum>, Tx<TxAntNum>, Mod<QAM>>;

// ===================== 线程局部结果 =====================
struct ThreadResult {
    long long err_frames  = 0;
    long long err_bits    = 0;
    long long err_symbols = 0;
    long long processed   = 0;
};

// ===================== 单个 SNR 点的结果 =====================
struct SnrResult {
    int    snr        = 0;
    double ber        = 0;
    double ser        = 0;
    double fer        = 0;
    long long samples = 0;
};

// ===================== 算法描述 =====================
struct AlgorithmEntry {
    std::string name;
    using WorkerFactory = std::function<std::function<void(unsigned int)>(
        int snr,
        std::atomic<long long>& global_progress,
        std::atomic<long long>& global_err_frames,
        std::atomic<long long>& global_err_bits,
        std::atomic<long long>& global_err_symbols,
        std::atomic<bool>&      should_stop,
        long long max_sample,
        long long err_frame_threshold
    )>;
    WorkerFactory factory;
};

// ===================== 通用 SNR 扫描框架 =====================
std::vector<SnrResult> run_sweep(
    const std::string& algo_name,
    const AlgorithmEntry::WorkerFactory& factory,
    int snr_start, int snr_end, int snr_step,
    long long max_sample, long long err_frame_threshold,
    unsigned int seed)
{
    std::vector<SnrResult> results;

    for (int snr = snr_start; snr <= snr_end; snr += snr_step) {
        std::atomic<long long> global_progress(0);
        std::atomic<long long> global_err_frames(0);
        std::atomic<long long> global_err_bits(0);
        std::atomic<long long> global_err_symbols(0);
        std::atomic<bool>      should_stop(false);
        std::atomic<size_t>    last_progress_len(0);

        const unsigned int num_threads = std::thread::hardware_concurrency();

        auto worker = factory(snr,
                              global_progress, global_err_frames,
                              global_err_bits, global_err_symbols,
                              should_stop, max_sample, err_frame_threshold);

        auto start = std::chrono::high_resolution_clock::now();
        std::vector<std::thread> threads;
        for (unsigned int t = 0; t < num_threads; ++t)
            threads.emplace_back(worker, seed + t);

        // 显示线程
        std::thread display_thread([&]() {
            while (!should_stop.load(std::memory_order_relaxed) &&
                   global_err_frames.load(std::memory_order_relaxed) < err_frame_threshold &&
                   global_progress.load(std::memory_order_relaxed) < max_sample) {
                const long long progress = global_progress.load(std::memory_order_relaxed);
                if (progress > 0) {
                    const long long ef = global_err_frames.load(std::memory_order_relaxed);
                    const long long eb = global_err_bits.load(std::memory_order_relaxed);
                    const long long es = global_err_symbols.load(std::memory_order_relaxed);

                    double cur_ber = static_cast<double>(eb) / (progress * TxAntNum * QAM::bitLength);
                    double cur_ser = static_cast<double>(es) / (progress * 2 * TxAntNum);
                    double cur_fer = static_cast<double>(ef) / progress;

                    std::ostringstream oss;
                    oss << "[" << algo_name << "] SNR " << snr << "dB | N=" << progress
                        << " | EF=" << ef << "/" << err_frame_threshold
                        << " | BER=" << std::scientific << std::setprecision(3) << cur_ber
                        << " | SER=" << std::scientific << std::setprecision(3) << cur_ser
                        << " | FER=" << std::scientific << std::setprecision(3) << cur_fer;

                    std::string line = oss.str();
                    if (line.size() > 140) line.resize(139);

                    const size_t prev_len = last_progress_len.load(std::memory_order_relaxed);
                    std::cout << '\r' << line;
                    if (prev_len > line.size())
                        std::cout << std::string(prev_len - line.size(), ' ');
                    std::cout << '\r';
                    last_progress_len.store(line.size(), std::memory_order_relaxed);
                    std::cout.flush();
                }
                std::this_thread::sleep_for(std::chrono::milliseconds(100));
            }
        });

        for (auto& t : threads) t.join();
        display_thread.join();

        const long long progress = global_progress.load();
        const long long ef = global_err_frames.load();
        const long long eb = global_err_bits.load();
        const long long es = global_err_symbols.load();

        double ber = (progress > 0) ? static_cast<double>(eb) / (progress * TxAntNum * QAM::bitLength) : 0.0;
        double ser = (progress > 0) ? static_cast<double>(es) / (progress * 2 * TxAntNum) : 0.0;
        double fer = (progress > 0) ? static_cast<double>(ef) / progress : 0.0;

        results.push_back({snr, ber, ser, fer, progress});

        // 清除进度行
        const size_t prev_len = last_progress_len.load();
        if (prev_len > 0)
            std::cout << std::string(prev_len, ' ') << '\r';

        auto end = std::chrono::high_resolution_clock::now();
        double elapsed = std::chrono::duration<double>(end - start).count();

        std::cout << "[" << algo_name << "] SNR " << snr << "dB  N=" << progress
                  << "  EF=" << ef
                  << "  BER=" << std::scientific << std::setprecision(4) << ber
                  << "  SER=" << ser
                  << "  FER=" << fer
                  << "  " << std::fixed << std::setprecision(2) << elapsed << "s\n";
    }
    return results;
}

// ===================== Worker 工厂模板 =====================
// 仅需提供「每帧运行检测并返回估计符号」的 lambda
// RunBody 签名: (Det& det) -> result_vector

template <typename RunBody>
AlgorithmEntry::WorkerFactory make_factory(RunBody body)
{
    return [body](int snr,
                  std::atomic<long long>& global_progress,
                  std::atomic<long long>& global_err_frames,
                  std::atomic<long long>& global_err_bits,
                  std::atomic<long long>& global_err_symbols,
                  std::atomic<bool>&      should_stop,
                  long long max_sample,
                  long long err_frame_threshold)
    {
        return [=, &global_progress, &global_err_frames, &global_err_bits,
                &global_err_symbols, &should_stop](unsigned int thread_seed)
        {
            constexpr int update_interval = 10;
            Kito::set_random_seed(thread_seed);
            Det det;
            det.setSNR(snr);

            ThreadResult local;
            int local_count = 0;

            while (!should_stop.load(std::memory_order_relaxed) &&
                   global_progress.load(std::memory_order_relaxed) < max_sample) {
                det.generate();
                auto est = body(det);
                auto [ser_cnt, ber_cnt, fer_cnt] = det.template judge<SER, BER, FER>(est);

                local.err_frames  += fer_cnt;
                local.err_bits    += ber_cnt;
                local.err_symbols += ser_cnt;
                local.processed++;
                local_count++;

                if (local_count % update_interval == 0) {
                    global_progress.fetch_add(local.processed, std::memory_order_relaxed);
                    long long new_ef = global_err_frames.fetch_add(local.err_frames, std::memory_order_relaxed)
                                     + local.err_frames;
                    global_err_bits.fetch_add(local.err_bits, std::memory_order_relaxed);
                    global_err_symbols.fetch_add(local.err_symbols, std::memory_order_relaxed);
                    local = ThreadResult();
                    local_count = 0;

                    if (new_ef >= err_frame_threshold) {
                        should_stop.store(true, std::memory_order_relaxed);
                        break;
                    }
                }
            }
            // flush remaining
            global_progress.fetch_add(local.processed, std::memory_order_relaxed);
            global_err_frames.fetch_add(local.err_frames, std::memory_order_relaxed);
            global_err_bits.fetch_add(local.err_bits, std::memory_order_relaxed);
            global_err_symbols.fetch_add(local.err_symbols, std::memory_order_relaxed);
        };
    };
}

// ===================== 打印汇总表 =====================
void print_summary(const std::string& name, const std::vector<SnrResult>& res)
{
    std::cout << "\n--- " << name << " ---\n";
    std::cout << "SNR: [";
    for (size_t i = 0; i < res.size(); ++i)
        std::cout << res[i].snr << (i + 1 < res.size() ? ", " : "");
    std::cout << "]\nBER: [";
    for (size_t i = 0; i < res.size(); ++i)
        std::cout << std::scientific << std::setprecision(6) << res[i].ber << (i + 1 < res.size() ? ", " : "");
    std::cout << "]\nSER: [";
    for (size_t i = 0; i < res.size(); ++i)
        std::cout << std::scientific << std::setprecision(6) << res[i].ser << (i + 1 < res.size() ? ", " : "");
    std::cout << "]\nFER: [";
    for (size_t i = 0; i < res.size(); ++i)
        std::cout << std::scientific << std::setprecision(6) << res[i].fer << (i + 1 < res.size() ? ", " : "");
    std::cout << "]\n";
}

// ===================== main =====================
int main(int argc, char* argv[])
{
    long long max_sample          = 100000000000LL;
    long long err_frame_threshold = 1000;
    int snr_start = 18;
    int snr_end   = 27;
    int snr_step  = 1;
    unsigned int seed = 114514;

    if (argc > 1) max_sample          = atoll(argv[1]);
    if (argc > 2) err_frame_threshold = atoll(argv[2]);
    if (argc > 3) snr_start           = atoi(argv[3]);
    if (argc > 4) snr_end             = atoi(argv[4]);
    if (argc > 5) snr_step            = atoi(argv[5]);
    if (argc > 6) seed                = atoi(argv[6]);

    std::cout << "=== Detection Benchmark ===\n"
              << "  MIMO: " << TxAntNum << "x" << RxAntNum << "\n"
              << "  QAM:  " << (1 << QAM::bitLength) << "-QAM\n"
              << "  SNR:  " << snr_start << " ~ " << snr_end << " dB (step " << snr_step << ")\n"
              << "  Threads: " << std::thread::hardware_concurrency() << "\n"
              << "===========================\n\n";

    // ---- 注册所有算法 ----
    std::vector<AlgorithmEntry> algorithms;

    // 1. MMSE
    algorithms.push_back({"MMSE", make_factory([](Det& det) {
        auto mmse = Kito::MMSE<QAM, typename Det::PrecType, TxAntNum, RxAntNum>(
            det.H, det.RxSymbols, static_cast<typename Det::PrecType>(det.Nv));
        return mmse.normalized_symbols();
    })});

    // 2. K-Best
    algorithms.push_back({"KBest-" + std::to_string(K_BEST_K), make_factory([](Det& det) {
        thread_local auto kbest = Kito::KBest<Det, K_BEST_K>();
        return kbest.run(det);
    })});

    // 3. EP
    algorithms.push_back({"EP-" + std::to_string(EP_ITER), make_factory([](Det& det) {
        thread_local auto ep = Kito::EP<Det, EP_ITER>();
        return ep.run(det);
    })});

    // ---- 逐算法运行 ----
    std::vector<std::pair<std::string, std::vector<SnrResult>>> all_results;

    for (const auto& algo : algorithms) {
        std::cout << "\n>>>>> Running: " << algo.name << " <<<<<\n";
        auto res = run_sweep(algo.name, algo.factory,
                             snr_start, snr_end, snr_step,
                             max_sample, err_frame_threshold, seed);
        all_results.emplace_back(algo.name, std::move(res));
    }

    // ---- 汇总输出 ----
    std::cout << "\n\n========== Summary ==========\n";
    for (const auto& [name, res] : all_results)
        print_summary(name, res);

    return 0;
}