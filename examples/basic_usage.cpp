#include "Kitokarosu.hpp"
#include <iostream>
#include <numeric>
#include <vector>
#include <array>
#include <iomanip> // 用于格式化输出
#include <cmath>   // 用于 std::round, 避免浮点数循环的精度问题

using namespace Kito;

// --- 1. 参数配置结构化 (遵循您的要求) ---
// 使用简短的名称，所有参数均为 static constexpr 以支持模板编译。
struct SimConfig {
    // MIMO 和调制参数
    static constexpr size_t TxAntNum = 4;
    static constexpr size_t RxAntNum = 4;
    using QAM = QAM16<float>;

    // SNR 扫描范围设置
    static constexpr double snr_start_db = 0.0;
    static constexpr double snr_end_db = 10.0;
    static constexpr double snr_step_db = 2.0;

    // 每个SNR点的仿真停止条件
    static constexpr long long max_frame_errors = 100;
    static constexpr long long max_total_frames = 100000;

    // 帧结构和编码参数
    static constexpr size_t S = 10;
    static constexpr double ldpc_rate = 0.5;
    static constexpr int ldpc_max_iter = 10;

    // 派生参数 (自动计算)
    static constexpr size_t M = S * TxAntNum * QAM::bitLength;
    static constexpr size_t K = static_cast<size_t>(ldpc_rate * M);
};

// 用于存储每个SNR点的仿真结果
struct SimulationResult {
    double snr_db;
    long long total_frames;
    long long frame_errors;
    long long bit_errors;
    double ber;
    double fer;
};

// --- 4. 实时进度显示函数 ---
void print_progress(double snr_db, long long frames, long long max_frames, long long err_frames, long long max_err_frames) {
    // 使用 \r 回到行首来刷新当前行，实现动态显示
    std::cout << "\r" << std::fixed << std::setprecision(1)
              << "--> SNR: " << snr_db << " dB | "
              << "Frames: " << frames << "/" << max_frames << " | "
              << "Errors: " << err_frames << "/" << max_err_frames
              << "   " << std::flush; // 添加空格以清除行尾可能残留的字符
}

// 打印最终结果的表格
void print_summary_table(const std::vector<SimulationResult>& results) {
    std::cout << "\n\n--- Simulation Summary ---" << std::endl;
    std::cout << "+----------+--------------------+----------------+----------------+------------+------------+" << std::endl;
    std::cout << "| SNR (dB) |      Frames        |  Frame Errors  |   Bit Errors   |    FER     |    BER     |" << std::endl;
    std::cout << "+----------+--------------------+----------------+----------------+------------+------------+" << std::endl;

    std::cout << std::scientific << std::setprecision(3); // 设置科学计数法格式
    for (const auto& res : results) {
        std::cout << "| " << std::setw(8) << std::fixed << std::setprecision(2) << res.snr_db << " | "
                  << std::setw(18) << res.total_frames << " | "
                  << std::setw(14) << res.frame_errors << " | "
                  << std::setw(14) << res.bit_errors << " | "
                  << std::scientific << std::setw(10) << res.fer << " | "
                  << std::scientific << std::setw(10) << res.ber << " |"
                  << std::endl;
    }
    std::cout << "+----------+--------------------+----------------+----------------+------------+------------+" << std::endl;
}

int main() {
    // 打印固定的系统参数
    std::cout << "--- System Parameters ---" << std::endl;
    std::cout << "  MIMO Config: " << SimConfig::TxAntNum << "x" << SimConfig::RxAntNum << std::endl;
    std::cout << "  Modulation:  " << "16-QAM" << std::endl; // 可以做得更通用，但目前写死即可
    std::cout << "  LDPC Rate:   " << SimConfig::ldpc_rate << std::endl;
    std::cout << "  Info bits/Frame (K): " << SimConfig::K << std::endl;
    std::cout << "  Coded bits/Frame (M): " << SimConfig::M << std::endl;
    std::cout << "-------------------------" << std::endl;

    // 用于存储所有SNR点的结果
    std::vector<SimulationResult> results_vec;

    // --- 2. SNR 扫描循环 (采用整数循环以避免浮点数精度问题) ---
    const int num_snr_steps = static_cast<int>(std::round((SimConfig::snr_end_db - SimConfig::snr_start_db) / SimConfig::snr_step_db));

    for (int i = 0; i <= num_snr_steps; ++i) {
        const double snr_db = SimConfig::snr_start_db + i * SimConfig::snr_step_db;
        
        // 初始化当前SNR点的仿真器和状态变量
        auto det = Detection<Rx<SimConfig::RxAntNum>, Tx<SimConfig::TxAntNum>, Mod<SimConfig::QAM>>();
        det.setSNR(snr_db);

        long long total_bit_errors = 0;
        long long total_frame_errors = 0;
        long long frame_count = 0;

        std::cout << "\nStarting simulation for SNR = " << std::fixed << std::setprecision(2) << snr_db << " dB..." << std::endl;

        // --- 3. 动态仿真停止条件的帧循环 ---
        while (total_frame_errors < SimConfig::max_frame_errors && frame_count < SimConfig::max_total_frames)
        {
            // 1. LDPC 编码与速率匹配 (使用 SimConfig 中的模板参数)
            auto ldpc = nrLDPC<SimConfig::K, SimConfig::ldpc_rate>();
            ldpc.encode();

            auto rm = std::array<bool, SimConfig::M>{};
            ldpc.rateMatch(rm);

            // 2. MIMO 检测与 LLR 计算
            auto LLR_all = std::array<double, SimConfig::M>{};
            for (size_t s = 0; s < SimConfig::S; s++)
            {
                det.generate(rm.begin() + s * SimConfig::TxAntNum * SimConfig::QAM::bitLength);

                auto mmse = MMSE<SimConfig::QAM, float, SimConfig::TxAntNum, SimConfig::RxAntNum>(det.H, det.RxSymbols, static_cast<float>(det.Nv));
                mmse.compute_llr();

                std::copy(mmse.llr.data(), mmse.llr.data() + mmse.llr.size(),
                          LLR_all.data() + s * SimConfig::TxAntNum * SimConfig::QAM::bitLength);
            }

            // 3. LDPC 速率恢复与译码
            ldpc.rateRecover(LLR_all);
            auto res = ldpc.decode(SimConfig::ldpc_max_iter);

            // 4. 错误统计
            size_t current_frame_bit_errors = 0;
            if (res.size() == ldpc.msg.size()) {
                for (size_t k = 0; k < SimConfig::K; ++k) {
                    if (res[k] != ldpc.msg[k]) {
                        current_frame_bit_errors++;
                    }
                }
            } else {
                current_frame_bit_errors = SimConfig::K;
            }

            if (current_frame_bit_errors > 0) {
                total_bit_errors += current_frame_bit_errors;
                total_frame_errors++;
            }
            frame_count++;

            // 持续显示当前SNR仿真情况 (有错误时或每10帧更新一次)
            if (total_frame_errors > 0 || frame_count % 10 == 0) {
                 print_progress(snr_db, frame_count, SimConfig::max_total_frames, total_frame_errors, SimConfig::max_frame_errors);
            }
        }
        
        // 确保进度条被清除并换行
        std::cout << std::endl; 

        // 计算并存储当前SNR点的结果
        double ber = (frame_count == 0) ? 0.0 : (static_cast<double>(total_bit_errors)) / (static_cast<double>(frame_count) * SimConfig::K);
        double fer = (frame_count == 0) ? 0.0 : (static_cast<double>(total_frame_errors)) / (static_cast<double>(frame_count));
        
        results_vec.push_back({snr_db, frame_count, total_frame_errors, total_bit_errors, ber, fer});

        // 打印单点仿真结束信息
        std::cout << "Finished. Frames: " << frame_count << ", FER: " << fer << ", BER: " << ber << std::endl;
    }

    // --- 5. 结果汇总与展示 ---
    print_summary_table(results_vec);

    return 0;
}