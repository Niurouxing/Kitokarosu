// main.cpp

#include "Kitokarosu.hpp"
#include <iostream>
#include <numeric>
#include <vector>
#include <array> // 确保包含了<array>头文件

using namespace Kito;
int main()
{
    // --- 仿真参数定义 ---
    constexpr size_t TxAntNum = 4;
    constexpr size_t RxAntNum = 4;
    using QAM = QAM16<float>;

    constexpr double snr_db = 30.0;
    constexpr size_t S = 10;
    constexpr int ldpc_max_iter = 10;
    const int numFrames = 2; // 总共要仿真的帧数

    set_random_seed(123); // 设置随机数种子

    // L = TxAntNum, M = 符号数 * 每符号比特数
    constexpr size_t M = S * TxAntNum * QAM::bitLength;
    constexpr double rate = 0.5;
    constexpr auto K = static_cast<size_t>(rate * M); // 每帧的信息比特数

    // --- 仿真器和状态变量初始化 ---
    auto det = Detection<Rx<RxAntNum>, Tx<TxAntNum>, Mod<QAM>>();
    det.setSNR(snr_db);

    long long total_bit_errors = 0;
    long long total_frame_errors = 0;

    // --- 主仿真循环 ---
    for (int frame = 0; frame < numFrames; ++frame)
    {
        // 1. LDPC 编码与速率匹配
        // 在循环内创建，以确保每帧都生成新的随机信息比特
        auto ldpc = nrLDPC<K, rate>();
        ldpc.encode();

        auto rm = std::array<bool, M>{};
        ldpc.rateMatch(rm);

        // 2. MIMO 检测与 LLR 计算
        auto LLR_all = std::array<double, M>{};
        // (修正) 原代码中存在错误的嵌套循环，这里已修正为单层循环
        for (int s = 0; s < S; s++)
        {
            det.generate(rm.begin() + s * TxAntNum * QAM::bitLength);

            auto mmse = MMSE<QAM, float,TxAntNum, RxAntNum>(det.H, det.RxSymbols, static_cast<float>(det.Nv));

            mmse.compute_llr();

            std::copy(mmse.llr.data(), mmse.llr.data() + mmse.llr.size(),
                      LLR_all.data() + s * TxAntNum * QAM::bitLength);
        }

        // 3. LDPC 速率恢复与译码
        ldpc.rateRecover(LLR_all);
        auto res = ldpc.decode(ldpc_max_iter);

        // 4. 错误统计
        size_t current_frame_bit_errors = 0;
        // 确保译码输出的比特数和原始信息比特数一致
        if (res.size() == ldpc.msg.size())
        {
            for (size_t i = 0; i < K; ++i)
            {
                if (res[i] != ldpc.msg[i])
                {
                    current_frame_bit_errors++;
                }
            }
        }
        else // 如果长度不一致，说明发生严重错误，整帧计为错误
        {
            current_frame_bit_errors = K;
        }

        if (current_frame_bit_errors > 0)
        {
            total_bit_errors += current_frame_bit_errors;
            total_frame_errors++;
        }
        // 打印进度
        std::cout << "Frames Processed: " << frame + 1 << "/" << numFrames << "\r";
        std::cout.flush();
    }
    std::cout << std::endl; // 结束进度条的换行

    // --- 计算并输出最终结果 ---
    double ber = (static_cast<double>(total_bit_errors)) / (static_cast<double>(numFrames) * K);
    double fer = (static_cast<double>(total_frame_errors)) / (static_cast<double>(numFrames));

    std::cout << "--- Simulation Finished ---" << std::endl;
    std::cout << "Parameters:" << std::endl;
    std::cout << "  MIMO Config: " << TxAntNum << "x" << RxAntNum << std::endl;
    std::cout << "  Modulation:  " << "16-QAM" << std::endl;
    std::cout << "  SNR:         " << snr_db << " dB" << std::endl;
    std::cout << "  LDPC Rate:   " << rate << std::endl;
    std::cout << "  Info bits/Frame (K): " << K << std::endl;
    std::cout << "---------------------------" << std::endl;
    std::cout << "Results:" << std::endl;
    std::cout << "  Total Frames: " << numFrames << std::endl;
    std::cout << "  Frame Errors: " << total_frame_errors << std::endl;
    std::cout << "  Bit Errors:   " << total_bit_errors << std::endl;
    std::cout << "  FER: " << fer << std::endl;
    std::cout << "  BER: " << ber << std::endl;
    std::cout << "---------------------------" << std::endl;

    return 0;
}