#include "Kitokarosu.hpp"
#include <iostream>


using namespace Kito;
int main()
{
    constexpr size_t TxAntNum = 32;
    constexpr size_t RxAntNum = 32;
    
    using QAM = QAM256<float>;
    auto det = Detection<Rx<RxAntNum>, Tx<TxAntNum>, Mod<QAM>>();

    det.setSNR(30);

    constexpr size_t S = 14;

    // always assume L = TxAntNum
    constexpr size_t M = S * TxAntNum * QAM::bitLength;
    
    constexpr double rate = 0.433;

    constexpr auto K = static_cast<size_t>(rate * M);

    auto ldpc = nrLDPC<K, rate>();
    ldpc.debug();


    auto cw = ldpc.encode();

    std::cout << "codeword: Size(" << cw.size() << ")\n";
    for(auto c: cw)
    {
        std::cout << c;
    }
    std::cout << std::endl;

    auto rm = std::array<bool, M>{};

    ldpc.rateMatch(rm);

    std::cout << "rate matched: Size(" << rm.size() << ")\n";
    for(auto c: rm)
    {
        std::cout << c;
    }
    std::cout << std::endl;

    auto LLR_all = std::array<double, M>{};


    for (int s = 0; s < S; s++)
    {
        std::cout << "s: " << s << std::endl;
        det.generate(rm.begin() + s * TxAntNum * QAM::bitLength);
        // std::cout << "TxSymbols: " << std::endl;
        // std::cout << det.TxSymbols.transpose() << std::endl;

        // mmse
        const auto& H = det.H;
        // std::cout << "H: " << std::endl;
        // std::cout << H << std::endl;

        const auto& y = det.RxSymbols;
        // std::cout << "y: " << std::endl;
        // std::cout << y.transpose() << std::endl;
        const auto& Nv = det.Nv;

        // // inv(H^T * H + Nv * I) * H^T * y
        // Eigen::Vector<float, 2 * TxAntNum> x = (H.transpose() * H + Nv * Eigen::Matrix<float, 2 * TxAntNum, 2 * TxAntNum>::Identity()).inverse() * H.transpose() * y;

        // std::cout << "x: " << std::endl;
        // std::cout << x << std::endl;
        // std::cout << std::endl;
        // std::cout << std::endl;

        using Matrix = Eigen::Matrix<float, 2 * TxAntNum, 2 * TxAntNum>;

        Matrix HtH = H.transpose() * H;
        Matrix W = (HtH + Nv * Matrix::Identity()).inverse() * H.transpose();

        // 等效增益矩阵对角线元素
        Eigen::Vector<float, 2 * TxAntNum> mu = (W * H).diagonal();

        // 计算等效噪声方差
        Eigen::Vector<float, 2 * TxAntNum> sigma_eff_sq;
        const float Es = 0.5; // 每个实数符号能量

        for(int i=0; i<2 * TxAntNum; ++i){
            // 残余干扰项
            float interference = (W.row(i) * H).squaredNorm() - mu[i]*mu[i];
            
            // 噪声放大项
            float noise_amp = W.row(i).squaredNorm();
            
            // 有效噪声方差 (考虑实数域能量)
            sigma_eff_sq[i] = (Es * interference + (Nv/2) * noise_amp) / (mu[i]*mu[i]);
        }

        // x_est = W * y
        Eigen::Vector<float, 2 * TxAntNum> x_est = W * y;

        // std::cout << "x_est: " << std::endl;
        // std::cout << x_est.transpose() << std::endl;

        // 符号归一化
        Eigen::Vector<float, 2 * TxAntNum> s_norm = x_est.array() / mu.array();

        // LLR计算
        constexpr int bits_per_dim = QAM::bitLength / 2;
        Eigen::Vector<float, QAM::bitLength * TxAntNum> llr;
        
        for(int i=0; i<2 * TxAntNum; ++i){
            const auto s = s_norm[i];
            const auto& symbols = QAM::symbolsRD;
            const int n_bits = bits_per_dim;
            
            for(int b=0; b<n_bits; ++b){
                float min_dist_0 = std::numeric_limits<float>::max();
                float min_dist_1 = min_dist_0;
                
                // 遍历星座点
                for(size_t k=0; k<symbols.size(); ++k){
                    const auto sym = symbols[k];
                    
                    // 提取当前比特位的值
                    const bool bit = (k >> (n_bits - 1 - b)) & 1;
                    const float dist = std::pow(s - sym, 2);
                    
                    // 更新最小距离
                    bit ? min_dist_1 = std::min(min_dist_1, dist)
                        : min_dist_0 = std::min(min_dist_0, dist);
                }
                
                // Max-Log LLR近似
                llr[i*n_bits + b] = (min_dist_1 - min_dist_0) / sigma_eff_sq[i];
            }
        }

        // std::cout << "LLR: " << std::endl;
        // std::cout << llr.transpose() << std::endl;

        std::copy(llr.data(), llr.data() + llr.size(), LLR_all.data() + s * TxAntNum * QAM::bitLength);
    
        
    }

    std::cout << "LLR_all: " << std::endl;
    for(auto l: LLR_all)
    {
        std::cout << l << " ";
    }
    std::cout << std::endl;

    ldpc.rateRecover(LLR_all);
    auto res = ldpc.decode(10);

    std::cout << "decoded: Size(" << res.size() << ")\n";
    for(auto c: res)
    {
        std::cout << c;
    }
    std::cout << std::endl;

    std::cout << "original: Size(" << ldpc.msg.size() << ")\n";
    for(auto c: ldpc.msg)
    {
        std::cout << c;
    }
    std::cout << std::endl;
}