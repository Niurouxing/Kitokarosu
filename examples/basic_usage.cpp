#include "Kitokarosu.hpp"
#include <iostream>

using namespace Kito;
int main() {
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
  for (auto c : cw) {
    std::cout << c;
  }
  std::cout << std::endl;

  auto rm = std::array<bool, M>{};

  ldpc.rateMatch(rm);

  std::cout << "rate matched: Size(" << rm.size() << ")\n";
  for (auto c : rm) {
    std::cout << c;
  }
  std::cout << std::endl;

  auto LLR_all = std::array<double, M>{};

  for (int s = 0; s < S; s++) {
    // 修改后的主函数检测部分
    for (int s = 0; s < S; s++) {
      det.generate(rm.begin() + s * TxAntNum * QAM::bitLength);

      // 创建MMSE检测器实例
      MMSE<QAM, float, TxAntNum, RxAntNum> mmse(
          det.H, det.RxSymbols, static_cast<float>(det.Nv));

      // 计算LLR
      mmse.compute_llr();

      // 获取计算结果
      const auto &current_llr = mmse.get_llr();
      std::copy(current_llr.data(), current_llr.data() + current_llr.size(),
                LLR_all.data() + s * TxAntNum * QAM::bitLength);
    }
  }

  std::cout << "LLR_all: " << std::endl;
  for (auto l : LLR_all) {
    std::cout << l << " ";
  }
  std::cout << std::endl;

  ldpc.rateRecover(LLR_all);
  auto res = ldpc.decode(10);

  std::cout << "decoded: Size(" << res.size() << ")\n";
  for (auto c : res) {
    std::cout << c;
  }
  std::cout << std::endl;

  std::cout << "original: Size(" << ldpc.msg.size() << ")\n";
  for (auto c : ldpc.msg) {
    std::cout << c;
  }
  std::cout << std::endl;
}