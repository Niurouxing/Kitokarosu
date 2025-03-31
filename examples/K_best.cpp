
#include "Kitokarosu.hpp"
using Kito::QAM16;
using Kito::QAM64;
using Kito::QAM256;


static constexpr size_t TxAntNum = 32;
static constexpr size_t RxAntNum = 32;

using QAM = QAM64<float>;
using Kito::Detection;
using Kito::Mod;
using Kito::Rx;
using Kito::Tx;


int main() {

    int maxLoop = 100;
    auto det = Detection<Rx<RxAntNum>, Tx<TxAntNum>, Mod<QAM>>();
    auto alg = Kito::KBest<decltype(det), 32>();
    det.setSNR(27);
    int errCount = 0;

    for (int i = 0; i < maxLoop; ++i) {


        det.generate();
        auto res = alg.run(det);
        auto err = det.judge(res);
        errCount += err;
    }

    std::cout << static_cast<float>(errCount) / maxLoop / TxAntNum / QAM::bitLength << std::endl;




}