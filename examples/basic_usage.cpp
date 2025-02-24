#include "Kitokarosu.hpp"


using namespace Kito;
int main()
{
    static constexpr size_t TxAntNum = 8;
    static constexpr size_t RxAntNum = 8;
    
    using QAM = QAM16<float>;
    auto det = Detection<Rx<RxAntNum>, Tx<TxAntNum>, Mod<QAM>>();


    constexpr auto len = 2 * TxAntNum * QAM::bitLength;

    std::array<bool, len> bitsInput;

    // a function randomly generate bits
    std::generate(bitsInput.begin(), bitsInput.end(), []() {
        return std::rand() % 2;
    }); 

    // print the generated bits
    for(int i = 0; i < 2 * TxAntNum; i++)
    {
        for(int j = 0; j < QAM::bitLength/2; j++)
        {
            std::cout << bitsInput[i * QAM::bitLength/2 + j] << " ";
        }
        std::cout << std::endl;
    }
 

    det.generateTx(bitsInput);

    for(auto indice : det.TxIndices)
    {
        std::cout << indice << " ";
    }
    std::cout << std::endl;

    for (auto symbol : det.TxSymbols)
    {
        std::cout << symbol << " ";
    }
    std::cout << std::endl;

}