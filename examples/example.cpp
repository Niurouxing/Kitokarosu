#include <Kitokarosu/Kitokarosu.hpp>

using namespace Kito;

int main()
{

    auto det = Detection<
    Rx<2>,
    Tx<2>
    >::create();

    det.setSNR(100);

    det.generateTxIndices();
    det.generateTxSymbols();

    det.generateH();

    det.generateRxSymbols();

 

    std::cout << "TxIndices: ";
    for (auto i : det.TxIndices)
        std::cout << i << " ";
    std::cout << std::endl;

    std::cout << "TxSymbols: ";
    for (auto i : det.TxSymbols)
        std::cout << i << " ";
    std::cout << std::endl;

    std::cout << "H: " << std::endl;
    for (size_t i = 0; i < 2 * det.RxAntNum; i++)
    {
        for (size_t j = 0; j < 2 * det.TxAntNum; j++)
        {
            std::cout << det.H_span[i, j] << " ";
        }
        std::cout << std::endl;
    }

    std::cout << "RxSymbols: ";
    for (auto i : det.RxSymbols)
        std::cout << i << " ";
    std::cout << std::endl;

    auto symbolsEst = det.TxSymbols;




}