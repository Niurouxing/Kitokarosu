#include <Kitokarosu.hpp>

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

 
    Tensor<double,2,2> z = {1,2,3,4};
    z.print();


}