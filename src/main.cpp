
#include <iostream>
#include <Eigen/Dense>
#include "Neural_Net.hpp"

int main(int argc, char *argv[])
{
   //Eigen::MatrixXd m(3, 2);
   //m(1, 1) = m(1, 0) + m(0, 1);
   //std::cout << m << std::endl;
   //Eigen::MatrixXd m2 = Eigen::MatrixXd::Random(3, 3);
   //m2 = (m2 + Eigen::MatrixXd::Constant(3, 3, 1.2)) * 50;
   //std::cout << "m2 = " << std::endl;
   //std::cout << m2 << std::endl;
   //std::cout << "v = " << std::endl;
   //std::cout << v << std::endl;
   //std::cout << "m2*v = " << std::endl;
   //std::cout << m2*v << std::endl;
   Neural_Net_Functions * functions = new Neural_Net_Functions();
   Eigen::VectorXd layers(3);
   layers << 1000, 100, 10;
   Neural_Net my_net(layers, functions);
   Eigen::VectorXd v(Eigen::VectorXd::Random(1000));
   my_net.compute(v);
   //my_net.print_neural_net();
   my_net.print_layer(2);

   return 0;
}

