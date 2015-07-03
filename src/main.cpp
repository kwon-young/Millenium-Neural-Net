
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
   //Eigen::VectorXd v(3);
   //v << 1, 2, 3;
   //std::cout << "v = " << std::endl;
   //std::cout << v << std::endl;
   //std::cout << "m2*v = " << std::endl;
   //std::cout << m2*v << std::endl;
   Neural_Net_Functions * functions = new Neural_Net_Functions();
   Eigen::VectorXd layers(3);
   layers << 3, 2, 1;
   Neural_Net my_net(layers, functions);
   my_net.print_neural_net();

   return 0;
}

