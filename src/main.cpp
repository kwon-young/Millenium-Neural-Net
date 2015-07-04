
#include <iostream>
#include <Eigen/Dense>
#include <ctime>
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
  layers << 3, 2, 2;
  Neural_Net my_net(layers, functions);
  Eigen::VectorXd v = Eigen::VectorXd::Random(3);
  clock_t t;
  t = clock();
  my_net.compute(v);
  t = clock()-t;
  std::cout << t << " tick to compute" << std::endl;
  std::cout << (float)t/CLOCKS_PER_SEC << " sec to compute" << std::endl;
  Eigen::VectorXd y = Eigen::VectorXd::Constant(2, 1.0);
  my_net.print_layer(2);
  std::cout << "cost : " << my_net.getCost(y) << std::endl;
  //my_net.print_neural_net();
  //my_net.print_layer(2);

  return 0;
}

