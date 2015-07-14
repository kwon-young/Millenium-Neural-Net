
#include <iostream>
#include <Eigen/Dense>
#include <ctime>
#include "Neural_Net.hpp"
#include "FNN_Model.hpp"

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
  
/*
 *  Neural_Net_Functions * functions = new Neural_Net_Functions();
 *  Eigen::VectorXd layers(3);
 *  layers << 3, 2, 2;
 *  Neural_Net my_net(layers, functions);
 *  Eigen::MatrixXd x = Eigen::MatrixXd::Constant(3, 100, 2.0);
 *  Eigen::MatrixXd y = Eigen::MatrixXd::Constant(2, 100, 1.5);
 *  my_net.print_neural_net();
 *  clock_t t;
 *  t = clock();
 *  my_net.backPropagation(x, y);
 *  t = clock()-t;
 *  std::cout << t << " tick to compute" << std::endl;
 *  std::cout << (float)t/CLOCKS_PER_SEC << " sec to compute" << std::endl;
 *
 *  my_net.print_neural_net();
 */
  //Eigen::VectorXd x = Eigen::VectorXd::Constant(3, 2.0);
  //Eigen::VectorXd y = Eigen::VectorXd::Constant(2, 1.5);
  //x << 1, 2, 3;
  //y << 4, 5;
  //std::cout << "x" << std::endl;
  //std::cout << x << std::endl;
  //std::cout << "y" << std::endl;
  //std::cout << y << std::endl;
  //std::cout << "x*yt" << std::endl;
  //std::cout << x*y.transpose() << std::endl;
  //return 0;
  
  
  std::vector<unsigned int> layers;
  layers.push_back(2);
  layers.push_back(1);
  FNN_Model my_net(layers);
  Eigen::MatrixXd inputs(2, 4);
  inputs << 0, 0, 1, 1,
            0, 1, 0, 1;
  Eigen::MatrixXd d_outputs(1, 4);
  d_outputs << 0, 0, 0, 1;
  my_net.print_FNN();
  my_net.train(inputs, d_outputs, 10000, 4, 0.5);
  my_net.print_FNN();
   

  //Eigen::MatrixXd foo1 = Eigen::MatrixXd::Constant(4, 4, 1.0);
  //Eigen::MatrixXd foo2 = Eigen::MatrixXd::Constant(4, 4, 1.0);
  //Eigen::MatrixXd foo(1, 1);
  //foo.resize(foo1.rows(), foo1.cols()+foo2.cols());
  //foo << foo1, foo2;

  //std::cout << foo << std::endl;
}

