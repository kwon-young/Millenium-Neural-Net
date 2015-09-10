
#include <iostream>
#include <iomanip>
#include <Eigen/Dense>
#include <ctime>
#include <chrono>
#include <map>
#include <random>
#include <cstdio> // popen
#include <cstring> // memset

#include "Neural_Net.hpp"
#include "FNN_Model.hpp"
#include "MNIST_Parser.hpp"
#include "SFML/System.hpp"

int main(int argc, char *argv[])
{

  // initialize the srand
  srand (time(NULL));

  sf::RenderWindow window(
      sf::VideoMode(600, 400),
      "SFML plot",
      sf::Style::Default);

  sf::Clock clock;

  std::vector<unsigned int> layers;
  layers.push_back(1);
  layers.push_back(1);
  FNN_Model my_net(layers);

  std::vector<Eigen::MatrixXd> weights;
  std::vector<Eigen::MatrixXd> bias;
  weights.push_back(Eigen::MatrixXd());
  weights.push_back(Eigen::MatrixXd::Constant(1, 1, 0.6));
  bias.push_back(Eigen::MatrixXd());
  bias.push_back(Eigen::MatrixXd::Constant(1, 1, 0.9));
  my_net.Manual_Set_FNN(weights, bias);
  Eigen::MatrixXd input = Eigen::MatrixXd::Constant(1, 1, 1);
  Eigen::MatrixXd output = Eigen::MatrixXd::Constant(1, 1, 0);
  Eigen::MatrixXd eval_input = Eigen::MatrixXd::Constant(1, 1, 1);
  Eigen::MatrixXd eval_output = Eigen::MatrixXd::Constant(1, 1, 0);
  my_net.train(input, output, 300, 1, 0.15, eval_input, eval_output);
  while (window.isOpen())
  {
    sf::Event event;
    while (window.pollEvent(event))
    {
      if(event.type == sf::Event::Closed)
        window.close();
    }

    // Each 200 ms, a new random value is add to the random curve
    if(clock.getElapsedTime().asMilliseconds() > 1000)
    {
      clock.restart();
    }

    window.clear();

    my_net._chart_mutex.lock();
    window.draw(my_net.unit_test);
    my_net._chart_mutex.unlock();

    window.display();
    sf::sleep(sf::milliseconds(100));
  }
  /*
     MNIST_Parser my_parser;
     Eigen::MatrixXd train_images(0, 0);
     Eigen::MatrixXd train_labels(0, 0);
     Eigen::MatrixXd eval_images(0, 0);
     Eigen::MatrixXd eval_labels(0, 0);
     std::cout << "Reading MNIST data ..." << std::endl;
     my_parser.read_train_img(train_images);
     my_parser.read_train_label(train_labels);
     my_parser.read_eval_img(eval_images);
     my_parser.read_eval_label(eval_labels);
  //std::cout << train_images.col(1) << std::endl; 

  std::vector<unsigned int> layers;
  layers.push_back(784);
  layers.push_back(30);
  layers.push_back(10);
  FNN_Model my_net(layers);
  std::cout << "Training Forward Neural Net ..." << std::endl;
  my_net.train(train_images, train_labels, 30, 10, 3.0, eval_images, eval_labels);
  */
  /*
     std::vector<unsigned int> layers;
     layers.push_back(2);
     layers.push_back(3);
     layers.push_back(1);
     FNN_Model my_net(layers);
     my_net.print_FNN();
     */


  return 0;
}

