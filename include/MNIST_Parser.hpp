#ifndef MNIST_PARSER_HPP
#define MNIST_PARSER_HPP

#include <string>
#include <fstream>
#include <Eigen/Dense>

class MNIST_Parser
{
public:
  MNIST_Parser (std::string data_directory = std::string("c:/Users/Kwon-Young/Documents/Programmation/Millenium-Neural-Net/data"));
  ~MNIST_Parser ();

  void read_MNIST_format(
      std::ifstream &file,
      Eigen::MatrixXd &storage,
      unsigned int header_size);
  void read_train_img (Eigen::MatrixXd &train_images);
  void read_train_label (Eigen::MatrixXd &train_labels);
  void read_eval_img (Eigen::MatrixXd &eval_images);
  void read_eval_label (Eigen::MatrixXd &eval_labels);
private:
  std::string _data_directory;
  std::ifstream _train_img;
  std::ifstream _train_label;
  std::ifstream _eval_img;
  std::ifstream _eval_label;
};

#endif
