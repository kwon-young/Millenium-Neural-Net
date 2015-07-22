#ifndef MNIST_PARSER_HPP
#define MNIST_PARSER_HPP

#include <string>
#include <fstream>
#include <Eigen/Dense>

class MNIST_Parser
{
public:
  MNIST_Parser (std::string data_directory = std::string("c:/Users/Kwon-Young/Documents/Prog/Millenium-Neural-Net/data"));
  ~MNIST_Parser ();

  int read_MNIST_format(
      std::ifstream &file,
      Eigen::MatrixXd &storage,
      unsigned int header_size);
  int read_train_img (Eigen::MatrixXd &train_images);
  int read_train_label (Eigen::MatrixXd &train_labels);
  int read_eval_img (Eigen::MatrixXd &eval_images);
  int read_eval_label (Eigen::MatrixXd &eval_labels);
private:
  std::string _data_directory;
  std::ifstream _train_img;
  std::ifstream _train_label;
  std::ifstream _eval_img;
  std::ifstream _eval_label;
};

#endif
