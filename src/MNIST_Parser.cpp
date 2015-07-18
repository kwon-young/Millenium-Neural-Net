
#include "MNIST_Parser.hpp"
#include <iostream>
#include <stdio.h>
#include <bitset>

MNIST_Parser::MNIST_Parser (std::string data_directory):
  _data_directory(data_directory),
  _train_img(),
  _train_label(),
  _eval_img(),
  _eval_label()
{
  std::string train_img_f(_data_directory);
  train_img_f += "/train-images.idx3-ubyte";
  _train_img.open(train_img_f.c_str(), std::ios::binary);
  if (!_train_img.good())
  {
    std::cout << "Couldn't open " << train_img_f << std::endl;
    _train_img.close();
  }
  std::string train_label_f(_data_directory);
  train_label_f += "/train-labels.idx1-ubyte";
  _train_label.open(train_label_f.c_str(), std::ios::binary);
  if (!_train_label.good())
  {
    std::cout << "Couldn't open " << train_label_f << std::endl;
    _train_label.close();
  }
  std::string eval_img_f(_data_directory);
  eval_img_f += "/t10k-images.idx3-ubyte";
  _eval_img.open(eval_img_f.c_str(), std::ios::binary);
  if (!_eval_img.good())
  {
    std::cout << "Couldn't open " << eval_img_f << std::endl;
    _eval_img.close();
  }
  std::string eval_label_f(_data_directory);
  eval_label_f += "/t10k-labels.idx1-ubyte";
  _eval_label.open(eval_label_f.c_str(), std::ios::binary);
  if (!_eval_label.good())
  {
    std::cout << "Couldn't open " << eval_label_f << std::endl;
    _eval_label.close();
  }
}

MNIST_Parser::~MNIST_Parser ()
{
  if (_train_img.is_open())
    _train_img.close();
  if (_train_label.is_open())
    _train_label.close();
  if (_eval_img.is_open())
    _eval_img.close();
  if (_eval_label.is_open())
    _eval_label.close();
}

int MNIST_Parser::read_MNIST_format(
    std::ifstream &file,
    Eigen::MatrixXd &storage,
    unsigned int header_size)
{
  if (!file.is_open())
    return -1;
  unsigned char *buffer = new unsigned char [header_size];
  int *header = new int [header_size];
  for (int i = 0; i < header_size; ++i) {
    file.read((char*)buffer, 4);
    //convert big-endian to little-endian
    header[i] = buffer[0] << 24 | buffer[1] << 16 | buffer[2] << 8 | buffer[3] << 0;
    for (int i = 0; i < 4; ++i) {
      std::cout << std::hex << (int)buffer[i] << std::endl;
    }
    if (i==0) std::cout << std::hex;
    else std::cout << std::dec;
    std::cout << header[i] << std::endl;
  }
  if (header_size == 4)
  {
    int img_size = header[2]*header[3];
    storage.resize(img_size, header[1]);
    unsigned char *img = new unsigned char [img_size];
    for (int i = 0; i < header[1]; ++i) {
      file.read((char*)img, img_size);
      for (int j = 0; j < img_size; ++j) {
        storage.col(i).row(j) << (double)(img[j]);
      }
    }
    delete[] img;
  }
  else if (header_size == 2)
  {
    int column_size = 10;
    storage.resize(column_size, header[1]);
    unsigned char *digit = new unsigned char [header[1]];
    std::cout << storage.cols() << std::endl;
    std::cout << storage.rows() << std::endl;
    file.read((char*)digit, header[1]);
    for (int i = 0; i < header[1]; ++i) {
      //std::cout << (int)digit[i] << std::endl;
      storage.col(i).row(digit[i]) << 1.0;
    }
    delete[] digit;
  }

  delete[] buffer;
  delete[] header;
  return 0;
}

int MNIST_Parser::read_train_img (Eigen::MatrixXd &train_images)
{
  return read_MNIST_format(_train_img, train_images, 4);
}

int MNIST_Parser::read_train_label (Eigen::MatrixXd &train_labels)
{
  return read_MNIST_format(_train_label, train_labels, 2);
}

int MNIST_Parser::read_eval_img (Eigen::MatrixXd &eval_images)
{
  return read_MNIST_format(_eval_img, eval_images, 4);
}

int MNIST_Parser::read_eval_label (Eigen::MatrixXd &eval_labels)
{
  return read_MNIST_format(_eval_label, eval_labels, 2);
}

