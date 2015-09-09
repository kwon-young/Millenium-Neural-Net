#include "chart.hpp"

Chart::Chart(const std::string& name, const sf::Vector2i& location) :
  name_(name)
{
  plot_.setSize(sf::Vector2f(600, 400));
  plot_.setTitle("Chart "+name);
  plot_.setFont("c:/Users/Kwon-Young/Documents/Prog/Millenium-Neural-Net/data/font.ttf");
  plot_.setXLabel("epoch");
  plot_.setYLabel("Cost");
  plot_.setBackgroundColor(sf::Color(rand()%255, rand()%255, rand()%255));
  plot_.setTitleColor(sf::Color::Black);
  plot_.setPosition(sf::Vector2f(600*location.x, 400*location.y));
  sf::plot::Curve &curve = plot_.createCurve("chart", sf::Color::Red);
  curve.setFill(rand() % 2);
  curve.setThickness(2 + rand() % 10);
  curve.setColor(sf::Color(rand()%255, rand()%255, rand()%255));
  curve.setLimit(10 + rand() % 100);
}

void Chart::update(double value)
{
  sf::plot::Curve &curve = plot_.getCurve("chart");
  curve.addValue(value);
  plot_.prepare();
}

void Chart::draw(sf::RenderTarget& target, sf::RenderStates states) const
{
  target.draw(plot_, states);
}