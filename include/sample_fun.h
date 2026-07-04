#ifndef SAMPLE_FUN_H
#define SAMPLE_FUN_H

#include <vector>

// Simple example function: returns x^2
int square(int x);

// Example function using <vector>
int sum(const std::vector<int>& v);

// Example small struct to test include visibility
struct Point {
    double x, y;
    Point(double x_=0, double y_=0) : x(x_), y(y_) {}
};

#endif // SAMPLE_FUN_H