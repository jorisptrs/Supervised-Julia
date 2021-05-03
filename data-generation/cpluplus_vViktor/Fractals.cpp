#include <windows.h>
#include <string>
#include <complex>
#include <ctime>  
#include <chrono>
#include <stdlib.h>
#include <stdio.h>
#include <iostream>
#include <string>
#include <math.h>
#include <fstream>

#include "Julia.h"

const int BMP_SIZE = 2000;
const int ITERATIONS = 1024;
const long double xmin = -2, xmax = 2;
const long double ymin = -2, ymax = 2;
const double ESCAPE_THRESHOLD = 3.0;

int main(int argc, char* argv[]) {
    std::complex<long double> c;

    long double cReal, cImg;

    while (true) {
        std::cout << "Real: ";
        std::cin >> cReal;
        std::cout << "Img: ";
        std::cin >> cImg;

        c.imag(cImg);
        c.real(cReal);
        auto start = std::chrono::system_clock::now();
        julia j(BMP_SIZE, ITERATIONS, xmin, xmax, ymin, ymax, ESCAPE_THRESHOLD); 
        j.draw(c);
        auto end = std::chrono::system_clock::now();
        std::chrono::duration<double> elapsed_seconds = end - start;
        printf("%f\n", elapsed_seconds.count());
    }
    return 0;
}