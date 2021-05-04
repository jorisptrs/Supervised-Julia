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
#include "Sampler.h"

const int BMP_SIZE = 2000;
const int ITERATIONS = 512;
const long double xmin = -2, xmax = 2;
const long double ymin = -2, ymax = 2;
const double ESCAPE_THRESHOLD = 3.0;

const bool INTERACTIVE = false;

/*
* Data generation
*/
const int nData = 5;
const double rStep = 0.5; // Radius step
const double minDistanceOnCircle = 0.5; // For ideal space representation this should be close to rStep


void interactive() {
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
}


void generateData() {
    Sampler * sampler = new Sampler(rStep, minDistanceOnCircle, nData);
    int i = 0;

    std::string header = "SIZE=" + std::to_string(BMP_SIZE) + "\n"
        + "ITERATIONS=" + std::to_string(ITERATIONS) + "\n"
        + "XMIN=" + std::to_string(xmin) + "\n"
        + "XMAX=" + std::to_string(xmax) + "\n"
        + "YMIN=" + std::to_string(ymin) + "\n"
        + "YMAX=" + std::to_string(ymax) + "\n"
        + "ESCAPE_THRESHOLD=" + std::to_string(ESCAPE_THRESHOLD) + "\n"
        + "RSTEP=" + std::to_string(rStep) + "\n"
        + "MIN_DST_ON_CIRCLE=" + std::to_string(minDistanceOnCircle) + "\n";

    while (sampler->hasNext()) {
        std::complex<long double> c = sampler->next();
        std::string data = "";

        data += header;

        julia j(BMP_SIZE, ITERATIONS, xmin, xmax, ymin, ymax, ESCAPE_THRESHOLD);
        data += j.draw(c, false);

        std::ofstream dataFile("./trainingData/data" +  std::to_string(i) + ".jset");
        dataFile << data;
        dataFile.close();
        i++;
        printf("%d / %d\n", i, nData);
    }

    delete sampler;
}

int main(int argc, char* argv[]) {
    if (INTERACTIVE) {
        interactive();
    }
    else {
        generateData();
    }
    return 0;
}