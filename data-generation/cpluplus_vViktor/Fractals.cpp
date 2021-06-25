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
#include <vector>

#include "Julia.h"
#include "Sampler.h"

const int BMP_WIDTH = 128;
const int BMP_HEIGHT = 64;
const int ITERATIONS = 256;
const long double xmin = -1.5, xmax = 1.5;
const long double ymin = 0, ymax = 1.5;
const double ESCAPE_THRESHOLD = 3.0;

/*
* Data generation
*/
const int nData = 2000;
const std::string type = "RANDOM";
/*
* REPRESENTATIVE data generation
*/
const double sampleRadius = 1;

/*
* RANDOM data generation
*/
const double rStep = 10; // Radius step
const double minDistanceOnCircle = 4; // For ideal space representation this should be close to rStep
const double noiseMagnitude = 0.1; // Making this higher than rStep or minDst can lead to a data duplication

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
        julia j(BMP_WIDTH, BMP_HEIGHT, ITERATIONS, xmin, xmax, ymin, ymax, ESCAPE_THRESHOLD);
        j.draw(c);
        auto end = std::chrono::system_clock::now();
        std::chrono::duration<double> elapsed_seconds = end - start;
        printf("%f\n", elapsed_seconds.count());
    }
}


void generateData() {
    Sampler* sampler = nullptr;
    std::vector<std::vector<double>> labels(nData, std::vector<double>(2, 0));

    if (type == "REPRESENTATIVE") {
        sampler = new Sampler(nData, rStep, minDistanceOnCircle, noiseMagnitude);
    }
    else if (type == "RANDOM") {
        sampler = new Sampler(nData, sampleRadius);
    }
    int i = 0;


    std::string header = "PICTURE_SIZE=" + std::to_string(BMP_WIDTH) + "," + std::to_string(BMP_HEIGHT) + "\n"
        + "N_DATA=" + std::to_string(nData) + "\n"
        + "SAMPLING_TYPE=" + type + "\n"
        + "ITERATIONS=" + std::to_string(ITERATIONS) + "\n"
        + "XMIN=" + std::to_string(xmin) + "\n"
        + "XMAX=" + std::to_string(xmax) + "\n"
        + "YMIN=" + std::to_string(ymin) + "\n"
        + "YMAX=" + std::to_string(ymax) + "\n"
        + "ESCAPE_THRESHOLD=" + std::to_string(ESCAPE_THRESHOLD) + "\n"
        + "[REPRESENTATIVE]\n"
        + "RSTEP=" + std::to_string(rStep) + "\n"
        + "MIN_DST_ON_CIRCLE=" + std::to_string(minDistanceOnCircle) + "\n"
        + "NOISE_MAGNITUDE=" + std::to_string(noiseMagnitude) + "\n"
        + "[RANDOM]\n"
        + "SAMPLE_RADIUS=" + std::to_string(sampleRadius) + "\n";

    std::ofstream headerFile("./trainingData/header.txt");
    headerFile << header;
    headerFile.close();

    while (sampler->hasNext()) {
        std::complex<long double> c = sampler->next();

        labels[i][0] = c.real();
        labels[i][1] = c.imag();

        julia j(BMP_WIDTH, BMP_HEIGHT, ITERATIONS, xmin, xmax, ymin, ymax, ESCAPE_THRESHOLD);
        std::string data;

        data = j.draw(c);

        std::ofstream dataFile("./trainingData/data" +  std::to_string(i) + ".jset");
        dataFile << data;
        dataFile.close();
        i++;
        if (i % 1000 == 999) {
            printf("%d / %d\n", i, nData);
        }
    }

    std::ofstream labelsFile("./trainingData/labels.txt");
    for (const auto& label : labels) {
        labelsFile << std::to_string(label[0])  << "," << std::to_string(label[1]) << "\n";
    }

    delete sampler;
}

int main(int argc, char* argv[]) {

    generateData();
    return 0;
}