#pragma once

// https://rosettacode.org/wiki/Julia_set#C.2B.2B Code

#include <string>


class julia {
public:
    int width, height, iterations;

    long double xmin, xmax, xrange;
    long double ymin, ymax, yrange;

    double escapeDistance;

    julia(int width, int height, int maxIterations, double xMin, double xMax, double yMin, double yMax, double escapeThreshold) {
        this->width = width;
        this->height = height;
        iterations = maxIterations;
        xmin = xMin;
        xmax = xMax;
        ymin = yMin;
        ymax = yMax;
        xrange = xmax - xmin;
        yrange = ymax - ymin;
        escapeDistance = escapeThreshold;
    }

    std::string draw(std::complex<long double> k) {
        std::string output = "";

        int res, pos, biggest = -1;
        std::complex<long double> c;

        for (int y = 0; y < height; y++) {
            pos = y * height;
            for (int x = 0; x < width; x++) {
                c.imag((double)y / (double)height * yrange + ymin);
                c.real((double)x / (double)width * xrange + xmin);
                res = inSet(c, k);
                if (biggest < res) {
                    biggest = res;
                }
                output += (x == 0 ? "" : ",") + std::to_string(res);
               
            }
            output += "\n";
        }
        // printf("Biggest %d\n", biggest);
        return output;
    }
private:
    int mapRound(int value, int min, int max, int newMin, int newMax) {
        double range = (double)(max - min);
        double newRange = (double)(newMax - newMin);
        double newVal = ((double)(value - min) / range) * newRange;
        return round(newVal + (double)newMin);
    }
    int inSet(std::complex<long double> z, std::complex<long double> c) {
        long double dist;
        for (int ec = 0; ec < iterations; ec++) {
            z = z * z; z = z + c;
            dist = (z.imag() * z.imag()) + (z.real() * z.real());
            if (dist > escapeDistance) return(ec);
        }
        return iterations;
    }
};