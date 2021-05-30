#pragma once

// https://rosettacode.org/wiki/Julia_set#C.2B.2B Code

#include <string>

#include "Bitmap.h"


class julia {
public:
    int pixels, iterations;

    long double xmin, xmax, xrange;
    long double ymin, ymax, yrange;

    double escapeDistance;

    julia(int pictureSize, int maxIterations, double xMin, double xMax, double yMin, double yMax, double escapeThreshold) {
        pixels = pictureSize;
        iterations = maxIterations;
        xmin = xMin;
        xmax = xMax;
        ymin = yMin;
        ymax = yMax;
        xrange = xmax - xmin;
        yrange = ymax - ymin;
        escapeDistance = escapeThreshold;
    }

    std::string draw(std::complex<long double> k, bool isPicture=true) {
        DWORD* bits = nullptr;
        std::string output = "";

        if (isPicture) {
            bmp.create(pixels, pixels);
            bits = bmp.bits();
        }
      
        int res, pos, biggest = -1;
        std::complex<long double> c;

        for (int y = 0; y < pixels; y++) {
            pos = y * pixels;
            for (int x = 0; x < pixels; x++) {
                c.imag((double)y / (double)pixels * yrange + ymin);
                c.real((double)x / (double)pixels * xrange + xmin);
                res = inSet(c, k);
                if (biggest < res) {
                    biggest = res;
                }
                if (res > 0) {
                    if (isPicture) {
                        int n_res = mapRound(res, 0, iterations, 0, 255);
                        res = RGB(0, n_res, 0);
                    }
                }
                if (isPicture) {
                    bits[pos++] = res;
                }
                else {
                    output += (x == 0 ? "" : ",") + std::to_string(res);
                }
            }
            output += "\n";
        }
        if (isPicture) {
            bmp.saveBitmap("./js.bmp");
        }
        printf("Biggest %d\n", biggest);
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
        return 0;
    }
    myBitmap bmp;
};