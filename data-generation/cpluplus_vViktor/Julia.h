#pragma once

#include "Bitmap.h"


class julia {
public:
    int pixels, iterations;

    long double xmin, xmax, xrange;
    long double ymin, ymax, yrange;

    double escapeDistance;

    julia(int pictureSize, int maxIterations, int xMin, int xMax, int yMin, int yMax, double escapeThreshold) {
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

    void draw(std::complex<long double> k) {
        bmp.create(pixels, pixels);
        DWORD* bits = bmp.bits();
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
                    //int n_res = res % 255;
                    int n_res = mapRound(res, 0, iterations, 0, 255);
                    //if (res < (iterations >> 1)) res = RGB(n_res << 2, n_res << 3, n_res << 4);
                    //else res = RGB(n_res << 4, n_res << 2, n_res << 5);
                    res = RGB(0, n_res, 0);
                }
                bits[pos++] = res;
            }
        }
        printf("Biggest %d\n", biggest);
        bmp.saveBitmap("./js.bmp");
    }
private:
    int mapRound(int value, int min, int max, int newMin, int newMax) {
        double range = (double)(max - min);
        double newRange = (double)(newMax - newMin);
        double newVal = ((double)(value - min) / range) * newRange;
        return round(newVal + (double)newMin);
    }
    int inSet(std::complex<long double> z, std::complex<long double> c) {
        long double dist;//, three = 3.0;
        for (int ec = 0; ec < iterations; ec++) {
            z = z * z; z = z + c;
            dist = (z.imag() * z.imag()) + (z.real() * z.real());
            if (dist > escapeDistance) return(ec);
        }
        return 0;
    }
    myBitmap bmp;
};