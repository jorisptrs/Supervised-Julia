#pragma once

#include <math.h>
#include <complex>

#define M_PI 3.14159265358979323846


class Sampler {
public:

	int N, n;
	double rStep, minDistance;

	int iCircle, nAchieved, nDesired;
	double alpha;

	Sampler(double radiusStep, double minDistanceOnCircle, int nData) {
		N = nData;
		rStep = radiusStep;
		minDistance = minDistanceOnCircle;
		n = iCircle = 0;
		nextCircle();
	}

	~Sampler() { }

	bool hasNext() {
		return n < N;
	}

	std::complex<long double> next() {
		std::complex<long double> constant;

		if (isNextCircle()) {
			nextCircle();
		}

		constant.real(cos(alpha * (double)nAchieved) * currentR());
		constant.imag(sin(alpha * (double)nAchieved) * currentR());

		nAchieved++;
		n++;
		return constant;
	}

private:

	double currentR() {
		return iCircle * rStep;
	}

	double angle() {
		double value = 1.0 - (minDistance * minDistance) / (2 * currentR() * currentR());
		if (value < -1 || value > 1) {
			printf("Invalid minDistance and rStep values. Most probably minDistance is too high for a given rStep. Exiting ..");
			exit(-1);
		}
		
		return acos(value);
	}

	bool isNextCircle() {
		return nAchieved >= nDesired;
	}

	void nextCircle() {
		nAchieved = 0;
		iCircle++;
		alpha = angle();
		nDesired = (int)floor(2.0 * M_PI / alpha);
		nDesired = min(nDesired, N - n);
		alpha = 2.0 * M_PI / (double)nDesired;
	}
};