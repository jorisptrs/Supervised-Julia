#pragma once

#include <math.h>
#include <complex>
#include <random>

#define M_PI 3.14159265358979323846

class Sampler {
public:

	int N, n;
	double rStep, minDistance;

	int iCircle, nAchieved, nDesired;
	double alpha;

	double noiseStrength;

	Sampler(int nData, double radiusStep, double minDistanceOnCircle, double noiseMagnitude) {
		N = nData;
		rStep = radiusStep;
		minDistance = minDistanceOnCircle;
		n = iCircle = 0;
		noiseStrength = noiseMagnitude;
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

		long double real = cos(alpha * (double)nAchieved) * currentR();
		long double imag = sin(alpha * (double)nAchieved) * currentR();

		double beta = randNum(M_PI * 2.0);
		double noiseMagnitude = randNum(noiseStrength);

		real += cos(beta) * noiseMagnitude;
		imag += sin(beta) * noiseMagnitude;

		constant.real(real);
		constant.imag(imag);

		nAchieved++;
		n++;
		return constant;
	}

private:

	double currentR() {
		return iCircle * rStep;
	}

	double randNum(double upperBound) {
		return randNum(0.0, upperBound);
	}

	double randNum(double lowerBound, double upperBound) {
		std::uniform_real_distribution<double> unif(lowerBound, upperBound);
		std::default_random_engine re;
		return unif(re);
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