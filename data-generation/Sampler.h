#pragma once

#include <math.h>
#include <complex>
#include <random>

#define M_PI 3.14159265358979323846
#define min(a,b) ((a)<(b)?(a):(b))

class Sampler {
public:

	enum Type {
		REPRESENTATIVE, RANDOM
	};

	int N, n;
	double rStep, minDistance;

	int iCircle, nAchieved, nDesired;
	double alpha;

	double noiseStrength;

	int type;

	std::mt19937 * generatorEngine;


	Sampler(int nData, double radiusStep, double minDistanceOnCircle, double noiseMagnitude) {
		init(nData, REPRESENTATIVE);
		rStep = radiusStep;
		minDistance = minDistanceOnCircle;
		iCircle = 0;
		noiseStrength = noiseMagnitude;
		nextCircle();
	}

	Sampler(int nData, double samplingRadius) {
		init(nData, RANDOM);
		rStep = samplingRadius;
	}

	void init(int nData, int type) {
		this->N = nData;
		this->n = 0;
		this->type = type;
		std::random_device randomEngine;
		generatorEngine = new std::mt19937(randomEngine());
	}

	~Sampler() { }

	bool hasNext() {
		return n < N;
	}

	std::complex<long double> next() {
		std::complex<long double> c;
		switch (type)
		{
		case REPRESENTATIVE:
			c = nextCircular();
			break;
		case RANDOM:
			c = nextUniform();
			break;
		default:
			printf("Bad sampler type.\n");
			c.imag(nan(""));
			c.real(nan(""));
			break;
		}
		n++;
		return c;
	}

private:

	std::complex<long double> nextUniform() {
		std::complex<long double> constant;
		double real, imag;
		double alpha, r, u;

		alpha = randNum(2 * M_PI);
		u = randNum(rStep) + randNum(rStep);
		r = u > rStep ? (2 * rStep) - u : u;
		
		real = cos(alpha) * r;
		imag = sin(alpha) * r;

		constant.real(real);
		constant.imag(imag);
		return constant;
	}

	std::complex<long double> nextCircular() {
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
		return constant;
	}

	double currentR() {
		return iCircle * rStep;
	}

	double randNum(double upperBound) {
		return randNum(0.0, upperBound);
	}

	double randNum(double lowerBound, double upperBound) {
		std::uniform_real_distribution<double> unif(lowerBound, upperBound);
		return unif(*generatorEngine);
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