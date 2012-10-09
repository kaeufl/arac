#include <stdio.h>
#include <math.h>
#include <float.h>

#include "mdn.h"

using namespace std;
using arac::structure::networks::MDN;

MDN::MDN(int M, int c) :
	Network(),
	_M(M), _c(c), _paramsize(_M*(c+2))
{
}

void
MDN::test()
{
    printf("%.6g", DBL_EPSILON);
}

void
MDN::get_output_error(const double* output_p, int outputsize,
						const double* target_p, int targetsize,
		 	 	 	 	double* outputerror_p)
{
	double params_p[_paramsize];
	get_mixture_params(output_p, _paramsize, params_p, _paramsize);
	double pi[_M];
	double tmp = 0.0;
	for (int i=0; i<_M; i++) {
		pi[i] = params_p[i] * phi(target_p, params_p, i);
		tmp += pi[i];
	}
	for (int i=0; i<_M; i++) {
		pi[i] = pi[i] / tmp;
		outputerror_p[i] = params_p[i] - pi[i];
		outputerror_p[_M+i] = - 0.5 * pi[i] * (dist(target_p, params_p, i) / params_p[_M+i] - _c);
		for (int j=0; j<_c; j++) {
			outputerror_p[2*_M+_c*i+j] = pi[i] * (params_p[2*_M+_c*i+j] - target_p[j]) / params_p[_M+i];
		}
	}
}

double
MDN::get_error(const double* output_p, int outputsize,
			    const double* target_p, int targetsize)
{
	double params_p[_paramsize];
	get_mixture_params(output_p, outputsize, params_p, _paramsize);
	double error = 0.0;
	for (int i=0; i<_M; i++) {
		error += params_p[i]*phi(target_p, params_p, i);
	}
	error = error < DBL_EPSILON ? DBL_EPSILON : error;
	return -log(error);
}

void
MDN::get_mixture_params(const double* output_p, int outsize,
						  double* params_p, int paramsize)
{
	// apply softmax to mixing coefficients
	double tmp[_M];
	double alpha[_M];
	copy(output_p, output_p+_M, tmp);
	softmax(tmp, alpha, _M);
	copy(alpha, alpha+_M, params_p);

	// apply exponent to scale outputs
	double sigma;
	for (int i=_M; i<2*_M; i++) {
		sigma = output_p[i] > 500 ? 500 : output_p[i];
		sigma = exp(sigma);
		params_p[i] = sigma < DBL_EPSILON ? DBL_EPSILON : sigma;
	}

	// copy means
	copy(output_p+2*_M, output_p+paramsize, params_p+2*_M);
}

void
MDN::softmax(const double* x_p, double* y_p, int len)
{
	double sum = 0;

	for (int i = 0; i < len; i++) {
		// Clip input argument if its to extreme to avoid NaNs and inf as a
		// result of exp().
		double inpt;
		inpt = x_p[i] < -500 ? -500 : x_p[i];
		inpt = inpt > 500 ? 500 : inpt;
		double item = exp(inpt);

		sum += item;
		y_p[i] = item;
	}
	for(int i = 0; i < len; i++)
	{
		y_p[i] /= sum;
	}
}

double
MDN::phi(const double* target_p, const double* params_p, int m)
{
	double phi, tmp;
	tmp = dist(target_p, params_p, m);
	phi = exp(-tmp / (2*params_p[_M+m]));
	phi = phi / pow(2*M_PI*params_p[_M+m], 0.5*_c);
	phi = phi < DBL_EPSILON ? DBL_EPSILON : phi;
	return phi;
}

double
MDN::dist(const double* target_p, const double* params_p, int m)
{
	double dist = 0.0;
	for (int j=0; j<_c; j++) {
		dist += pow(target_p[j]-params_p[2*_M+_c*m+j], 2);
	}
	return dist;
}
