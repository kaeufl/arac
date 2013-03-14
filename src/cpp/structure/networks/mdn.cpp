#include <stdio.h>
#include <math.h>
#include <float.h>
#include <stdlib.h>

#include "mdn.h"

#define TWOPI 2*M_PI

using namespace std;
using arac::structure::networks::MDN;
using arac::structure::networks::PeriodicMDN;

MDN::MDN(int M, int c) :
	Network(),
	_M(M), _c(c), _paramsize(_M*(c+2))
{
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
		sigma = output_p[i] > 5000 ? 5000 : output_p[i];
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
		inpt = x_p[i] < -5000 ? -5000 : x_p[i];
		inpt = inpt > 5000 ? 5000 : inpt;
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
	phi = phi / pow(TWOPI*params_p[_M+m], 0.5*_c);
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

PeriodicMDN::PeriodicMDN(int M, int c) :
    MDN(M, c),
    _nperiods(7)
{
}

void
PeriodicMDN::get_output_error(const double* output_p, int outputsize,
                        const double* target_p, int targetsize,
                        double* outputerror_p)
{
    double params_p[_paramsize];
    double pi[_M];
    double Phi[_M][2*_nperiods+1];
    double pitot = 0.0;
    double chi[2*_nperiods+1][targetsize];

    get_mixture_params(output_p, _paramsize, params_p, _paramsize);

    for (int l=0; l<=2*_nperiods; l++) {
        for (int i=0; i<targetsize; i++) {
            chi[l][i] = target_p[i] + (l-_nperiods)*TWOPI;
        }
    }

    for (int i=0; i<_M; i++) {
        pi[i] = 0.0;
        for (int l=0; l<=2*_nperiods; l++) {
            Phi[i][l] = phi(chi[l], params_p, i);
            pi[i] += params_p[i] * Phi[i][l];
        }
        pitot += pi[i];
    }

//    printf("Phi:\n");
//    for (int i=0; i < _M; i++) {
//        for (int l = 0; l <= 2*_nperiods; l++) {
//            printf("%g\t", Phi[i][l]);
//        }
//        printf("\n");
//    }

    for (int i=0; i<_M; i++) {
        outputerror_p[i] = params_p[i] - (pi[i] / pitot);
        outputerror_p[_M+i] = 0.0;
        for (int l=0; l<=2*_nperiods; l++) {
            outputerror_p[_M+i] += Phi[i][l] * (dist(chi[l], params_p, i) / params_p[_M+i] - _c);
        }
        outputerror_p[_M+i] *= -0.5 * params_p[i] / pitot;
        for (int j=0; j<_c; j++) {
            outputerror_p[2*_M+_c*i+j] = 0.0;
            for (int l=0; l<=2*_nperiods; l++) {
                outputerror_p[2*_M+_c*i+j] += Phi[i][l]*((params_p[2*_M+_c*i+j] - chi[l][j]) / params_p[_M+i]);
            }
            outputerror_p[2*_M+_c*i+j] *= params_p[i] / pitot;
        }
    }
}

double
PeriodicMDN::get_error(const double* output_p, int outputsize,
                const double* target_p, int targetsize)
{
    double params_p[_paramsize];
    double error = 0.0;
    double chi[targetsize];

    get_mixture_params(output_p, outputsize, params_p, _paramsize);

    for (int l=0; l<=2*_nperiods; l++) {
        for (int j=0; j<targetsize; j++) {
            chi[j] = target_p[j] + (l-_nperiods)*TWOPI;
        }
        for (int i=0; i<_M; i++) {
            error += params_p[i]*phi(chi, params_p, i);
        }
    }
    error = error < DBL_EPSILON ? DBL_EPSILON : error;
    //printf("%f", error);
    return -log(error);
}
