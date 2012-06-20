#ifndef Arac_STRUCTURE_NETWORKS_MDN_INCLUDED
#define Arac_STRUCTURE_NETWORKS_MDN_INCLUDED

#include "network.h"

namespace arac {
namespace structure {
namespace networks {


class MDN : public Network
{
public:
	///
	/// Create a new MDN object.
	///
	MDN(int M, int c);

	void test();

	///
	/// Return error for the given network output and target vector.
	///
	double get_error(const double* output_p, int outputsize,
		    		   const double* target_p, int targetsize);

	///
	/// Get mixture coefficients, means and standard deviations for given
	/// network output.
	///
	void get_mixture_params(const double* output_p, int outsize,
			  	  	  	  	   double* params_p, int paramsize);

	///
	/// Apply softmax function to given input vector.
	///
	void softmax(const double* x_p, double* y_p, int len);

	///
	/// Get the output error for every output node for given network output
	/// and target vector.
	///
	void get_output_error(const double* output_p, int outputsize,
							 const double* target_p, int targetsize,
							 double* outputerror_p);

protected:
	int _M;
	int _c;
	int _paramsize;

	double phi(const double* target_p, const double* params_p, int m);
	double dist(const double* target_p, const double* params_p, int m);
};


}
}
}

#endif
