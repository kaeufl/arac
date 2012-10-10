// added Sept 2012 by Paul Kaeufl, <p.j.kaufl@uu.nl>
// provides a trainer for mixture density networks

#ifndef Arac_OPTIMIZER_MDNTRAINER_INCLUDED
#define Arac_OPTIMIZER_MDNTRAINER_INCLUDED

#include <signal.h>
#include "../structure/networks/mdn.h"
#include "../datasets/datasets.h"
#include "../utilities/alglib/optimization.h"

using arac::structure::networks::MDN;
using arac::datasets::SupervisedDataset;
using namespace alglib;

namespace arac {
namespace optimization {

class MDNTrainerAbortError : public std::exception {
//  public:
//    MDNTrainerAbortError() : std::exception() {}
};

class MDNTrainer 
{
    public:
        int _n_params;

        MDNTrainer(MDN& network,
                    SupervisedDataset<double*, double*>& dataset);
        MDNTrainer(MDN& network,
					SupervisedDataset<double*, double*>& dataset,
					SupervisedDataset<double*, double*>& testset);
        ~MDNTrainer();

        ///
        /// Return a reference to the network of the optimizer.
        ///
        MDN& network();
        
        ///
        /// Return a reference to the dataset of the optimizer.
        ///
        SupervisedDataset<double*, double*>& dataset();

        ///
		/// Return a reference to the (optional) test set.
		///
		SupervisedDataset<double*, double*>& testset();

        ///
        /// Train network for 'epochs' iterations.
		/// Note: if a test has been provided during object initialization,
		/// the attached module parameters are set to the optimal parameters with
		/// respect to the test set minimum.
        ///
        int train(int epochs);
        int train();

        ///
        /// Return total error for dataset.
        ///
        void get_total_error(double &err);

        ///
        /// Get parameters of attached network.
        ///
        void get_params(real_1d_array& x);
        void get_params(std::vector<double>& x);

        ///
        /// Set parameters of attached network.
        ///
        void set_params(const real_1d_array& x);
        void set_params(const std::vector<double>& x);

        ///
        /// Get derivatives of attached network.
        ///
        void get_derivs(real_1d_array& x);
        void get_derivs(double* derivs);

        ///
        /// Return reason for termination of the optimization stage
        ///
        int get_terminationtype();

        ///
        /// Return the error (test set error) trace for all epochs performed so far.
        ///
        std::vector<double> getErrorTrace();
        std::vector<double> getTestErrorTrace();

        ///
		/// Return the set of parameters for which the minimum test set error occured
		///
        std::vector<double> getOptimalParams();

        ///
		/// Return the epoch number at which the minimum test set error occured
		///
        int getOptimalEpoch();


    protected:
        ///
        /// Network to be optimized
        ///
        MDN& _network;

        ///
        /// Dataset the network is to be optimized upon.
        ///
        SupervisedDataset<double*, double*>* _dataset;

        ///
        /// Test set (optional)
        ///
        SupervisedDataset<double*, double*>* _testset;

    private:
        void initTrainer();

        minlbfgsstate _lbfgsstate;
        minlbfgsreport _lbfgsrep;
        bool _new_run;
        int _it_count;
        int _report_every;
        int _terminationtype;
        std::vector<double> _errors;
        std::vector<double> _test_errors;
        std::vector<double> _optimal_x;
        int _optimal_it;
        double _optimal_err;

        static const double _epsg = 0.0000000001;
        static const double _epsf = 0;
        static const double _epsx = 0;

        static void f_df(const real_1d_array &x, double &func, real_1d_array &grad, void *ptr);
        static void report(const real_1d_array &x, double func, void *ptr);

        static void abort_handler(int s);
};

}
}

#endif
