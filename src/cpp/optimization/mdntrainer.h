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
        /// Train network for 'epochs' iterations.
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

        ///
        /// Set parameters of attached network.
        ///
        void set_params(const real_1d_array& x);

        ///
        /// Get derivatives of attached network.
        ///
        void get_derivs(real_1d_array& x);
        void get_derivs(double* derivs);

        ///
        /// Return reason for termination of the optimization stage
        ///
        int get_terminationtype();

    protected:
        ///
        /// Network to be optimized
        ///
        MDN& _network;

        ///
        /// Dataset the network is to be optimized upon.
        ///
        SupervisedDataset<double*, double*>& _dataset;

    private:
        minlbfgsstate _lbfgsstate;
        minlbfgsreport _lbfgsrep;
        bool _new_run;
        int _it_count;
        int _report_every;
        int _terminationtype;

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
