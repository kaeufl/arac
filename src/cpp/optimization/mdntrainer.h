// added Sept 2012 by Paul Kaeufl, <p.j.kaufl@uu.nl>
// provides a trainer for mixture density networks

#ifndef Arac_OPTIMIZER_MDNTRAINER_INCLUDED
#define Arac_OPTIMIZER_MDNTRAINER_INCLUDED

#include <signal.h>
#include <algorithm>
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

template<typename NetworkType>
class MDNTrainer 
{
    public:
        int _n_params;
        std::vector<int> _idxs;
        int _batch_size;
        int _it_count;

        MDNTrainer(NetworkType& network,
                    SupervisedDataset<double*, double*>& dataset,
                    bool use_cg,
                    int batch_size,
                    int batch_epochs);
        MDNTrainer(NetworkType& network,
		   SupervisedDataset<double*, double*>& dataset,
		   SupervisedDataset<double*, double*>& validationset,
		   bool use_cg,
		   int batch_size,
           int batch_epochs);
        ~MDNTrainer();

        ///
        /// Return a reference to the network of the optimizer.
        ///
        NetworkType& network();
        
        ///
        /// Return a reference to the dataset of the optimizer.
        ///
        SupervisedDataset<double*, double*>& dataset();

        ///
        /// Return a reference to the (optional) validation set.
        ///
        SupervisedDataset<double*, double*>& validationset();

        ///
        /// Train network for 'epochs' iterations.
	/// Note: if a validation set has been provided during object initialization,
	/// the attached module parameters are set to the optimal parameters with
	/// respect to the validation set minimum.
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
        /// Return the error (validation set error) trace for all epochs performed so far.
        ///
        std::vector<double> getErrorTrace();
        std::vector<double> getValidationErrorTrace();

        ///
		/// Return the set of parameters for which the minimum validation set error occured
		///
        std::vector<double> getOptimalParams();

        ///
		/// Return the epoch number at which the minimum validation set error occured
		///
        int getOptimalEpoch();


    protected:
        ///
        /// Network to be optimized
        ///
        NetworkType& _network;

        ///
        /// Dataset the network is to be optimized upon.
        ///
        SupervisedDataset<double*, double*>* _dataset;

        ///
        /// Validation set (optional)
        ///
        SupervisedDataset<double*, double*>* _validationset;

    private:
        void initTrainer();

        minlbfgsstate _lbfgsstate;
        minlbfgsreport _lbfgsrep;
        mincgstate _cgstate;
        mincgreport _cgrep;
        bool _new_run;
        int _report_every;
        int _terminationtype;
        std::vector<double> _errors;
        std::vector<double> _validation_errors;
        std::vector<double> _optimal_x;
        int _optimal_it;
        double _optimal_err;
        int _batch_epochs;

        bool use_cg;

        //static const double _epsg = 0.0000000001;
        static const double _epsg = 0.0;
        static const double _epsf = 0.0;
        static const double _epsx = 0.0;

        static void f_df(const real_1d_array &x, double &func, real_1d_array &grad, void *ptr);
        static void report(const real_1d_array &x, double func, void *ptr);

        static void abort_handler(int s);
        //void restore_abort_handler(struct sigaction sigAction);
};

}
}

#include <signal.h>
#include <iostream>
#include <omp.h>
#include "../utilities/alglib/stdafx.h"
#include "../utilities/alglib/optimization.h"

using arac::optimization::MDNTrainer;
using arac::optimization::MDNTrainerAbortError;
using arac::structure::networks::MDN;
using arac::structure::Parametrized;
using namespace alglib;

template<typename NetworkType>
MDNTrainer<NetworkType>::MDNTrainer(NetworkType& network,
                         SupervisedDataset<double*, double*>& dataset,
                         bool use_cg,
                         int batch_size,
                         int batch_epochs) :
    _network(network),
    _dataset(&dataset),
    _validationset(0),
    _errors(),
    _validation_errors(),
    _optimal_x(),
    _optimal_it(0),
    _optimal_err(0),
    use_cg(use_cg),
    _batch_size(batch_size),
    _terminationtype(0),
    _batch_epochs(batch_epochs)
{
    initTrainer();
}

template<typename NetworkType>
MDNTrainer<NetworkType>::MDNTrainer(NetworkType& network,
						  SupervisedDataset<double*, double*>& dataset,
						  SupervisedDataset<double*, double*>& validationset,
						  bool use_cg,
						  int batch_size,
                          int batch_epochs) :
	_network(network),
	_dataset(&dataset),
	_validationset(&validationset),
    _errors(),
    _validation_errors(),
    _optimal_x(),
	_optimal_it(0),
	_optimal_err(0),
	use_cg(use_cg),
	_batch_size(batch_size),
	_terminationtype(0),
	_batch_epochs(batch_epochs)
{
	initTrainer();
}

template<typename NetworkType>
MDNTrainer<NetworkType>::~MDNTrainer() {}

template<typename NetworkType>
void MDNTrainer<NetworkType>::initTrainer()
{
	_n_params = 0;
	_it_count = 0;
	real_1d_array x;

	std::vector<Parametrized*>::iterator param_iter;
	for (param_iter = network().parametrizeds().begin();
		param_iter != network().parametrizeds().end();
		param_iter++)
	{
	   _n_params += (*param_iter)->size();
	}

	_optimal_x.resize(_n_params);

	x.setlength(_n_params);
	get_params(x);

	try
	{
	   if (use_cg == true) {
	     std::cout << "Initializing cg trainer" << std::endl;
	     mincgcreate(_n_params, x,_cgstate);
	     mincgsetxrep(_cgstate, true);
	   } else {
         std::cout << "Initializing lbfgs trainer" << std::endl;
         minlbfgscreate(_n_params, 5, x, _lbfgsstate);
         minlbfgssetxrep(_lbfgsstate, true);
	   }
	}
	catch (ap_error& e)
	{
	   std::cout << e.msg << std::endl;
	}

	// initialize the index vector, this is needed to generate random index sequences later
	_idxs.reserve(dataset().size());
	for (int i=0; i<dataset().size(); ++i) _idxs.push_back(i);
	if (_batch_size == 0) {
	    _batch_size = dataset().size();
	}
	std::cout << "Batch size is " << _batch_size << std::endl;
}

//template<typename NetworkType>
//void MDNTrainer<NetworkType>::restore_abort_handler(struct sigaction sigAction)
//{
//
//}

template<typename NetworkType>
NetworkType& MDNTrainer<NetworkType>::network()
{
    return _network;
}

template<typename NetworkType>
SupervisedDataset<double*, double*>& MDNTrainer<NetworkType>::dataset()
{
    return *_dataset;
}

template<typename NetworkType>
SupervisedDataset<double*, double*>& MDNTrainer<NetworkType>::validationset()
{
	return *_validationset;
}

template<typename NetworkType>
int MDNTrainer<NetworkType>::train(int epochs)
{
    // set custom signal handler
    struct sigaction sigIntHandler, oldSigAction;
    bool is_batch = false;
    int opt_iterations = epochs;

    sigIntHandler.sa_handler = MDNTrainer<NetworkType>::abort_handler;
    sigemptyset(&sigIntHandler.sa_mask);
    //sigIntHandler.sa_flags = SA_RESETHAND;
    sigaction(SIGINT, &sigIntHandler, &oldSigAction);

    _errors.reserve(_errors.size()+epochs);
    _validation_errors.reserve(_validation_errors.size()+epochs);

    if (_it_count == 0) {
        _report_every = epochs >= 100 ? epochs / 10 : 1;
    }

    if (_batch_size < dataset().size()) {
        is_batch = true;
        opt_iterations = _batch_epochs;
    }

    try
    {
        if (use_cg == true) {
            mincgsetcond(_cgstate, _epsg, _epsf, _epsx, opt_iterations);
        } else {
            minlbfgssetcond(_lbfgsstate, _epsg, _epsf, _epsx, opt_iterations);
        }
    }
    catch (ap_error& e)
    {
        std::cout << e.msg << std::endl;
    }

    try
    {
        if (_it_count == 0) {
            if (use_cg == true) {
              alglib::mincgoptimize(
                  _cgstate, MDNTrainer::f_df, MDNTrainer::report, this
              );
            } else {
              alglib::minlbfgsoptimize(
                  _lbfgsstate, MDNTrainer::f_df, MDNTrainer::report, this
              );
            }
        } else {
            real_1d_array x;
            x.setlength(_n_params);
            get_params(x);
            if (use_cg == true) {
              alglib::mincgrestartfrom(_cgstate, x);
              alglib::mincgoptimize(
                _cgstate, MDNTrainer::f_df, MDNTrainer::report, this
              );
            } else {
              alglib::minlbfgsrestartfrom(_lbfgsstate, x);
              alglib::minlbfgsoptimize(
                  _lbfgsstate, MDNTrainer::f_df, MDNTrainer::report, this
              );
            }
        }
    }
    catch (ap_error& e)
    {
        std::cout << "An ALGLIB error occurred!" << std::endl;
        std::cout << e.msg << std::endl;
    }
    catch (MDNTrainerAbortError& e)
    {
    	// If a validation set was provided, set parameters to the set of parameters
		// where the minimum validation set error has occured.
    	if (_validationset != 0) {
    	    set_params(_optimal_x);
    	}
    	sigaction(SIGINT, &oldSigAction, NULL);
    	//restore_abort_handler(oldSigIntHandler);
        return -1;
    }

    real_1d_array param_new;
    param_new.setlength(_n_params);
    if (use_cg == true) {
        mincgresults(_cgstate, param_new, _cgrep);
        _terminationtype = _cgrep.terminationtype;
    } else {
        minlbfgsresults(_lbfgsstate, param_new, _lbfgsrep);
        _terminationtype = _lbfgsrep.terminationtype;
    }

    // if this is a batch run, draw a new batch by randomizing the index sequence and restart
    if (is_batch) {
        std::random_shuffle(_idxs.begin(), _idxs.end());
        int tmp = epochs - (int) _lbfgsrep.iterationscount - 1;
        if (tmp > 0) {
            std::cout << "New batch at iteration " << _it_count << std::endl;
            sigaction(SIGINT, &oldSigAction, NULL);
            return train(tmp);
        }
    }

    // If a validation set was provided, set parameters to the set of parameters
	// where the minimum validation set error has occured.
    if (_validationset != 0) {
    	set_params(_optimal_x);
    } else {
		set_params(param_new);
    }

    sigaction(SIGINT, &oldSigAction, NULL);
    //std::cout << "Finished training" << std::endl;

    return _terminationtype;
}

template<typename NetworkType>
int MDNTrainer<NetworkType>::train()
{
    return train(1);
}

template<typename NetworkType>
void MDNTrainer<NetworkType>::get_params(std::vector<double>& x)
{
    int idx = 0;
    std::vector<Parametrized*>::iterator param_iter;
    //std::vector<double> tmp(_n_params);
    for (param_iter = network().parametrizeds().begin();
         param_iter != network().parametrizeds().end();
         ++param_iter)
    {
        std::copy((*param_iter)->get_parameters(),
                  (*param_iter)->get_parameters()+(*param_iter)->size(),
                  x.begin()+idx);
        idx += (*param_iter)->size();
    }
}

template<typename NetworkType>
void MDNTrainer<NetworkType>::get_params(real_1d_array& x)
{
    int idx = 0;
    std::vector<Parametrized*>::iterator param_iter;
    //std::vector<double> tmp(_n_params);
    for (param_iter = network().parametrizeds().begin();
         param_iter != network().parametrizeds().end();
         ++param_iter)
    {
        std::copy((*param_iter)->get_parameters(),
                  (*param_iter)->get_parameters()+(*param_iter)->size(),
                  x.getcontent()+idx);
        idx += (*param_iter)->size();
    }
    //x.setcontent(_n_params, &tmp[0]);
}

template<typename NetworkType>
void MDNTrainer<NetworkType>::set_params(const real_1d_array& x)
{
    const double* tmp = x.getcontent();
    double* parameters;
    int idx=0;
    std::vector<Parametrized*>::iterator param_iter;
    for (param_iter = network().parametrizeds().begin();
         param_iter != network().parametrizeds().end();
         ++param_iter)
    {
        parameters = (*param_iter)->get_parameters();
        std::copy(tmp+idx, tmp+idx+(*param_iter)->size(),
                  parameters);
        idx += (*param_iter)->size();
    }
}

template<typename NetworkType>
void MDNTrainer<NetworkType>::set_params(const std::vector<double>& x)
{
    double* parameters;
    int idx=0;
    std::vector<Parametrized*>::iterator param_iter;
    for (param_iter = network().parametrizeds().begin();
         param_iter != network().parametrizeds().end();
         ++param_iter)
    {
        parameters = (*param_iter)->get_parameters();
        std::copy(x.begin()+idx, x.begin()+idx+(*param_iter)->size(),
                  parameters);
        idx += (*param_iter)->size();
    }
}

template<typename NetworkType>
void MDNTrainer<NetworkType>::get_derivs(real_1d_array& x)
{
    int idx = 0;
    std::vector<Parametrized*>::iterator param_iter;

    for (param_iter = network().parametrizeds().begin();
         param_iter != network().parametrizeds().end();
         ++param_iter)
    {
        std::copy((*param_iter)->get_derivatives(),
                  (*param_iter)->get_derivatives()+(*param_iter)->size(),
                  x.getcontent()+idx);
        idx += (*param_iter)->size();
    }
}

template<typename NetworkType>
void MDNTrainer<NetworkType>::get_derivs(double* derivs)
{
    int idx = 0;
    std::vector<Parametrized*>::iterator param_iter;
    std::vector<double> tmp(_n_params);
    for (param_iter = network().parametrizeds().begin();
         param_iter != network().parametrizeds().end();
         ++param_iter)
    {
        std::copy((*param_iter)->get_derivatives(),
                  (*param_iter)->get_derivatives()+(*param_iter)->size(),
                  derivs+idx);
        idx += (*param_iter)->size();
    }
}

template<typename NetworkType>
void MDNTrainer<NetworkType>::f_df(const real_1d_array &x, double &func, real_1d_array &grad,
                      void *ptr)
{
    //std::cout << "f_df" << std::endl;
    MDNTrainer* trainer = (MDNTrainer *)ptr;
    // TODO: optimize performance by not copying old parameters, but just keeping
    // a pointer
    //real_1d_array old_params;
    //old_params.setlength(trainer->_n_params);
    //trainer->get_params(old_params);

    trainer->set_params(x);

    double* output_err = new double[trainer->_n_params];
    const double* y;
    //double* tmp = new double[trainer->_n_params];
    //double* derivs = new double[trainer->_n_params];

    trainer->network().clear_derivatives();
    //trainer->get_derivs(grad);
    //std::cout << "Grad: " << grad.tostring(4).c_str() << std::endl;

    //func = 0;
    double err = 0.0;
    int k = 0;
    int idx;

    /*
    printf("Orig trainer at %p\n", trainer);
    printf("Orig network at %p\n", &trainer->network());
    printf("Orig inp buffer at %p\n", &trainer->network().input());
    printf("Orig outp buffer at %p\n", &trainer->network().output());
    printf("Orig inerr buffer at %p\n", &trainer->network().inerror());
    printf("Orig outerr buffer at %p\n", &trainer->network().outerror());

	#pragma omp parallel firstprivate(trainer)
    {
    	MDNTrainer ctrainer = *trainer;
    	MDN cnetwork = trainer->network();

    	printf("Thread %d, network %p\n", omp_get_thread_num(), &cnetwork);
    	printf("Thread %d, inp buffer %p\n", omp_get_thread_num(), &cnetwork.input());
    	printf("Thread %d, out buffer %p\n", omp_get_thread_num(), &cnetwork.output());
    	printf("Thread %d, inerr buffer %p\n", omp_get_thread_num(), &cnetwork.inerror());
    	printf("Thread %d, outerr buffer %p\n", omp_get_thread_num(), &cnetwork.outerror());
    	//printf("Thread %d, trainer %p\n", omp_get_thread_num(), trainer);
    	//printf("Thread %d, trainer->network() %p\n", omp_get_thread_num(), &trainer->network());
    	//printf("Thread %d, ctrainer (local copy) %p\n", omp_get_thread_num(), &ctrainer);
    	//printf("Thread %d, ctrainer->network() %p\n", omp_get_thread_num(), &ctrainer.network());
    }
    */

    //for (k=0; k < trainer->dataset().size(); ++k)
    for (k=0; k < trainer->_batch_size; ++k)
    {
        idx = trainer->_idxs.at(k);
        //std::cout << idx << ", ";
        y = trainer->network().activate(trainer->dataset()[idx].first);
        err += trainer->network().get_error(y, trainer->dataset().targetsize(),
            trainer->dataset()[idx].second, trainer->dataset().targetsize());
        trainer->network().get_output_error(y, trainer->dataset().targetsize(),
            trainer->dataset()[idx].second, trainer->dataset().targetsize(),
           output_err);
        trainer->network().back_activate(output_err);
    }
    //std::cout << err << std::endl;

    //double eps = static_cast <double> (rand()) / (static_cast <double> (RAND_MAX/1.0));
    //std::cout << eps << std::endl;
    //func = err + eps;
    func = err;
    trainer->get_derivs(grad);
    //trainer->get_derivs(derivs);
    //grad.setlength(trainer->_n_params);
    //grad.setcontent(trainer->_n_params, derivs);
//    for (int p=0; p!=trainer->_n_params; ++p)
//    {
//        grad[p] = derivs[p];
//    }

    // restore old parameters
    //trainer->set_params(old_params);
    //std::cout << func << std::endl;
    //std::cout << "Error: " << (func/trainer->_batch_size) << std::endl;
    //std::cout << "Grad: " << grad.tostring(4).c_str() << std::endl;
}


template<typename NetworkType>
void MDNTrainer<NetworkType>::report(const real_1d_array &x, double func, void *ptr)
{
    MDNTrainer* trainer = (MDNTrainer *)ptr;
    trainer->_it_count++;
    //trainer->_errors.push_back(func/trainer->dataset().size());
    trainer->_errors.push_back(func/trainer->_batch_size);
    double validation_err = 0.0;
    if (trainer->_validationset != 0) {
    	const double* y;
    	for (int k=0; k < trainer->validationset().size(); ++k) {
			y = trainer->network().activate(trainer->validationset()[k].first);
			validation_err += trainer->network().get_error(
				y, trainer->validationset().targetsize(),
				trainer->validationset()[k].second,
				trainer->validationset().targetsize()
			);
    	}
    	trainer->_validation_errors.push_back(validation_err/trainer->validationset().size());

    	// keep track of early stopping information
    	if (validation_err < trainer->_optimal_err or trainer->_optimal_err == 0) {
    		trainer->_optimal_err = validation_err;
    		trainer->_optimal_it = trainer->_it_count-1;
    		trainer->get_params(trainer->_optimal_x);
    	}
    }
    if ((trainer->_it_count % trainer->_report_every) == 0) {
        std::cout << "Epoch " << trainer->_it_count
            << ", E=" << trainer->_errors.back();
        if (validation_err != 0.0) {
        	std::cout << ", E_validation=" << trainer->_validation_errors.back();
        }
        std::cout << std::endl;
    }
}

template<typename NetworkType>
int MDNTrainer<NetworkType>::get_terminationtype()
{
    return _terminationtype;
}

template<typename NetworkType>
void MDNTrainer<NetworkType>::abort_handler(int s)
{
    throw MDNTrainerAbortError();
}

template<typename NetworkType>
std::vector<double> MDNTrainer<NetworkType>::getErrorTrace()
{
	return _errors;
}

template<typename NetworkType>
std::vector<double> MDNTrainer<NetworkType>::getValidationErrorTrace()
{
	return _validation_errors;
}

template<typename NetworkType>
std::vector<double> MDNTrainer<NetworkType>::getOptimalParams()
{
	return _optimal_x;
}

template<typename NetworkType>
int MDNTrainer<NetworkType>::getOptimalEpoch()
{
	return _optimal_it;
}

#endif
