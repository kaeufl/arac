#include <signal.h>
#include <iostream>
#include <omp.h>
//#include <gsl/gsl_blas.h>
#include "mdntrainer.h"
#include "../utilities/alglib/stdafx.h"
#include "../utilities/alglib/optimization.h"

using arac::optimization::MDNTrainer;
using arac::optimization::MDNTrainerAbortError;
using arac::structure::networks::MDN;
using arac::structure::Parametrized;
using namespace alglib;

MDNTrainer::MDNTrainer(MDN& network,
                       SupervisedDataset<double*, double*>& dataset) :
    _network(network),
    _dataset(dataset) 
{
    _n_params = 0;
    _it_count = 0;
    real_1d_array x;

    std::vector<Parametrized*>::iterator param_iter;
    for (param_iter = network.parametrizeds().begin();
        param_iter != network.parametrizeds().end();
        param_iter++)
    {
       _n_params += (*param_iter)->size();
    }

    x.setlength(_n_params);
    get_params(x);

    try
    {
       minlbfgscreate(_n_params, 5, x, _lbfgsstate);
       minlbfgssetxrep(_lbfgsstate, true);
    }
    catch (ap_error& e)
    {
       std::cout << e.msg << std::endl;
    }

    // set custom signal handler
    struct sigaction sigIntHandler;

    sigIntHandler.sa_handler = MDNTrainer::abort_handler;
    sigemptyset(&sigIntHandler.sa_mask);
    sigIntHandler.sa_flags = 0;
    sigaction(SIGINT, &sigIntHandler, NULL);
}

MDNTrainer::~MDNTrainer() {}

MDN& MDNTrainer::network()
{
    return _network;
}

SupervisedDataset<double*, double*>& MDNTrainer::dataset()
{
    return _dataset;
}

int MDNTrainer::train(int epochs)
{
    _report_every = epochs >= 100 ? epochs / 10 : 1;

    try
    {
        minlbfgssetcond(_lbfgsstate, _epsg, _epsf, _epsx, epochs);
    }
    catch (ap_error& e)
    {
        std::cout << e.msg << std::endl;
    }

    try
    {
        if (_it_count == 0) {
            alglib::minlbfgsoptimize(
                _lbfgsstate, MDNTrainer::f_df, MDNTrainer::report, this
            );
        } else {
            real_1d_array x;
            x.setlength(_n_params);
            get_params(x);
            alglib::minlbfgsrestartfrom(_lbfgsstate, x);
            alglib::minlbfgsoptimize(
                _lbfgsstate, MDNTrainer::f_df, MDNTrainer::report, this
            );
        }
    }
    catch (ap_error& e)
    {
        std::cout << e.msg << std::endl;
    }
    catch (MDNTrainerAbortError& e)
    {
        return -1;
    }
    real_1d_array param_new;
    param_new.setlength(_n_params);
    minlbfgsresults(_lbfgsstate, param_new, _lbfgsrep);
    set_params(param_new);

    _terminationtype = _lbfgsrep.terminationtype;
    return _terminationtype;
}

int MDNTrainer::train()
{
    return train(1);
}

void MDNTrainer::get_params(real_1d_array& x)
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

void MDNTrainer::set_params(const real_1d_array& x)
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

void MDNTrainer::get_derivs(real_1d_array& x)
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

void MDNTrainer::get_derivs(double* derivs)
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

void MDNTrainer::f_df(const real_1d_array &x, double &func, real_1d_array &grad,
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

//    printf("Orig trainer at %p\n", trainer);
//
//	#pragma omp parallel firstprivate(trainer)
//    {
//    	MDNTrainer ctrainer = *trainer;
//    	printf("Thread %d, trainer %p\n", omp_get_thread_num(), trainer);
//    	printf("Thread %d, trainer->network() %p\n", omp_get_thread_num(), &trainer->network());
//    	printf("Thread %d, ctrainer (local copy) %p\n", omp_get_thread_num(), &ctrainer);
//    	printf("Thread %d, ctrainer->network() %p\n", omp_get_thread_num(), &ctrainer.network());
//    }

    for (k=0; k < trainer->dataset().size(); ++k)
    {
        y = trainer->network().activate(trainer->dataset()[k].first);
        err += trainer->network().get_error(y, trainer->dataset().targetsize(),
            trainer->dataset()[k].second, trainer->dataset().targetsize());
        trainer->network().get_output_error(y, trainer->dataset().targetsize(),
            trainer->dataset()[k].second, trainer->dataset().targetsize(),
           output_err);
        trainer->network().back_activate(output_err);
    }

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

    //std::cout << "Error: " << (func/trainer->dataset().size()) << std::endl;
    //std::cout << "Grad: " << grad.tostring(4).c_str() << std::endl;
}


void MDNTrainer::report(const real_1d_array &x, double func, void *ptr)
{
    MDNTrainer* trainer = (MDNTrainer *)ptr;
    trainer->_it_count++;
    if ((trainer->_it_count % trainer->_report_every) == 0) {
        std::cout << "Epoch " << trainer->_it_count
            << ", E=" << (func/trainer->dataset().size())
            << std::endl;
    }
}

int MDNTrainer::get_terminationtype()
{
    return _terminationtype;
}

void MDNTrainer::abort_handler(int s)
{
    throw MDNTrainerAbortError();
}
