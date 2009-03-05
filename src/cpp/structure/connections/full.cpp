// Part of Arac Neural Network Composition Library.
// (c) 2008 by Justin S Bayer, <bayer.justin@googlemail.com>


#include <iostream>

extern "C"
{
    #include "cblas.h"
}

#include "full.h"


using arac::structure::connections::FullConnection;
using arac::structure::connections::Connection;
using arac::structure::Parametrized;
using arac::structure::modules::Module;


FullConnection::FullConnection(Module* incoming_p, Module* outgoing_p) :
    Connection(incoming_p, outgoing_p),
    Parametrized(incoming_p->outsize() * outgoing_p->insize())
{
    
}


FullConnection::FullConnection(Module* incoming_p, Module* outgoing_p,
                               int incomingstart, int incomingstop, 
                               int outgoingstart, int outgoingstop) :
    Connection(incoming_p, outgoing_p, 
               incomingstart, incomingstop, 
               outgoingstart, outgoingstop),
    Parametrized((incomingstop - incomingstart) * (outgoingstop - outgoingstart))
{
    
}


FullConnection::FullConnection(Module* incoming_p, Module* outgoing_p,
               double* parameters_p, double* derivatives_p,
               int incomingstart, int incomingstop, 
               int outgoingstart, int outgoingstop) :
    Connection(incoming_p, outgoing_p, 
               incomingstart, incomingstop,
               outgoingstart, outgoingstop),
    Parametrized((incomingstop - incomingstart) * (outgoingstop - outgoingstart),
                 parameters_p, derivatives_p)           
{
}   

            
FullConnection::~FullConnection()
{
}


void FullConnection::_forward()
{
    if (timestep() - get_recurrent() < 0)
    {
        return;
    }

    int indim = _incomingstop - _incomingstart;
    int outdim = _outgoingstop - _outgoingstart;

    double* sink_p = _outgoing_p->input()[timestep()] + _outgoingstart;
    double* source_p = _incoming_p->output()[timestep() - get_recurrent()];
    source_p += _incomingstart;

    cblas_dgemv(CblasRowMajor, 
                // Do not transpose the matrix since we want to multiply from 
                // the right
                CblasNoTrans,
                // Dimensions of the matrix
                outdim,        
                indim,
                // Scalar for the matrix
                1.0,                    
                // Pointer to the matrix
                get_parameters(),    
                // Dimension of the vector
                indim,
                // Pointer to the vector
                source_p,
                // Some incrementer.
                1,                      
                // Scalar of the target vector
                1.0,                    
                // Pointer to the target vector
                sink_p,
                // Incrementer.
                1);   
}


void FullConnection::_backward()
{
    int this_timestep = timestep() - 1;
    if (this_timestep + get_recurrent() > sequencelength())
    {
        return;
    }
    
    int indim = _incomingstop - _incomingstart;
    int outdim = _outgoingstop - _outgoingstart;
    
    double* inerror_p = _incoming_p->outerror()[this_timestep] \
                        + _incomingstart;
    
    double* outerror_p = _outgoing_p->inerror()[this_timestep + get_recurrent()] \
                          + _outgoingstart;
                          
    double* input_p = _incoming_p->output()[this_timestep] \
                      + _incomingstart;

    cblas_dgemv(CblasColMajor, 
                // Do not transpose the matrix since we want to multiply from 
                // the right
                CblasNoTrans,
                // Dimensions of the matrix
                indim,        
                outdim,
                // Scalar for the matrix
                1.0,                    
                // Pointer to the matrix
                get_parameters(),    
                // Dimension of the vector
                indim,
                // Pointer to the vector
                outerror_p,
                // Some incrementer.
                1,                      
                // Scalar of the target vector
                1.0,                    
                // Pointer to the target vector
                inerror_p,
                // Incrementer.
                1);   

    double* derivs_p = get_derivatives();
    for (int i = 0; i < outdim; i++)
    {
        for (int j = 0; j < indim; j++)
        {
            derivs_p[i * indim + j] += outerror_p[i] * input_p[j];
        }
    }
}
