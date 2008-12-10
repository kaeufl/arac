// Part of Arac Neural Network Composition Library.
// (c) 2008 by Justin S Bayer, <bayer.justin@googlemail.com>


#ifndef Arac_STRUCTURE_NETWORKS_BASENETWORK_INCLUDED
#define Arac_STRUCTURE_NETWORKS_BASENETWORK_INCLUDED


#include "../modules/module.h"


namespace arac {
namespace structure {
namespace networks {


class BaseNetwork : public arac::structure::modules::Module
{

    public:
    
        BaseNetwork();
        virtual ~BaseNetwork();
    
        virtual const double* activate(double* input_p);
        virtual const double* back_activate(double* error_p);
        
        virtual void forward();
        
    protected:
        
        bool _dirty;
        
        virtual void sort() = 0;
};


}
}
}


#endif