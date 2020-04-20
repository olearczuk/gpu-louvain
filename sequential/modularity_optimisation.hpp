#ifndef __MODULARITY_OPTIMISATION__HPP__
#define __MODULARITY_OPTIMISATION__HPP__
#include "utils.hpp"

/**
 * Finds new communities for vertices as long as total gain >= minimal gain.
 * Returns information, whether any changes in vertex -> community were done.
 */
bool optimise_modularity(float min_gain, data_structures& structures);

#endif /* __MODULARITY_OPTIMISATION__HPP__ */