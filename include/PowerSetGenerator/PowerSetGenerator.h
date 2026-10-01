#ifndef POWER_SET_GENERATOR_H
#define POWER_SET_GENERATOR_H

#include <vector>
#include <stdexcept>

using namespace std;

class PowerSetGenerator {
private:
    int setSize; // Size of the original set
    long iterationCount; // tracks number of subsets created (max 2^n)
    vector<int> subset;
    vector<int> complement;

public:
    PowerSetGenerator(int n); //constructor
    bool next(); //generates next subset element in power set
    vector<int> getSet() const; // list of indices representing subset (power set element)
    vector<int> getComplement() const; //gets complement of getSet() with original set as domain
};

#endif // POWER_SET_GENERATOR_H