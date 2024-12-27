#ifndef OPTS_H__
#define OPTS_H__

#include <iostream>
#include <sstream>
#include <cassert>
// extern "C"
//{
#include <getopt.h>
//}
#include <string>
#include <stdio.h>
#include <stdlib.h>
#include "../util/exception.h"
#include <fstream>

struct options
{
	char input[255];
	char output[255];
	char initial[255];
	char angle[255];
	// geometry
	float pitch_angle;
	float zshift;
	int thickness;
	float offset;
	// method
	std::string method;
	// reconstruction axis
	std::string axis;
	// params for iteration
	int iteration;
	int cgiter;
	float gamma;
	float soft;
	bool f2b;
};

void UsageDual();

// Overload for handling const char* strings
void PrintOption(const char *label, const char *value);

// Overload for handling std::string
void PrintOption(const char *label, const std::string &value);

// Overload for handling float
void PrintOption(const char *label, float value);

// Overload for handling int
void PrintOption(const char *label, int value);

void PrintOpts(const options &opt);

void InitOpts(options *opt);

int GetOpts(int argc, char **argv, options *opts_);

#endif