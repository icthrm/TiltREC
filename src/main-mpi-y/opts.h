#ifndef OPTS_H__
#define OPTS_H__

#include <iostream>
#include <sstream>
#include <cassert>
extern "C"{
#include <getopt.h>
}
#include <string>
#include <stdio.h>
#include <stdlib.h>
#include "util/exception.h"

struct options{
	char input[255];
	char output[255];
	char initial[255];
	char angle[255];
	char xangle[255];
	//geometry
	float pitch_angle;
	float zshift;
	int thickness;
	float offset;
	//method
	std::string method;
	//params for iteration
	int iteration;
	float gamma;
};

inline void UsageDual(){
	std::cout << "[-INPUT(-i) Input Filename]\n"
			  << std::endl;
	std::cout << "    MRC file for reconstruction\n"
			  << std::endl;
	std::cout << "[-OUTPUT(-o) Output Filename]\n"
			  << std::endl;
	std::cout << "    MRC filename for result\n"
			  << std::endl;
	std::cout << "[-TILTFILE(-t) Tilt Angle Filename]\n"
			  << std::endl;
	std::cout << "    Tilt Angles\n"
			  << std::endl;
	std::cout << "[-INITIAL Initial Reconstruction Filename]\n"
			  << std::endl;
	std::cout << "    MRC file as initial model (reconstruction) for iteration methods (optinal)\n"
			  << std::endl;
	std::cout << "[-GEOMETRY(-g) 4 ints]\n"
			  << std::endl;
	std::cout << "    Geometry information: offset,pitch_angle,zshift,thickness\n"
			  << std::endl;
	std::cout << "[-METHOD(-m) Method Name(I,R)]\n"
			  << std::endl;
	std::cout << "    Back Projection: BPT\n"
			  << std::endl;
	std::cout << "    SART: SART,iteration_number,relax_parameter\n"
			  << std::endl;
	std::cout << "    SIRT: SIRT,iteration_number,relax_paramete \n"
			  << std::endl;
	std::cout << "-help(-h)" << std::endl;
	std::cout << "    Help Information\n"
			  << std::endl;
	std::cout << "EXAMPLES:\n"
			  << std::endl;
	std::cout << "mpirun -n 2 ./TiltRec-mpi --input ../../data/BBb/BBb_fin.mrc --output ../../data/BBb/BBb_BPT_z.mrc --tiltfile ../../data/BBb/BBb.rawtlt --geometry 0,0,0,300 --method BPT\n"
			  << std::endl;
}

inline void PrintOpts(const options& opt){
	std::cout<<"pitch_angle =  "<<opt.pitch_angle<<std::endl;
	std::cout<<"zshift = "<<opt.zshift<<std::endl;
	std::cout<<"thickness = "<<opt.thickness<<std::endl;
	std::cout<<"offset = "<<opt.offset<<std::endl;
	std::cout<<"input = "<<opt.input<<std::endl;
	std::cout<<"output = "<<opt.output<<std::endl;
	std::cout<<"initial = "<<opt.initial<<std::endl;
	std::cout<<"method = "<<opt.method<<std::endl;
	std::cout<<"iter = "<<opt.iteration<<std::endl;
	std::cout<<"step = "<<opt.gamma<<std::endl;
}

inline void InitOpts(options* opt){
	opt->pitch_angle = 0;
	opt->zshift = 0;
	opt->thickness = 0;
	opt->offset = 0;
	opt->input[0] = '\0';
	opt->output[0] = '\0';
	opt->initial[0] = '\0';
	opt->xangle[0] = '\0';
}

inline int GetOpts(int argc, char **argv, options* opts_){
	
	static struct option longopts[] ={
       { "help",            no_argument,            NULL,              'h' },
		{ "input",    	     required_argument,      NULL,              'i' },
		{ "output",       required_argument,      NULL,              'o' },
		{ "initial",        required_argument,      NULL,               'n' },
		{ "tiltfile",        required_argument,      NULL,              't' },
       { "geometry",        required_argument,      NULL,              'g' },
		{ "method",        required_argument,      NULL,              'm' },
       { NULL,              0,                      NULL,               0  }
    };

	int ch;
	while((ch = getopt_long(argc, argv, "hi:o:n:g:m:a:x:", longopts, NULL)) != -1){
		switch (ch){

		case '?':
			EX_TRACE("Invalid option '%s'.", argv[optind - 1]);
			return -1;

		case ':':
			EX_TRACE("Missing option argument for '%s'.", argv[optind - 1]);
			return -1;

		case 'h':
			UsageDual();
			return 0;

		case 'i':{
			std::stringstream iss(optarg);
			iss >> opts_->input;
			if(iss.fail()){
				EX_TRACE("Invalid argument '%s'.", optarg);
				return -1;
			}
		}
		break;
		
		case 't':{
			std::stringstream iss(optarg);
			iss >> opts_->angle;
			if(iss.fail()){
				EX_TRACE("Invalid argument '%s'.", optarg);
				return -1;
			}
		}
		break;
		
		//case 'x':{
		// 	std::stringstream iss(optarg);
		// 	iss >> opts_->xangle;
		// 	if(iss.fail()){
		// 		EX_TRACE("Invalid argument '%s'.", optarg);
		// 		return -1;
		// 	}
		// }
		// break;
		
		case 'o':{
			std::stringstream iss(optarg);
			iss >> opts_->output;
			if(iss.fail()){
				EX_TRACE("Invalid argument '%s'.", optarg);
			}
		}
		break;

		case 'n':{
			std::stringstream iss(optarg);
			iss >> opts_->initial;

			if(iss.fail()){
				EX_TRACE("Invalid argument '%s'.", optarg);
			}
		}
		break;

		case 'g':{ //offset,xaxistilt,zshift,thickness
			std::stringstream iss(optarg);
			std::string tmp;
			getline(iss, tmp, ',');
			opts_->offset = atof(tmp.c_str());

			getline(iss, tmp, ',');
			opts_->pitch_angle = atof(tmp.c_str());

			getline(iss, tmp, ',');
			opts_->zshift = atof(tmp.c_str());

			getline(iss, tmp);
			opts_->thickness = atoi(tmp.c_str());

			if(iss.fail()){
				EX_TRACE("Invalid argument '%s'.", optarg);
				return -1;
			}
		}
		break;

		case 'm':{
			std::stringstream iss(optarg);
			std::string tmp;
			if ( strcmp(optarg,"BPT") && strcmp(optarg,"RP") ){
				getline(iss, opts_->method, ',');

				if(opts_->method == "SIRT"||opts_->method == "SART"){
					getline(iss, tmp, ',');
					opts_->iteration = atoi(tmp.c_str());
					getline(iss, tmp);
					opts_->gamma = atof(tmp.c_str());
				}
				else{
					EX_TRACE("Iteration Method: [--method/-m method(BPT,SIRT or SART) Iteration numbers,relax parameter]\n");
				}

			}
			else{
				getline(iss, opts_->method);
			}
			if(iss.fail()){
				EX_TRACE("Invalid argument '%s'.", optarg);
				return -1;
			}
		}
		break;

		case 0:
		break;

		default:
			assert(false);
		} //end switch
	} //end while
	return 1;
}

#endif