
#include "opts.h"

void UsageDual()
{
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
	std::cout << "    Filtered Back Projection: FBP\n"
			  << std::endl;
	std::cout << "    Weighted Back Projection: WBP\n"
			  << std::endl;
	std::cout << "    SART: SART,iteration_number,relax_parameter\n"
			  << std::endl;
	std::cout << "    SIRT: SIRT,iteration_number,relax_paramete \n"
			  << std::endl;
	std::cout << "    ADMM: ADMM,iteration_number,cgiter,relax_paramete,soft \n"
			  << std::endl;
				  << std::endl;
	std::cout << "[-f2b 2 ]\n"
			  << std::endl;
	std::cout << "    Save the output file in byte (0 - 255) format to reduce storage size and improve processing efficiency\n"
			  << std::endl;
	std::cout << "-help(-h)" << std::endl;
	std::cout << "    Help Information\n"
			  << std::endl;
	std::cout << "EXAMPLES:\n"
			  << std::endl;
	std::cout << "mpirun -n 1 ./TiltRec-cuda --input ../../data/BBb/BBb_fin.mrc --output ../../data/BBb/BBb_WBP_y.mrc --tiltfile ../../data/BBb/BBb.rawtlt --geometry 0,0,0,300 --method WBP\n"
			  << std::endl;
}

// Overload for handling const char* strings
void PrintOption(const char *label, const char *value)
{
	if (value[0] != '\0')
		std::cout << label << " = " << value << std::endl;
}

// Overload for handling std::string
void PrintOption(const char *label, const std::string &value)
{
	if (!value.empty())
		std::cout << label << " = " << value << std::endl;
}

// Overload for handling float
void PrintOption(const char *label, float value)
{
	std::cout << label << " = " << value << std::endl;
}

// Overload for handling int
void PrintOption(const char *label, int value)
{
	std::cout << label << " = " << value << std::endl;
}

void PrintOption(const char *label, bool value)
{
	std::cout << label << " = " << value << std::endl;
}


void PrintOpts(const options &opt)
{
	PrintOption("input", opt.input);
	PrintOption("output", opt.output);
	PrintOption("initial", opt.initial);
	PrintOption("angle", opt.angle);

	// geometry
	PrintOption("pitch_angle", opt.pitch_angle);
	PrintOption("zshift", opt.zshift);
	PrintOption("thickness", opt.thickness);
	PrintOption("offset", opt.offset);

	// axis
	PrintOption("axis", opt.axis);

	// method
	PrintOption("method", opt.method);

	// params for iteration
	PrintOption("iteration", opt.iteration);
	PrintOption("cgiter", opt.cgiter);
	PrintOption("gamma", opt.gamma);
	PrintOption("soft", opt.soft);
	PrintOption("f2b", opt.f2b);
}

void InitOpts(options *opt)
{
	opt->pitch_angle = 0;
	opt->zshift = 0;
	opt->thickness = 0;
	opt->offset = 0;
	opt->angle[0] = '\0';
	opt->input[0] = '\0';
	opt->output[0] = '\0';
	opt->initial[0] = '\0';
	opt->axis = "y";
	opt->f2b = false;
}

int GetOpts(int argc, char **argv, options *opts_)
{

	static struct option longopts[] = {
		{"help", no_argument, NULL, 'h'},
		{"input", required_argument, NULL, 'i'},
		{"output", required_argument, NULL, 'o'},
		{"inital", required_argument, NULL, 'n'},
		{"tiltfile", required_argument, NULL, 't'},
		{"geometry", required_argument, NULL, 'g'},
		{"method", required_argument, NULL, 'm'},
		{"axis", required_argument, NULL, 'a'},
		{"f2b", no_argument, NULL, 'f'},
		{NULL, 0, NULL, 0}};

	int ch;
	while ((ch = getopt_long(argc, argv, "hi:o:n:t:g:m:a:f", longopts, NULL)) != -1)
	{
		switch (ch)
		{
		case '?':
			EX_TRACE("Invalid option '%s'.", argv[optind - 1]);
			return -1;

		case ':':
			EX_TRACE("Missing option argument for '%s'.", argv[optind - 1]);
			return -1;

		case 'h':
			UsageDual();
			return 0;

		case 'i':
		{
			std::stringstream iss(optarg);
			iss >> opts_->input;

			if (iss.eof() == false)
			{
				EX_TRACE("Too many arguments for input '%s'.\n", optarg);
				return -1;
			}

			if (iss.fail())
			{
				EX_TRACE("Invalid argument for input '%s'.\n", optarg);
				return -1;
			}
		}
		break;

		case 't':
		{
			std::stringstream iss(optarg);
			iss >> opts_->angle;

			if (iss.eof() == false)
			{
				EX_TRACE("Too many arguments for tiltfile '%s'.\n", optarg);
				return -1;
			}

			if (iss.fail())
			{
				EX_TRACE("Invalid argument for tiltfile'%s'.\n", optarg);
				return -1;
			}
		}
		break;

		case 'o':
		{
			std::stringstream iss(optarg);
			iss >> opts_->output;

			if (iss.eof() == false)
			{
				EX_TRACE("Too many arguments for output '%s'.\n", optarg);
				return -1;
			}

			if (iss.fail())
			{
				EX_TRACE("Invalid argument  for output '%s'.\n", optarg);
			}
		}
		break;

		case 'n':
		{
			std::stringstream iss(optarg);
			iss >> opts_->initial;

			if (iss.eof() == false)
			{
				EX_TRACE("Too many arguments for initial '%s'.\n", optarg);
				return -1;
			}

			if (iss.fail())
			{
				EX_TRACE("Invalid argument for initial '%s'.\n", optarg);
			}
		}
		break;

		case 'g': // offset,xaxistilt,zshift,thickness
		{
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

			if (iss.eof() == false)
			{
				EX_TRACE("Too many arguments for geometry '%s'.\n", optarg);
				return -1;
			}

			if (iss.fail())
			{
				EX_TRACE("Invalid argument  for geometry '%s'.\n", optarg);
				return -1;
			}
		}
		break;

		case 'a':
		{
			std::stringstream iss(optarg);
			iss >> opts_->axis;

			if (!strcmp(optarg, "y") && !strcmp(optarg, "Y") && !strcmp(optarg, "Z") && !strcmp(optarg, "z"))
			{
				EX_TRACE("Invalid argument for axis '%s'.\n", optarg);
				return -1;
			}

			if (iss.eof() == false)
			{
				EX_TRACE("Too many arguments for axis '%s'.\n", optarg);
				return -1;
			}

			if (iss.fail())
			{
				EX_TRACE("Invalid argument for axis '%s'.\n", optarg);
				return -1;
			}
		}
		break;

		case 'm':
		{
			std::stringstream iss(optarg);
			std::string tmp;
			if (strcmp(optarg, "BPT") && strcmp(optarg, "FBP") && strcmp(optarg, "WBP") && strcmp(optarg, "SART") && strcmp(optarg, "SIRT") && strcmp(optarg, "ADMM"))
			{
				getline(iss, opts_->method, ',');
				if (opts_->method == "SIRT" || opts_->method == "SART")
				{
					getline(iss, tmp, ',');
					opts_->iteration = atoi(tmp.c_str());
					getline(iss, tmp);
					opts_->gamma = atof(tmp.c_str());
				}
				else if (opts_->method == "ADMM")
				{
					getline(iss, tmp, ',');
					opts_->iteration = atoi(tmp.c_str());
					getline(iss, tmp, ',');
					opts_->cgiter = atof(tmp.c_str());
					getline(iss, tmp, ',');
					opts_->gamma = atof(tmp.c_str());
					getline(iss, tmp);
					opts_->soft = atof(tmp.c_str());
				}
				else
				{
					EX_TRACE("Iteration Method: [--mode/-m method(SIRT or SART or ADMM) Iteration numbers,relax parameter]\n"
							 "FBP Method: [--mode/-m method(FBP or BPT or WBP) ]\n");
				}
			}
			else
			{
				getline(iss, opts_->method);
			}

			if (iss.eof() == false)
			{
				EX_TRACE("Too many arguments for method '%s'.\n", optarg);
				return -1;
			}

			if (iss.fail())
			{
				EX_TRACE("Invalid argument for method '%s'.\n", optarg);
				return -1;
			}
		}
		break;
		
		case 'f':
		{
			opts_->f2b = true;
	
			if (optarg!= NULL)
			{
				EX_TRACE("Too many arguments for f2b '%s'.\n", optarg);
				return -1;
			}

		}
		break;

		case 0:
			break;

		default:
			std::cerr << "Unknown option: " << argv[optind - 1] << std::endl;
			return -1;
			// assert(false);
		} // end switch
	}	  // end while
	return 1;
}