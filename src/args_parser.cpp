#include <fstream>
#include <iostream>

#include "args_parser.h"



Parameters::Parameters(int argc_, char **argv_) {


    // default values
    sa = (1);
    sb = (4);
    gapo = (6);
    gape = (1);
    start_pos = (WITHOUT_START); 
    print_out = (0);
    n_threads = (1);
    
    k_band = (0);

    isPacked = false;
    isReverseComplement = false;

    secondBest = FALSE;

    // query head, target head, query tail, target tail
    semiglobal_skipping_head = TARGET;
    semiglobal_skipping_tail = TARGET;

    algo = (UNKNOWN);

    query_batch_fasta_filename = "";
    target_batch_fasta_filename = "";

    argc = argc_;
    argv = argv_;

}

Parameters::~Parameters() {
    query_batch_fasta.close();
    target_batch_fasta.close();
}

void Parameters::print() {
    std::cerr <<  "sa=" << sa <<" , sb=" << sb <<" , gapo=" <<  gapo << " , gape="<<gape << std::endl;
    std::cerr <<  "start_pos=" << start_pos <<" , print_out=" << print_out <<" , n_threads=" <<  n_threads << std::endl;
    std::cerr <<  "semiglobal_skipping_head=" << semiglobal_skipping_head <<" , semiglobal_skipping_tail=" << semiglobal_skipping_tail <<" , algo=" <<  algo << std::endl;
    std::cerr <<  std::boolalpha << "isPacked = " << isPacked  << " , secondBest = " << secondBest << std::endl;
    std::cerr <<  "query_batch_fasta_filename=" << query_batch_fasta_filename <<" , target_batch_fasta_filename=" << target_batch_fasta_filename << std::endl;
}

void Parameters::failure(fail_type f) {
    switch(f)
    {
            case NOT_ENOUGH_ARGS:
                std::cerr << "Not enough Parameters. Required: -y AL_TYPE file1.fasta file2.fasta. See help (--help, -h) for usage. " << std::endl;
            break;
            case WRONG_ARG:
                std::cerr << "Wrong argument. See help (--help, -h) for usage. " << std::endl;
            break;
            case WRONG_FILES:
                std::cerr << "File error: either a file doesn't exist, or cannot be opened." << std::endl;
            break;

            default:
            break;
    }
    exit(1);
}

void Parameters::help() {
            std::cerr << "Usage: ./test_prog.out [-a] [-b] [-q] [-r] [-s] [-p] [-n] [-y] <query_batch.fasta> <target_batch.fasta>" << std::endl;
            std::cerr << "Options: -a INT    match score ["<< sa <<"]" << std::endl;
            std::cerr << "         -b INT    mismatch penalty [" << sb << "]"<< std::endl;
            std::cerr << "         -q INT    gap open penalty [" << gapo << "]" << std::endl;
            std::cerr << "         -r INT    gap extension penalty ["<< gape <<"]" << std::endl;
            std::cerr << "         -s        also find the start position" << std::endl;
            std::cerr << "         -p        print the alignment results" << std::endl;
            std::cerr << "         -n INT    Number of threads ["<< n_threads<<"]" << std::endl;
            std::cerr << "         -y AL_TYPE       Alignment type . Must be \"local\", \"semi_global\", \"global\", \"ksw\" "  << std::endl;
	    std::cerr << "         -x HEAD TAIL     specifies, for semi-global alignment, wha should be skipped for heads and tails of the sequences. (NONE, QUERY, TARGET, BOTH)" << std::endl;
            std::cerr << "         -k INT    Band width in case \"banded\" is selected."  << std::endl;
            std::cerr << "         --help, -h : displays this message." << std::endl;
            std::cerr << "         --second-best   displays second best score (WITHOUT_START only)." << std::endl;
            std::cerr << "Single-pack multi-Parameters (e.g. -sp) is not supported." << std::endl;
            std::cerr << "		  "  << std::endl;
}


void Parameters::parse() {

    // before testing anything, check if calling for help.
    int c;
        
    std::string arg_next = "";
    std::string arg_cur = "";

    for (c = 1; c < argc; c++)
    {
        arg_cur = std::string((const char*) (*(argv + c) ) );
        arg_next = "";
        if (!arg_cur.compare("--help") || !arg_cur.compare("-h"))
        {
            help();
            exit(0);
        }
    }

    if (argc < 4)
    {
        failure(NOT_ENOUGH_ARGS);
    }

    for (c = 1; c < argc - 2; c++)
    {
        arg_cur = std::string((const char*) (*(argv + c) ) );
        if (arg_cur.at(0) == '-' && arg_cur.at(1) == '-' )
        {
            if (!arg_cur.compare("--help"))
            {
                help();
                exit(0);
            }
            if (!arg_cur.compare("--second-best"))
            {
                secondBest = TRUE;
            }

        } else if (arg_cur.at(0) == '-' )
        {
            if (arg_cur.length() > 2)
                failure(WRONG_ARG);
            char param = arg_cur.at(1);
            switch(param)
            {
                case 'y':
                    c++;
                    arg_next = std::string((const char*) (*(argv + c) ) );
                    if (!arg_next.compare("local"))
                        algo = LOCAL;
                    else if (!arg_next.compare("semi_global"))
                        algo = SEMI_GLOBAL;
                    else if (!arg_next.compare("global"))
                        algo = GLOBAL;
                    else if (!arg_next.compare("microloc"))
                        algo = MICROLOCAL;
                    else if (!arg_next.compare("ksw"))
                    {
                        algo = KSW;
                    }
                break;
                case 'a':
                    c++;
                    arg_next = std::string((const char*) (*(argv + c) ) );
                    sa = std::stoi(arg_next);
                break;
                case 'b':
                    c++;
                    arg_next = std::string((const char*) (*(argv + c) ) );
                    sb = std::stoi(arg_next);
                break;
                case 'q':
                    c++;
                    arg_next = std::string((const char*) (*(argv + c) ) );
                    gapo = std::stoi(arg_next);
                break;
                case 'r':
                    c++;
                    arg_next = std::string((const char*) (*(argv + c) ) );
                    gape = std::stoi(arg_next);
                break;
                case 's':
                    start_pos = WITH_START;
                break;
                case 'p':
                    print_out = 1;
                break;
                case 'n':
                    c++;
                    arg_next = std::string((const char*) (*(argv + c) ) );
                    n_threads = std::stoi(arg_next);
                break;
                case 'k':
                    c++;
                    arg_next = std::string((const char*) (*(argv + c) ) );
                    k_band = std::stoi(arg_next);
                break;
                case 'x':
                    c++;
                    arg_next = std::string((const char*) (*(argv + c) ) );
                    if (!arg_next.compare("NONE"))
                        semiglobal_skipping_head = NONE;
                    else if (!arg_next.compare("TARGET"))
                        semiglobal_skipping_head = TARGET;
                    else if (!arg_next.compare("QUERY"))
                        semiglobal_skipping_head = QUERY;
                    else if (!arg_next.compare("BOTH"))
                        semiglobal_skipping_head = BOTH;
                    else 
                    {
                        failure(WRONG_ARG);
                    }

                    c++;
                    arg_next = std::string((const char*) (*(argv + c) ) );
                    if (!arg_next.compare("NONE"))
                        semiglobal_skipping_tail = NONE;
                    else if (!arg_next.compare("TARGET"))
                        semiglobal_skipping_tail = TARGET;
                    else if (!arg_next.compare("QUERY"))
                        semiglobal_skipping_tail = QUERY;
                    else if (!arg_next.compare("BOTH"))
                        semiglobal_skipping_tail = BOTH;
                    else 
                    {
                        failure(WRONG_ARG);
                    }
                break;

            }

            
        } else {
            failure(WRONG_ARG);
        }
    }


    // the last 2 Parameters are the 2 filenames.
    query_batch_fasta_filename = std::string( (const char*)  (*(argv + c) ) );
    c++;
    target_batch_fasta_filename = std::string( (const char*) (*(argv + c) ) );

    // Parameters retrieved successfully, open files.
    fileopen();
}

void Parameters::fileopen() {
    query_batch_fasta.open(query_batch_fasta_filename, std::ifstream::in);
    if (!query_batch_fasta)
        failure(WRONG_FILES);

    target_batch_fasta.open(target_batch_fasta_filename);
    if (!target_batch_fasta)
        failure(WRONG_FILES);
}
