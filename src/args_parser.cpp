#include "args_parser.h"

Arguments::Arguments(int argc_, char **argv_) {

        sa = (1);
        sb = (4);
        gapo = (6);
        gape = (1);
        start_pos = (WITHOUT_START); 
        print_out = (0);
        n_threads = (1);
        
        k_band = (0);

        // query head, target head, query tail, target tail
        semiglobal_skipping_head = NONE;
        semiglobal_skipping_tail = NONE;

        algo = (UNKNOWN);

        query_batch_fasta_filename = "";
        target_batch_fasta_filename = "";

        argc = argc_;
        argv = argv_;

}

Arguments::~Arguments() {
    query_batch_fasta.close();
    target_batch_fasta.close();
}

void Arguments::print() {
    std::cerr <<  "sa=" << sa <<" , sb=" << sb <<" , gapo=" <<  gapo << " , gape="<<gape << std::endl;
    std::cerr <<  "start_pos=" << start_pos <<" , print_out=" << print_out <<" , n_threads=" <<  n_threads << std::endl;
    std::cerr <<  "semiglobal_skipping_head=" << semiglobal_skipping_head <<" , semiglobal_skipping_tail=" << semiglobal_skipping_tail <<" , algo=" <<  algo << std::endl;
    std::cerr <<  "query_batch_fasta_filename=" << query_batch_fasta_filename <<" , target_batch_fasta_filename=" << target_batch_fasta_filename << std::endl;
}

void Arguments::failure(fail_type f) {
    switch(f)
    {
            case NOT_ENOUGH_ARGS:
                std::cerr << "Not enough arguments. Required: -y AL_TYPE file1.fasta file2.fasta. See help (--help, -h) for usage. " << std::endl;
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

void Arguments::help() {
            std::cerr << "Usage: ./test_prog.out [-a] [-b] [-q] [-r] [-s] [-p] [-n] [-y] <query_batch.fasta> <target_batch.fasta>" << std::endl;
            std::cerr << "Options: -a INT    match score ["<< sa <<"]" << std::endl;
            std::cerr << "         -b INT    mismatch penalty [" << sb << "]"<< std::endl;
            std::cerr << "         -q INT    gap open penalty [" << gapo << "]" << std::endl;
            std::cerr << "         -r INT    gap extension penalty ["<< gape <<"]" << std::endl;
            std::cerr << "         -s        also find the start position" << std::endl;
            std::cerr << "         -p        print the alignment results" << std::endl;
            std::cerr << "         -n INT    Number of threads ["<< n_threads<<"]" << std::endl;
            std::cerr << "         -y AL_TYPE       Alignment type . Must be \"local\", \"semi_global\", \"global\"  \"banded INT\" (size of band)"  << std::endl;
            std::cerr << "         -k INT    Band width in case \"banded\" is selected."  << std::endl;
            std::cerr << "         --help, -h : displays this message." << std::endl;
            std::cerr << "Single-pack multi-arguments (e.g. -sp) is not supported." << std::endl;
            std::cerr << "		  "  << std::endl;
}


void Arguments::parse() {
    if (argc < 4)
    {
        failure(NOT_ENOUGH_ARGS);
    }
    int c;
    for (c = 1; c < argc - 2; c++)
    {
        std::string arg_cur = std::string((const char*) (*(argv + c) ) );
        std::string arg_next = "";
        if (arg_cur.at(0) == '-' && arg_cur.at(1) == '-' )
        {
            if (!arg_cur.compare("--help"))
            {
                help();
                exit(0);
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
                    else if (!arg_next.compare("banded"))
                        algo = BANDED;
                    else if (!arg_next.compare("microloc"))
                        algo = MICROLOCAL;
                    else if (!arg_next.compare("fixedband"))
                    {
                        algo = FIXEDBAND;
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
            //failure(WRONG_ARG);
        }
    }


    // the last 2 arguments are the 2 filenames.
    query_batch_fasta_filename = std::string( (const char*)  (*(argv + c) ) );
    c++;
    target_batch_fasta_filename = std::string( (const char*) (*(argv + c) ) );

    fileopen();
}

void Arguments::fileopen() {
    query_batch_fasta.open(query_batch_fasta_filename, std::ifstream::in);
    if (!query_batch_fasta)
        failure(WRONG_FILES);

    target_batch_fasta.open(target_batch_fasta_filename);
    if (!target_batch_fasta)
        failure(WRONG_FILES);
}