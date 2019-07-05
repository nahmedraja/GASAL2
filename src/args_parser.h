#ifndef ARGS_PARSER_H
#define ARGS_PARSER_H

/*
#include <stdint.h>


#include "gasal.h"
*/
#include <fstream>
#include <iostream>
#include "gasal.h"
#include <string.h>


enum fail_type {
    NOT_ENOUGH_ARGS,
    TOO_MANY_ARGS,
    WRONG_ARG,
    WRONG_FILES,
    WRONG_ALGO
};

class Parameters{

    public: 
        Parameters(int argc, char** argv);
        ~Parameters();
        void print();
        void failure(fail_type f);
        void help();
        void parse();
        void fileopen();



        int32_t sa;
        int32_t sb;
        int32_t gapo;
        int32_t gape;
        comp_start start_pos; 
        int print_out;
        int n_threads;
        int32_t k_band;

        Bool secondBest;

        bool isPacked;
        bool isReverseComplement;

        data_source semiglobal_skipping_head;
        data_source semiglobal_skipping_tail;

        algo_type algo;

        std::string query_batch_fasta_filename;
        std::string target_batch_fasta_filename;

        std::ifstream query_batch_fasta;
        std::ifstream target_batch_fasta;


    protected:

    private:
        int argc;
        char** argv;
};


#endif
