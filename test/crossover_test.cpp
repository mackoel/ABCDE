#include "pch.h"

class TestSolution: public Solution
{
    public:
        TestSolution(const Abcde& _main_model, const Deep& _aux_model, const Parametrs& _param):Solution(_main_model, _aux_model, _param) {};
        void run_test(int iter, int index_thetha);
};

void TestSolution::run_test(int iter, int index_thetha)
{
        int size = 1;

        for (int j = 0; j < size; j++)
		{
			for (int i = 0; i < main_model.count_iter / size; i++)
			{
				main_model.curr_thetha = main_model.generate_vector_param(Distribution::NORM_WITH_PARAM);
			    main_model.posterior.thetha[i] = main_model.curr_thetha;
			    main_model.posterior.w[i] = 1.0 / main_model.count_iter;
			    main_model.posterior.error[i] = i;
			    main_model.new_posterior.thetha[i] = main_model.curr_thetha;
			    main_model.new_posterior.w[i] = 1.0 / main_model.count_iter;
			    main_model.new_posterior.error[i] = i;
			    main_model.posterior.thetha[i].delta = main_model.new_posterior.thetha[i].delta = main_model.generator.prior_distribution(Distribution::TYPE_DISTR::EXPON, 0.005);
			}
		}
		main_model.set_sample_dist_param();
		main_model.get_index_best();
		print_log(-1);

		for (int t = iter; t < main_model.start_iter + main_model.t; t++)
		{
			for (int j = 0; j < size; j++)
			{
				for (int i = 0; i < main_model.count_iter / size; i++)
				{
					if (main_model.crossing_mode == Abcde::CROSSING_MODE::ALL)
					{
						double choice = main_model.generator.prior_distribution(Distribution::TYPE_DISTR::RANDOM, 0.0, 1.0);
						if (choice < 0.05)
						{
							main_model.curr_thetha = main_model.mutation(i + j * main_model.count_iter / size);
						}
						else
						{
							main_model.curr_thetha = main_model.crossover(i + j * main_model.count_iter / size);
						}
					}
					else if (main_model.crossing_mode == Abcde::CROSSING_MODE::ONLY_CROSSOVER)
						main_model.curr_thetha = main_model.crossover(i + j * main_model.count_iter / size);
					else if (main_model.crossing_mode == Abcde::CROSSING_MODE::ONLY_MUTATION)
						main_model.curr_thetha = main_model.mutation(i + j * main_model.count_iter / size);

                    main_model.new_posterior.thetha[j * main_model.count_iter / (size)+i] = main_model.curr_thetha;
                    main_model.new_posterior.error[j * main_model.count_iter / (size)+i] = j * main_model.count_iter / (size)+i;
				}
			}
			main_model.update_posterior();//перерасчет весов
			Solution::copy_posterior(main_model.posterior, main_model.new_posterior);//перестановка
			main_model.set_sample_dist_param();

			for (int i = 0; i < main_model.count_iter; i++)
				manager.create_log_file(manager.state, main_model.posterior, main_model.new_posterior, main_model.norm_error, t, i, main_model.count_opt_param);
			print_log(t);
	    }
}


int main(int argc, char* argv[])
{
	int numTask, rank;
#ifdef MPIZE
	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &numTask);
#endif
	Parametrs param;
	param.process_program_options(argc, argv);
	Abcde abcde(param.config_file);
	Deep deep(param.config_file);
	TestSolution solution(abcde, deep, param);
	solution.run_test(0, 0);
#ifdef MPIZE
	MPI_Finalize();
#endif
    return 0;
}
