import pygad
import numpy

"""
Given the following function:
    y = f(w1:w6) = w1x1 + w2x2 + w3x3 + w4x4 + w5x5 + 6wx6
    where (x1,x2,x3,x4,x5,x6)=(4,-2,3.5,5,-11,-4.7) and y=44

What are the best values for the 6 weights (w1 to w6)? We are going to use the genetic algorithm to optimize this function.
"""


# 적합도 계산함수
def fitness_func(solution, solution_idx):
    
    output = numpy.sum(solution*function_inputs)
    fitness = 1.0 / (numpy.abs(output - desired_output)) # (numpy.abs(output - desired_output)+ 0.000001))
    print(output,"output")
    print(fitness,"적합도")
    
    return fitness

fitness_function = fitness_func


function_inputs = [2,2,1,1] # Function inputs. 
desired_output = 135 # Function output.
init_range_low = -10 # 파라미터 탐색 공간 (low, high)
init_range_high = 10
parent_selection_type = "rws" # parent_selection_type = "sss" :" sss"(steady-state selection) "rws"(roulette wheel selection) "sus"(stochastic universal selection) "rank"(rank selection)"random"(random selection) "tournament"(tournament selection)
keep_parents = 3  # keep_parents : -1(다음 세대에 부모염색체 모두 포함), 0(다음 세대에 부모염색체 포함 X)
crossover_type = "single_point" # crossover_type : "single_point"(single-point crossover), "two_points", "uniform", "scattered", "None"(no crossover and no offspring, next gen is current)
mutation_type = "random" # mutation_type : "random" or "None" 
mutation_percent_genes = 10 # mutation_percent_genes : percentage of genes to mutate. mutation_type이 None이면 존재하지 않음.
random_mutation_min_val = -1.0 # random_mutation_min_val or max_val : mutation_type이 None이면 존재하지 않음, mutation 염색체의 어느 위치에서 할지 정해주는 변수
random_mutation_max_val = 1.0 # gene_space = None : 각 파라미터 별로 범위를 제약할 수 있음



# num_generation : number of generations, 유전 알고리즘 반복 횟수
# num_parents_mating : 부모로 선택되는 염색체 파라미터 개수
# fitness_func : return the fitness value of the solution, 적합도 계산 함수
# sol_per_pop : number of solutions in the population
# num_genes : length of function_inputs(찾아야하는 파라미터의 개수)
ga_instance = pygad.GA(num_generations=200,
                       num_parents_mating=5,
                       fitness_func=fitness_function,
                       sol_per_pop=10,
                       num_genes=len(function_inputs),
                       init_range_low = init_range_low,
                       init_range_high = init_range_high,
                       parent_selection_type= parent_selection_type,
                       keep_parents=keep_parents,
                       crossover_type= crossover_type,
                       mutation_type= mutation_type,
                       mutation_percent_genes= mutation_percent_genes)


ga_instance.run()
ga_instance.plot_result()

print("Generation = {generation}".format(generation=ga_instance.generations_completed)) # 실행한 generation 수

solution, solution_fitness, solution_idx = ga_instance.best_solution()
print("K 파라미터 최적해: {solution}".format(solution=solution)) # 최적값들
print("Fitness value of the best solution = {solution_fitness}".format(solution_fitness=solution_fitness)) # 가장 높은 적합도
print("Index of the best solution : {solution_idx}".format(solution_idx=solution_idx)) # 높은 적합도를 가진 solution의 인덱스

prediction = numpy.sum(numpy.array(function_inputs)*solution)
print("GA prediction과 실제값의 차 :{}", abs(prediction - desired_output))

'''
if ga_instance.best_solution_generation != -1:
    print("Best fitness value reached after {best_solution_generation} generations.".format(best_solution_generation=ga_instance.best_solution_generation))

'''