import random
import matplotlib.pyplot as plt
import numpy as np


class Operation:
    def __init__(self, machine, duration):
        self.machine = machine
        self.duration = duration


class Job:
    def __init__(self, name, operations):
        self.name = name
        self.operations = operations


def generate_initial_population(jobs, population_size):
    population = []
    for _ in range(population_size):
        individual = []
        for job_id, job in enumerate(jobs):
            for op_index in range(len(job.operations)):
                individual.append((job_id, op_index))
        random.shuffle(individual)
        population.append(individual)
    return population


def calculate_makespan(schedule, jobs, num_machines):
    if not schedule:
        print("Error: Empty schedule")
        return None

    machine_schedules = [[] for _ in range(num_machines)]
    job_end_times = [0] * len(jobs)
    time = 0

    for job_id, operation_index in schedule:
        if job_id < 0 or job_id >= len(jobs) or operation_index < 0 or operation_index >= len(jobs[job_id].operations):
            return None

        operation = jobs[job_id].operations[operation_index]
        machine_id = operation.machine - 1  # Adjust for 0-based index
        if machine_id < 0 or machine_id >= num_machines:
            return None

        prev_op_end = job_end_times[job_id]
        machine_free = machine_schedules[machine_id][-1][1] if machine_schedules[machine_id] else 0
        start_time = max(prev_op_end, machine_free)
        end_time = start_time + operation.duration
        machine_schedules[machine_id].append((start_time, end_time, job_id))
        job_end_times[job_id] = operation_index + 1  # Update the job end time based on the completed operation
        time = max(time, end_time)

    return time


def evaluate_fitness(individual, jobs, num_machines):
    return calculate_makespan(individual, jobs, num_machines)


def tournament_selection(population, fitnesses, tournament_size):
    selected = []
    for _ in range(2):
        tournament = random.sample(list(zip(population, fitnesses)), tournament_size)
        valid_individuals = [ind for ind, fit in tournament if fit is not None]
        if valid_individuals:
            selected.append(min(valid_individuals, key=lambda x: fitnesses[population.index(x)]))
        else:
            return None, None  # Return None if no valid individuals found in tournament
    return selected


def crossover(parent1, parent2):
    size = len(parent1)
    point1, point2 = sorted(random.sample(range(size), 2))

    child1 = parent1[:point1] + parent2[point1:point2] + parent1[point2:]
    child2 = parent2[:point1] + parent1[point1:point2] + parent2[point2:]

    def repair(child):
        position_map = {job_id: 0 for job_id in range(len(jobs))}
        for i, (job_id, op_index) in enumerate(child):
            correct_op_index = position_map[job_id]
            position_map[job_id] += 1
            child[i] = (job_id, correct_op_index)
        return child  # Add return statement to return the repaired child

    child1 = repair(child1)
    child2 = repair(child2)

    return child1, child2


def mutate(individual, mutation_rate):
    size = len(individual) - 1  # Corrected size calculation
    for i in range(size):
        if random.random() < mutation_rate:
            j = random.randint(0, size - 1)
            if individual[i][0] == individual[j][0]:
                individual[i], individual[j] = individual[j], individual[i]
    return individual


def is_valid_individual(individual, jobs):
    job_operation_map = {job_id: [] for job_id in range(len(jobs))}
    for job_id, op_index in individual:
        job_operation_map[job_id].append(op_index)
    for job_id, operations in job_operation_map.items():
        if operations != sorted(operations):
            return False
    return True


def genetic_algorithm(jobs, num_machines, population_size=50, generations=100, mutation_rate=0.1, tournament_size=5):
    population = generate_initial_population(jobs, population_size)
    for generation in range(generations):
        fitnesses = [evaluate_fitness(individual, jobs, num_machines) for individual in population]
        if not population:  # Check if population is empty
            population = generate_initial_population(jobs, population_size)  # Regenerate population
            continue  # Skip the rest of the loop iteration

        # Ensure tournament size is smaller than population size
        tournament_size = min(tournament_size, len(population))
        new_population = []
        for _ in range(population_size // 2):
            parent1, parent2 = tournament_selection(population, fitnesses, tournament_size)
            if parent1 and parent2:
                child1, child2 = crossover(parent1, parent2)
                child1 = mutate(child1, mutation_rate)
                child2 = mutate(child2, mutation_rate)
                if is_valid_individual(child1, jobs):
                    new_population.append(child1)
                if is_valid_individual(child2, jobs):
                    new_population.append(child2)
        population = new_population
    fitnesses = [evaluate_fitness(individual, jobs, num_machines) for individual in population]
    best_individual = min(zip(population, fitnesses), key=lambda x: x[1])[0]
    print("The optimal solution is : ", best_individual)
    print("its fitness = ", evaluate_fitness(best_individual, jobs, num_machines))
    return best_individual



def decode_schedule(chromosome, jobs, num_machines):
    machine_schedules = [[] for _ in range(num_machines)]
    job_end_times = [0] * len(jobs)

    for job_id, op_index in chromosome:
        operation = jobs[job_id].operations[op_index]
        machine_id = operation.machine - 1  # Adjust for 0-based index
        prev_op_end = job_end_times[job_id]
        machine_free = machine_schedules[machine_id][-1][1] if machine_schedules[machine_id] else 0
        start_time = max(prev_op_end, machine_free)
        end_time = start_time + operation.duration
        machine_schedules[machine_id].append((start_time, end_time, job_id))
        job_end_times[job_id] = end_time

    return machine_schedules


def plot_gantt_chart(machine_schedules, jobs):
    fig, gnt = plt.subplots()
    gnt.set_xlabel('Time')
    gnt.set_ylabel('Machines')

    machine_labels = [f'Machine {i+1}' for i in range(len(machine_schedules))]
    gnt.set_yticks(np.arange(len(machine_schedules)) + 1)
    gnt.set_yticklabels(machine_labels)

    for i, schedule in enumerate(machine_schedules):
        for task in schedule:
            start_time, end_time, job_id = task
            gnt.broken_barh([(start_time, end_time - start_time)], (i + 0.5, 1), facecolors=(f'C{job_id}'))
            gnt.text(start_time + (end_time - start_time) / 2, i + 1, jobs[job_id].name, ha='center', va='center', color='black')

    plt.show()


# Example usage:
jobs = [
    Job("Job 1", [Operation(1, 10), Operation(2, 5), Operation(1, 12)]),
    Job("Job 2", [Operation(2, 7), Operation(7, 15), Operation(7, 8)]),
    Job("Job 3", [Operation(3, 10), Operation(6, 15), Operation(5, 20)]),
    Job("Job 4", [Operation(8, 10), Operation(8, 10), Operation(5, 15)]),
    Job("Job 5", [Operation(6, 9), Operation(7, 10), Operation(9, 20)]),
    Job("Job 6", [Operation(9, 7), Operation(4, 10), Operation(3, 13)]),
    Job("Job 7", [Operation(5, 15), Operation(9, 10), Operation(2, 9)]),
    Job("Job 8", [Operation(9, 7), Operation(4, 10), Operation(1, 6)]),
    Job("Job 9", [Operation(6, 15), Operation(7, 10), Operation(8, 10)])
]

num_machines = 9
best_schedule = genetic_algorithm(jobs, num_machines)
machine_schedules = decode_schedule(best_schedule, jobs, num_machines)
plot_gantt_chart(machine_schedules, jobs)
