import numpy as np

delimiter = '|'
trials = 100

for trial in range(trials):
    adjustment = sorted(np.random.normal(loc=0, scale=5.0, size=68))

    with open('mascot_rank.5_0.{}.csv'.format(trial), 'w') as output_file:
        with open('mascot_rank.csv', 'r') as input_file:
            for line_number, line in enumerate(input_file):
                print_data = [line.strip(), line_number + 1, adjustment[line_number]]
                print >> output_file, delimiter.join(map(str, print_data))

