import pandas as pd
import numpy as np

data = pd.read_csv('data.csv')


def generate_data(data, col_name, empty=0.0):
    empty_range = data[data[col_name] == empty]

    avg = data[col_name].mean()
    std = data[col_name].std()

    # generate data for the empty range but always above 0
    generated_data = np.random.normal(avg, std, len(empty_range))
    generated_data = np.where(generated_data < 0, 0, generated_data)

    if col_name == 'success_rate':
        generated_data = np.where(generated_data > 100, 100, generated_data)

    # convert generated data into int64 if the column is transaction_count
    if 'transaction_count' in col_name:
        generated_data = generated_data.astype(np.int64)
    
    # replace the empty range with the generated data
    data.loc[data[col_name] == empty, col_name] = generated_data

    return data


success_rate_gen = generate_data(data, 'success_rate')
transaction_count_success_gen = generate_data(data, 'transaction_count_success', empty=0)
total_amount_success_gen = generate_data(data, 'total_amount_success')
transaction_count_refused_gen = generate_data(data, 'transaction_count_refused', empty=0)
total_amount_refused_gen = generate_data(data, 'total_amount_refused')

# combine the generated data to the original data replacing the empty range
data = pd.concat([success_rate_gen, transaction_count_success_gen, total_amount_success_gen,
                  transaction_count_refused_gen, total_amount_refused_gen], axis=1)

# write the data into a csv file
data.to_csv('data_generated.csv', index=False)
