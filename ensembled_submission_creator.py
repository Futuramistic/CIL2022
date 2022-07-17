import os
from tqdm import tqdm


"""
Given a set of submissions in CSV format, ensemble them into a single submission
"""


pred_dir = "submissions_to_ensemble/"
output_filename = "ensembled_submission.csv"


def main():
    if not os.path.isdir(pred_dir):
        print(f'Directory "{pred_dir}" (supposed to contain CSV files with submissions to ensemble) does not exist')
        return
    
    csv_filenames = []
    for filename in os.listdir(pred_dir):
        if filename.lower().endswith('.csv'):
            csv_filenames.append(filename)

    if len(csv_filenames) == 0:
        print(f'No CSV files found in directory "{pred_dir}"')
        return

    if len(csv_filenames) % 2 == 0:
        print('WARNING: found even number of submissions to ensemble; not recommended; will use value 1 to break ties')

    patch_order_list = []
    patches = {}
    for file_idx, filename in tqdm(enumerate(csv_filenames)):
        with open(os.path.join(pred_dir, filename), 'r') as file:
            file_str = file.read()
            for line in map(lambda l: l.strip().lower(), file_str.splitlines()):
                if line.startswith('id') or line.startswith('#') or not ',' in line:
                    continue
                key, value = line.split(',')
                patches[key] = patches.get(key, 0) + (-1 + 2 * int(value))
                if file_idx == 0:
                    patch_order_list.append(key)
    
    patches = {k: 1 if v >= 0 else 0 for k, v in patches.items()}
    out_lines = ['id,prediction', *[f'{k},{patches.get(k, 0)}' for k in patch_order_list]]
    with open(output_filename, 'w') as file:
        file.write('\n'.join(out_lines))
    print(f'Ensembled {len(csv_filenames)} submissions into "{output_filename}"')


if __name__ == '__main__':
    main()
