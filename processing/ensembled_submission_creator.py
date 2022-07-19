import argparse
import os
from tqdm import tqdm


"""
Given a set of submissions in CSV format, ensemble them into a single submission
"""


def main(submission_dir, output_filename):
    if not os.path.isdir(submission_dir):
        print(f'Directory "{submission_dir}" (supposed to contain CSV files with submissions to ensemble) '
               'does not exist')
        return
    
    csv_filenames = []
    for filename in os.listdir(submission_dir):
        if filename.lower().endswith('.csv'):
            csv_filenames.append(filename)

    if len(csv_filenames) == 0:
        print(f'No CSV files found in directory "{submission_dir}"')
        return

    if len(csv_filenames) % 2 == 0:
        print('WARNING: found even number of submissions to ensemble; not recommended; will use value 0 to break ties')

    patch_order_list = []
    patches = {}
    for file_idx, filename in tqdm(enumerate(csv_filenames)):
        with open(os.path.join(submission_dir, filename), 'r') as file:
            file_str = file.read()
            for line in map(lambda l: l.strip().lower(), file_str.splitlines()):
                if line.startswith('id') or line.startswith('#') or not ',' in line:
                    continue
                key, value = line.split(',')
                patches[key] = patches.get(key, 0) + (-1 + 2 * int(value))
                if file_idx == 0:
                    patch_order_list.append(key)
    
    patches = {k: 1 if v > 0 else 0 for k, v in patches.items()}
    out_lines = ['id,prediction', *[f'{k},{patches.get(k, 0)}' for k in patch_order_list]]
    with open(output_filename, 'w') as file:
        file.write('\n'.join(out_lines))
    print(f'Ensembled {len(csv_filenames)} submissions into "{output_filename}"')


if __name__ == '__main__':
    desc_str = 'Given a set of submissions in CSV format, ensemble them into a single submission'
    parser = argparse.ArgumentParser(description=desc_str)
    parser.add_argument('-d', '--submission_dir', required=False, default='submissions_to_ensemble', type=str,
                        help='Directory with submissions to ensemble')
    parser.add_argument('-f', '--output_filename', required=False, default='ensembled_submission.csv', type=str,
                        help='File to ensemble submissions into')
    options = parser.parse_args()
    main(options.submission_dir, options.output_filename)
