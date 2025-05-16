import h5py

def filter_trials(input_file, output_file, trials_to_keep):
    """
    Filter an HDF5 file to keep only specified trials.

    Args:
        input_file (str): Path to the input .h5 file.
        output_file (str): Path to save the filtered .h5 file.
        trials_to_keep (list): List of trial names to retain (e.g., ["T1", "T2"]).
    """
    with h5py.File(input_file, 'r') as infile, h5py.File(output_file, 'w') as outfile:
        for trial in trials_to_keep:
            # Check for trial in all top-level keys
            found = False
            for key in infile.keys():
                if key == trial:
                    infile.copy(key, outfile)
                    found = True
                    break
            if not found:
                print(f"Warning: Trial {trial} not found in the input file.")



# Example usage
input_file = 'data-test/STG-PD-extra.h5'
output_file = 'data-test/STG-PD-extra-small.h5'
trials_to_keep = ['Trial1', 'Trial2']
def main():

    filter_trials(input_file, output_file, trials_to_keep)

main()
