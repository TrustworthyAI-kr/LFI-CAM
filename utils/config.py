
def save_config(config, file_path):
    with open(file_path, 'wt') as output_file:
        output_file.write('Training configuration:\n')
        for k, v in vars(config).items():
            output_file.write('  {:>20} {}\n'.format(k, v))
        output_file.flush()
        output_file.close()
