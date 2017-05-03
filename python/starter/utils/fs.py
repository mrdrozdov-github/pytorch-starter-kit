import os


def fs_log_dir(FLAGS):
    return os.path.join(FLAGS.log_path, FLAGS.experiment_name)


def fs_log_path(FLAGS):
    return os.path.join(fs_log_dir(FLAGS), FLAGS.experiment_name + '.log')


def fs_checkpoint_path(FLAGS):
    return os.path.join(fs_log_dir(FLAGS), FLAGS.experiment_name + '.pt')


def fs_best_checkpoint_path(FLAGS):
    return os.path.join(fs_log_dir(FLAGS), FLAGS.experiment_name + '.pt-best')


def fs_flags_json_path(FLAGS):
    return os.path.join(fs_log_dir(FLAGS), FLAGS.experiment_name + '.json')
