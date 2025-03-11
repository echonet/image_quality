from configs import transforms_config
from configs.paths_config import dataset_paths


DATASETS = {
	'my_data_encode': {
		'transforms': transforms_config.EncodeTransforms,
		'train_source_root': dataset_paths['my_train_data'],
		'train_target_root': dataset_paths['my_train_data'],
		'test_source_root': dataset_paths['my_test_data'],
		'test_target_root': dataset_paths['my_test_data'],
	}
}
