## Pretrained models paths

experiments_dir = ''

e4e = "pretrained_models/e4e_v2.pt"
stylegan2_echo_grayscale_256 = "pretrained_models/stylegan2_echo_grayscale_256.pkl"

## Dirs for output files
checkpoint_dir = '' 
embedding_dir = ''

## Input info
### Input dir, where the images reside
input_data_path = ''
### Inversion identifier, used to keeping track of the inversion results. Both the latent code and the generator
input_data_id = ''

## Keywords
multi_id_model_type = "multi_id"

model_paths = {
    'stylegan_echo': 'pretrained_models/stylegan2_echo_grayscale_256.pt',
	'moco': 'pretrained_models/moco_v2_800ep_pretrain.pt'
}
