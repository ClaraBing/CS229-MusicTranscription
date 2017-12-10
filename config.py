def config_1():
  return {
    'net': 'conv5',
    'annot_folder': '/root/new_data/mixed_context46/annot_melody3/',
    'image_folder': '/root/new_data/mixed_context46/image/',
    'audio_type': 'MIX',
    'multiple': True,
    'save_dir': './output_model/multiple/',
    'save_prefix': 'model_conv5_multi_train',
    'use_pretrained': False, # whether or not to use a pretrained model
    'pretrained_path': './output_mode/context46/model_conv5_epoch10.pt',
  }

def config_2():
  # NOTE: add your own configuration
  pass
