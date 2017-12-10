# NOTE: **ALWAYS** add a trailing '/' at the end of a folder path

def config_mixed():
  # for outputing multiple notes
  return {
    'net': 'conv5',
    'fusion_mode': 'no_fusion',
    'annot_folder': '/root/new_data/mixed_context46/annot_melody3/',
    'image_folder': '/root/new_data/mixed_context46/image/',
    'sr_ratio': 6,
    'audio_type': 'MIX',
    'multiple': True,
    'save_dir': './output_model/multiple/',
    'save_prefix': 'model_conv5_multi_train',
    'use_pretrained': False, # whether or not to use a pretrained model
    'pretrained_path': './output_model/context46/model_conv5_train_epoch10.pt',
  }

def config_cqt():
  # for cqt images & single output
  return {
    'net': 'conv5',
    'fusion_mode': 'no_fusion',
    'annot_folder': '/root/new_data/context46/annot/',
    'image_folder': '/root/new_data/context46/cqt_image/',
    'sr_ratio': 6,
    'audio_type': 'RAW',
    'multiple': False,
    'save_dir': './output_model/context46/cqt/',
    'save_prefix': 'model_conv5_cqt_train',
    'use_pretrained': True, # whether or not to use a pretrained model
    'pretrained_path': './output_model/context46/model_conv5_train_epoch10.pt',
  }

def config_context6():
  # for single output & smaller context
  return {
    'net': 'conv5',
    'fusion_mode': 'no_fusion',
    'annot_folder': '/root/new_data/context6/annot/',
    'image_folder': '/root/new_data/context6/image/',
    'sr_ratio': 1,
    'audio_type': 'RAW',
    'multiple': False,
    'save_dir': './output_model/context6/',
    'save_prefix': 'model_conv5_context6_train',
    'use_pretrained': False, # whether or not to use a pretrained model
    'pretrained_path': None,
  }  

def config_stacking():
  # single output + stacking mel & CQT spectrograms
  return {
    'net': 'conv5',
    'fusion_mode': 'stacking',
    'annot_folder': '/root/new_data/context46/annot/',
    'image_folder': '/root/new_data/context46/image/',
    'sr_ratio': 6,
    'audio_type': 'RAW',
    'multiple': False,
    'save_dir': './output_model/context46/stacking/',
    'save_prefix': 'model_conv5_stacking_train',
    'use_pretrained': False, # whether or not to use a pretrained model
    'pretrained_path': None,
  }


def config_early_fusion():
  # single output + concatenate mel & CQT after the first conv layer
  return {
    'net': 'conv5',
    'fusion_mode': 'early_fusion',
    'annot_folder': '/root/new_data/context46/annot/',
    'image_folder': '/root/new_data/context46/image/',
    'sr_ratio': 6,
    'audio_type': 'RAW',
    'multiple': False,
    'save_dir': './output_model/context46/early_fusion/',
    'save_prefix': 'model_conv5_early_fusion_train',
    'use_pretrained': False, # whether or not to use a pretrained model
    'pretrained_path': None,
  }

