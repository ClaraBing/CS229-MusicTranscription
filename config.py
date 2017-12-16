# NOTE: **ALWAYS** add a trailing '/' at the end of a folder path


def config_base():
  # for base model
  return {
    'net': None,
    'annot_folder': '/root/MedleyDB_selected/Annotations/Melody_Annotations/MELODY1/val/',
    'image_folder': '/root/data/val/',
    'sr_ratio': 1,
    'audio_type': 'MIX',
    'multiple': False,
    'saved_weights': 'dataset/val_result_mtrx.npy',
  }

def config_mixed():
  # for outputing multiple notes
  return {
    'net': 'conv5',
    'fusion_mode': 'no_fusion',
    'annot_folder': '/root/new_data/mixed_context46/annot_melody3/',
    'image_folder': '/root/new_data/mixed_context46/image/',
    'update_lr_in_epoch': False,
    'sr_ratio': 6,
    'audio_type': 'MIX',
    'multiple': True,
    'save_dir': './output_model/multiple/',
    'save_prefix': 'model_conv5_multi_train',
    'use_pretrained': False, # whether or not to use a pretrained model
    'pretrained_path': './output_model/multiple/model_conv5_multi_train_epoch18.pt',
  }

def config_cqt():
  # for cqt images & single output
  return {
    'net': 'conv5',
    'fusion_mode': 'no_fusion',
    'annot_folder': '/root/new_data/context46/annot/',
    'image_folder': '/root/new_data/context46/cqt_image/',
    'update_lr_in_epoch': False,
    'sr_ratio': 6,
    'audio_type': 'RAW',
    'multiple': False,
    'save_dir': './output_model/context46/cqt/',
    'save_prefix': 'model_conv5_cqt_train',
    'use_pretrained': True, # whether or not to use a pretrained model
    'pretrained_path': '/root/CS229-MusicTranscription/output_model/context46/cqt/model_conv5_cqt_train_epoch7.pt', # './output_model/context46/model_conv5_train_epoch10.pt',
  }

def config_context6():
  # for single output & smaller context
  return {
    'net': 'conv5',
    'fusion_mode': 'no_fusion',
    'annot_folder': '/root/new_data/context6/annot/',
    'image_folder': '/root/new_data/context6/image/',
    'update_lr_in_epoch': True,
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
    'update_lr_in_epoch': False,
    'sr_ratio': 6,
    'audio_type': 'RAW',
    'multiple': False,
    'save_dir': './output_model/context46/stacking/',
    'save_prefix': 'model_conv5_stacking_train',
    'use_pretrained': False, # whether or not to use a pretrained model
    'pretrained_path': '/root/CS229-MusicTranscription/output_model/context46/stacking/model_conv5_stacking_train_epoch6.pt',
  }


def config_early_fusion():
  # single output + concatenate mel & CQT after the first conv layer
  return {
    'net': 'conv5',
    'fusion_mode': 'early_fusion',
    'annot_folder': '/root/new_data/context46/annot/',
    'image_folder': '/root/new_data/context46/image/',
    'update_lr_in_epoch': False,
    'sr_ratio': 6,
    'audio_type': 'RAW',
    'multiple': False,
    'save_dir': './output_model/context46/early_fusion/',
    'save_prefix': 'model_conv5_early_fusion_train',
    'use_pretrained': False, # whether or not to use a pretrained model
    'pretrained_path': '/root/CS229-MusicTranscription/output_model/context46/early_fusion/model_conv5_early_fusion_train_epoch11.pt',
  }

def config_late_fusion():
  # single output + concatenate mel & CQT after the first conv layer
  return {
    'net': 'conv5',
    'fusion_mode': 'late_fusion',
    'annot_folder': '/root/new_data/context46/annot/',
    'image_folder': '/root/new_data/context46/image/',
    'update_lr_in_epoch': False,
    'sr_ratio': 6,
    'audio_type': 'RAW',
    'multiple': False,
    'save_dir': './output_model/context46/late_fusion/',
    'save_prefix': 'model_conv5_late_fusion_train',
    'use_pretrained': False, # whether or not to use a pretrained model
    'pretrained_path': '/root/CS229-MusicTranscription/output_model/context46/late_fusion/model_conv5_late_fusion_train_epoch6.pt',
  }

def config_mel_test():
  # for cqt images & single output
  return {
    'net': 'conv5',
    'fusion_mode': 'no_fusion',
    'annot_folder': '/root/new_data/context46/annot/',
    'image_folder': '/root/new_data/context46/image/',
    'update_lr_in_epoch': False,
    'sr_ratio': 6,
    'audio_type': 'RAW',
    'multiple': False,
    'save_dir': '/root/', # Shouldn't be used
    'save_prefix': 'model_conv5_mel_train',
    'use_pretrained': True, # whether or not to use a pretrained model
    'pretrained_path': '/root/CS229-MusicTranscription/output_model/context46/model_conv5_cqt_train_epoch7.pt', # './output_model/context46/model_conv5_train_epoch10.pt',
  }

def config_avg_test():
  # for cqt images & single output
  return {
    'net': 'conv5',
    'fusion_mode': 'no_fusion',
    'annot_folder': '/root/new_data/context46/annot/',
    'image_folder': '/root/new_data/context46/image/',
    'update_lr_in_epoch': False,
    'sr_ratio': 6,
    'audio_type': 'RAW',
    'multiple': False,
    'save_dir': '/root/', # Shouldn't be used
    'save_prefix': 'model_conv5_mel_train',
    'use_pretrained': True, # whether or not to use a pretrained model
    'pretrained_path_mel': '/root/CS229-MusicTranscription/output_model/context46/model_conv5_cqt_train_epoch7.pt', # './output_model/context46/model_conv5_train_epoch10.pt',
    'pretrained_path_cqt': '/root/CS229-MusicTranscription/output_model/context46/cqt/model_conv5_cqt_train_epoch9.pt',
  }

def config_mel_conv3():
  # for cqt images & single output
  return {
    'net': 'conv3',
    'fusion_mode': 'no_fusion',
    'annot_folder': '/root/new_data/context46/annot/',
    'image_folder': '/root/new_data/context46/image/',
    'update_lr_in_epoch': False,
    'sr_ratio': 6,
    'audio_type': 'RAW',
    'multiple': False,
    'save_dir': 'output_model/context46/conv3/',
    'save_prefix': 'model_conv3_mel_train',
    'use_pretrained': False, # whether or not to use a pretrained model
    'pretrained_path': '/root/CS229-MusicTranscription/output_model/context46/model_conv5_cqt_train_epoch7.pt', # './output_model/context46/model_conv5_train_epoch10.pt',
  }

def config_mel_conv3_fc():
  # for cqt images & single output
  return {
    'net': 'conv3_fc',
    'fusion_mode': 'no_fusion',
    'annot_folder': '/root/new_data/context46/annot/',
    'image_folder': '/root/new_data/context46/image/',
    'update_lr_in_epoch': False,
    'sr_ratio': 6,
    'audio_type': 'RAW',
    'multiple': False,
    'save_dir': 'output_model/context46/conv3_fc/',
    'save_prefix': 'model_conv3_fc_mel_train',
    'use_pretrained': False, # whether or not to use a pretrained model
    'pretrained_path': '/root/CS229-MusicTranscription/output_model/context46/model_conv5_cqt_train_epoch7.pt', # './output_model/context46/model_conv5_train_epoch10.pt',
  }

def config_mel_bin():
  # for cqt images & single output
  return {
    'net': 'conv5_bin',
    'fusion_mode': 'no_fusion',
    'annot_folder': '/root/new_data/context46/annot/',
    'image_folder': '/root/new_data/context46/image/',
    'update_lr_in_epoch': False,
    'sr_ratio': 6,
    'audio_type': 'RAW',
    'multiple': False,
    'save_dir': 'output_model/context46/conv5_bin/',
    'save_prefix': 'model_conv5_mel_train',
    'use_pretrained': True, # whether or not to use a pretrained model
    'pretrained_path': '/root/CS229-MusicTranscription/output_model/context46/model_conv5_train_epoch7.pt', # './output_model/context46/model_conv5_train_epoch10.pt',
    'use_pretrained_bin': True, # this is not used though; main() should always load a pretrained bin classifier
    'pretrained_bin_path': '/root/CS229-MusicTranscription/output_model/context46/conv5_bin/model_conv5_bin_mel_train_epoch1.pt',
  }
