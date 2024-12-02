

affine_default_config = {
    'affine_probability': 1,
    'min_rotation': -15, 
    'max_rotation': 15,
    'min_scale_x': 0.9, 
    'max_scale_x': 1.1,
    'min_scale_y': 0.9,
    'max_scale_y': 1.1,
    'min_shear_x': -5,  
    'max_shear_x': 5,
    'min_shear_y': -5,
    'max_shear_y': 5,
    'min_translate_x': -50,
    'max_translate_x': 50,
    'min_translate_y': -50,
    'max_translate_y': 50,
    'num_affines_per_image': 5,
    'max_clipped_area_per_category': None, 
    'random_seed': 42,
    'enable_affine': True,
    'visualize_overlays': True,
    'output_visualizations_dir': 'visualize/visualize_affine',
    'output_images_dir': 'raw_images/augmented_images_affine',
}

# augflow/utils/configs.py

crop_default_config = {
    'modes': ['targeted'],  # Default to targeted mode
    'focus_categories': [],  # Empty by default
    'num_crops_per_image': 3,
    'margin': 50,  # Margin around focus polygons
    'min_crop_size': 256,
    'desired_output_size': None,  # Can be set to (width, height)
    'clipping_parameters': {
        'clipping_mode': 'constant',  # Options: 'constant', 'reflect', 'edge'
        'padding_color': (0, 0, 0),  # Default padding color (black)
    },
    'crop_modes': ['fixed_size', 'random_area'],  # Modes for non-targeted cropping
    'crop_size_parameters': {
        'fixed_size': {
            'crop_width': 800,
            'crop_height': 800,
        },
        'random_area': {
            'min_area_ratio': 0.5,
            'max_area_ratio': 0.9,
        },
    },
    'aspect_ratio_parameters': {
        'preserve_aspect_ratio': False,
        'target_aspect_ratio': (1, 1),
    },
    'overlap_parameters': {
        'max_overlap': 0.5,  # Maximum allowed overlap between crops
    },
    'max_clipped_area_per_category': None,  # Will be set to default in code if not provided
    'random_seed': 42,
    'enable_cropping': True,
    'visualize_overlays': True,
    'output_visualizations_dir': 'visualize/visualize_cropping',
    'output_images_dir': 'raw_images/augmented_images_cropping',
}



cutout_default_config = {
    'modes': ['non_targeted'],  # Default to non-targeted mode
    'focus_categories': [],  # No focus categories by default
    'cutout_probability': 1.0,
    'num_augmented_images': 5,
    'num_cutouts_per_image': 1,
    'cutout_size_percent': ((0.1, 0.2), (0.1, 0.2)),  # (height_percent_range, width_percent_range)
    'margin_percent': 0.05,
    'max_shift_percent': 1,
    'shift_steps': 20,
    'max_clipped_area_per_category': None,  # Will be set to default in code
    'random_seed': 42,
    'enable_cutout': True,
    'visualize_overlays': True,
    'output_visualizations_dir': 'visualize/visualize_cutout',
    'output_images_dir': 'raw_images/augmented_images_cutout',
}


# cutout_default_config = {
#     'cutout_probability': 1.0,
#     'cutout_size': (300, 300),     # (height, width)
#     'num_cutouts': 1,              # Number of cutout regions per augmentation
#     'cutout_p': 1.0,               # Probability to apply each cutout
#     'output_images_dir': 'raw_images/augmented_images_cutout',
#     'visualize_overlays': True,
#     'output_visualizations_dir': 'visualize/visualize_cutout',
#     'max_clipped_area_per_category': None,  # {category_id: max_allowed_reduction}
#     'random_seed': 42,
#     'enable_cutout': True,
#     'systematic_cutout': False,
#     'systematic_cutout_mode': 'largest',  # 'largest' or 'smallest'
#     'margin_pixels': 50,  # Margin in pixels around the polygon
# }

flip_default_config = {
    'flip_modes': ['both'],  # Options: 'horizontal', 'vertical', 'both'
    'num_flips_per_image': 1,
    'output_images_dir': 'raw_images/augmented_images_flip',
    'output_visualizations_dir': 'visualize/visualize_flip',
    'visualize_overlays': True,
    'max_clipped_area_per_category': {},  # Not typically needed for flipping
    'random_seed': 42,
    'enable_flipping': True,
}



mosaic_default_config = {
    'output_size': None,  # Allow output_size to be None
    'grid_size': (2, 2),
    'num_mosaics': 500,  # Number of unique mosaics to generate
    'random_seed': 42,
    'enable_mosaic': True,
    'max_allowed_area_reduction_per_category': {},  # {category_id: max_allowed_reduction}
    'randomize_positions': True,
    'filter_scale': 0,  # Minimum size for annotations to keep (in pixels)
    'max_attempts_per_cell': 100,
    'output_images_dir': 'raw_images/augmented_images_mosaic',
    'allow_significant_area_reduction_due_to_scaling':True,
    'visualize_overlays': True,
    'output_visualizations_dir':'visualize/visualize_mosaic',
    'max_usage_per_image': 1,  # Parameter to limit image reuse
    # New parameters for random offsets
    'max_offset_x': 0.2,  # As a fraction of cell width
    'max_offset_y': 0.2,  # As a fraction of cell height
}


# rotate_default_config

# configs.py

rotate_default_config = {
    'modes': ['non_targeted'],  # Default to non-targeted mode
    'focus_categories': [],  # Empty by default
    'rotation_probability': 1.0,
    'rotation_point_modes': ['center'],  # Default rotation point
    'rotation_angle_modes': ['predefined_set'],
    'angle_parameters': {
        'predefined_set': [-45, 45],
        'alpha': 45
    },
    'num_rotations_per_image': 3,
    'max_clipped_area_per_category': None,  # Will be set to default in code if not provided
    'random_seed': 42,
    'enable_rotation': True,
    'visualize_overlays': True,
    'output_visualizations_dir': 'visualize/visualize_rotation',
    'output_images_dir': 'raw_images/augmented_images_rotation',
}



scale_default_config = {
    'scale_mode': 'uniform',  # 'uniform', 'non_uniform', 'range_random', 'range_step', 'list'
    'scale_factors': [0.8, 1.0, 1.2],  # Used if mode == 'list' or 'uniform'
    'scale_factor_range': (0.8, 1.2),  # Used if mode == 'range_random'
    'scale_step': 0.1,  # Used if mode == 'range_step'
    'interpolation_methods': ['nearest', 'linear', 'cubic', 'area', 'lanczos4'],
    'preserve_aspect_ratio': True,
    'num_scales_per_image': 1,
    'output_images_dir': 'raw_images/augmented_images_scale',
    'output_visualizations_dir': 'visualize/visualize_scale',
    'visualize_overlays': True,
    'max_clipped_area_per_category': {},  # {category_id: max_allowed_reduction}
    'random_seed': 42,
    'enable_scaling': True,
}


shear_default_config = {
    'shear_probability': 0.8,
    'min_shear_x': -15,
    'max_shear_x': 15,
    'min_shear_y': -15,
    'max_shear_y': 15,
    'num_shears_per_image': 1,
    'max_clipped_area_per_category': None,  # {category_id: max_allowed_reduction}
    'random_seed': 42,
    'enable_shearing': True,
    'visualize_overlays': True,
    'output_visualizations_dir': 'visualize/visualize_shear',
    'output_images_dir': 'raw_images/augmented_images_shear',
}


translate_default_config = {
    'modes': ['non_targeted'],  # Default to non-targeted mode
    'focus_categories': [],  # Empty by default
    'translate_probability': 1.0,
    'min_translate_x': -0.2,  # Minimum translation in x direction as a fraction of image width
    'max_translate_x': 0.2,   # Maximum translation in x direction as a fraction of image width
    'min_translate_y': -0.2,  # Minimum translation in y direction as a fraction of image height
    'max_translate_y': 0.2,   # Maximum translation in y direction as a fraction of image height
    'num_translations_per_image': 1,
    'max_clipped_area_per_category': None,  # Will be set to default in code if not provided
    'random_seed': 42,
    'enable_translation': True,
    'visualize_overlays': True,
    'output_visualizations_dir': 'visualize/visualize_translation',
    'output_images_dir': 'raw_images/augmented_images_translation',
}

