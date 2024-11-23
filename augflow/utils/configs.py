

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

crop_default_config = {
    'desired_output_size': None,  # (width, height) or None
    'crop_modes': ['fixed_size', 'random_area', 'random_aspect', 'grid', 'sliding_window', 'predefined'],
    'crop_size_parameters': {
        'fixed_size': {
            'crop_width': 800,
            'crop_height': 800
        },
        'random_area': {
            'min_area_ratio': 0.5,
            'max_area_ratio': 0.9
        },
        'random_aspect': {
            'min_aspect_ratio': 0.75,
            'max_aspect_ratio': 1.33
        },
        'sliding_window': {
            'window_size': (400, 400),
            'step_size': (200, 200)
        },
        'grid': {
            'grid_rows': 2,
            'grid_cols': 2
        }
    },
    'num_crops_per_image': 3,
    'predefined_crops': None,  # {image_id: [(x, y, w, h), ...]}
    'clipping_parameters': {
        'clipping_mode': 'pad',  # 'pad', 'resize_crop', 'ignore'
        'padding_color': (0, 0, 0)  # BGR
    },
    'max_clipped_area_per_category': {},  # {category_id: max_allowed_reduction}
    'aspect_ratio_parameters': {
        'preserve_aspect_ratio': True,
        'target_aspect_ratio': (1, 1)  # Tuple (width, height)
    },
    'overlap_parameters': {
        'max_overlap': 0.3
    },
    'random_seed': 42,
    'enable_cropping': True,
    'visualize_overlays': True,
    'output_visualizations_dir': 'visualize/visualize_crop',
    'output_images_dir': 'raw_images/augmented_images_crop',
    'systematic_crops': False,
    'systematic_crops_parameters': {
        'mode': 'largest',  # 'largest' or 'categories'
        'categories': [],    # List of category IDs to focus on if mode is 'categories'
        'margin': 100         # Margin in pixels around the polygon
    },
    'min_crop_size': 256  # Minimum allowed dimension (width or height) for a crop
}


cutout_default_config = {
    'cutout_probability': 1.0,
    'cutout_size': (300, 300),     # (height, width)
    'num_cutouts': 1,              # Number of cutout regions per augmentation
    'cutout_p': 1.0,               # Probability to apply each cutout
    'output_images_dir': 'raw_images/augmented_images_cutout',
    'visualize_overlays': True,
    'output_visualizations_dir': 'visualize/visualize_cutout',
    'max_clipped_area_per_category': None,  # {category_id: max_allowed_reduction}
    'random_seed': 42,
    'enable_cutout': True,
    'systematic_cutout': False,
    'systematic_cutout_mode': 'largest',  # 'largest' or 'smallest'
    'margin_pixels': 50,  # Margin in pixels around the polygon
}

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


rotate_default_config = {
    'rotation_probability': 0.8,
    'rotation_point_modes': ['center','random'],  # 'center', 'random'
    'rotation_angle_modes': ['random_range'],  # 'predefined_set', 'random_range'
    'angle_parameters': {
        'random_range': (-30, 30),  # Min and max angles
        'predefined_set': [90, 180, 270]  # Specific angles
    },
    'num_rotations_per_image': 5,
    'max_clipped_area_per_category': None,  # {category_id: max_allowed_reduction}
    'random_seed': 42,
    'enable_rotation': True,
    'visualize_overlays': True,
    'output_visualizations_dir': 'visualize/visualize_rotate',
    'output_images_dir': 'raw_images/augmented_images_rotate',
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
    'translate_probability': 1,
    'min_translate_x': -0.35,  # -35% of image width
    'max_translate_x': 0.35,   # +35% of image width
    'min_translate_y': -0.35,  # -35% of image height
    'max_translate_y': 0.35,   # +35% of image height
    'num_translations_per_image': 1,
    'max_clipped_area_per_category': None,  # {category_id: max_allowed_reduction}
    'random_seed': 42,
    'enable_translation': True,
    'visualize_overlays': True,
    'output_visualizations_dir': 'visualize/visualize_translate',
    'output_images_dir': 'raw_images/augmented_images_translate',
}

