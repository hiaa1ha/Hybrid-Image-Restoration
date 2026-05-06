"""
select_dataset.py

Dataset factory for ARANet-B. Routes the dataset_type key from the
option JSON to the correct dataset class.
"""


def define_Dataset(dataset_opt):
    dataset_type = dataset_opt['dataset_type'].lower()

    # ----------------------------------------
    # ARANet-B denoising dataset
    # ----------------------------------------
    if dataset_type in ['aranet_b', 'denoising']:
        from data.dataset_aranet_b import DatasetARANetB as D

    else:
        raise NotImplementedError('Dataset [{:s}] is not found.'.format(dataset_type))

    dataset = D(dataset_opt)
    print('Dataset [{:s} - {:s}] is created.'.format(dataset.__class__.__name__, dataset_opt['name']))
    return dataset
