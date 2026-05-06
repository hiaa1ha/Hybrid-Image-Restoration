"""
select_model.py

Model factory for ARANet-B. Routes the model key from the option JSON
to the correct training wrapper class.
"""


def define_Model(opt):
    model = opt['model']

    # ----------------------------------------
    # ARANet-B uses the plain single-generator training wrapper
    # ----------------------------------------
    if model == 'aranet_b':
        from models.model_aranet_b import ModelARANetB as M

    else:
        raise NotImplementedError('Model [{:s}] is not defined.'.format(model))

    m = M(opt)
    print('Training model [{:s}] is created.'.format(m.__class__.__name__))
    return m
