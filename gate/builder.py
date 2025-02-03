import models


def build_gate(cfg):
    param = dict()
    for key in cfg:
        if key == 'type':
            continue
        param[key] = cfg[key]

    model = models.gate.__dict__[cfg.type](**param)

    return model