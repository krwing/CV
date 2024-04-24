import torch

def load_state_dict(model, state_dict):
    """Load state_dict into model, handling DataParallel and DistributedDataParallel. Also checks for "model" key in state_dict.

    DataParallel prefixes state_dict keys with 'module.' when saving.
    If the model is not a DataParallel model but the state_dict is, then prefixes are removed.
    If the model is a DataParallel model but the state_dict is not, then prefixes are added.
    """
    state_dict = state_dict.get('model', state_dict)
    # if model is a DataParallel model, then state_dict keys are prefixed with 'module.'

    do_prefix = isinstance(
        model, (torch.nn.DataParallel, torch.nn.parallel.DistributedDataParallel))
    state = {}
    for k, v in state_dict.items():
        if k.startswith('module.') and not do_prefix:
            k = k[7:]

        if not k.startswith('module.') and do_prefix:
            k = 'module.' + k

        state[k] = v

    model.load_state_dict(state, strict=False)
    print("Loaded successfully")
    return model


def load_wts(model, checkpoint_path):
    ckpt = torch.load(checkpoint_path, map_location='cpu')
    return load_state_dict(model, ckpt)


def load_state_dict_from_url(model, url, **kwargs):
    state_dict = torch.hub.load_state_dict_from_url(url, map_location='cpu', **kwargs)
    return load_state_dict(model, state_dict)


def load_state_from_resource(model, resource: str):
    """Loads weights to the model from a given resource. A resource can be of following types:
        1. URL. Prefixed with "url::"
                e.g. url::http(s)://url.resource.com/ckpt.pt

        2. Local path. Prefixed with "local::"
                e.g. local::/path/to/ckpt.pt


    Args:
        model (torch.nn.Module): Model
        resource (str): resource string

    Returns:
        torch.nn.Module: Model with loaded weights
    """
    print(f"Using pretrained resource {resource}")

    if resource.startswith('url::'):
        url = resource.split('url::')[1]
        return load_state_dict_from_url(model, url, progress=True)

    elif resource.startswith('local::'):
        path = resource.split('local::')[1]
        return load_wts(model, path)
        
    else:
        raise ValueError("Invalid resource type, only url:: and local:: are supported")
    