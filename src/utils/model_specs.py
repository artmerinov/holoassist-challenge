def model_size(model):
    # https://discuss.pytorch.org/t/finding-model-size/130275

    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()

    size_all_mb = (param_size + buffer_size) / 1024**2
    return size_all_mb

    # print('model size: {:.3f}MB'.format(size_all_mb))
