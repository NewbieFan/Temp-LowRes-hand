
def get_num_layer_for_vit(var_name, num_max_layer):
    if var_name in ('backbone.cls_token', 'backbone.mask_token',
                    'backbone.pos_embed'):
        return 0
    elif var_name.startswith('backbone.patch_embed'):
        return 0
    elif var_name.startswith('backbone.conv_embed'):
        return 0
    # or var_name.startswith('backbone.seqblocks')
    elif var_name.startswith('backbone.blocks') :
        layer_id = int(var_name.split('.')[2])
        return layer_id
    else:
        return num_max_layer - 1

def get_optimizer_grouped_params(model, layerwise_decay_rate, lr, weight_decay=0.1, num_layers=12):
    parameter_groups = {}

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue  # frozen weights
        if (len(param.shape) == 1 or name.endswith('.bias')
                or 'pos_embed' in name):
            group_name = 'no_decay'
            this_weight_decay = 0.
        else:
            group_name = 'decay'
            this_weight_decay = weight_decay
        layer_id = get_num_layer_for_vit(name, num_layers)
        group_name = 'layer_%d_%s' % (layer_id, group_name)

        if group_name not in parameter_groups:
            scale = layerwise_decay_rate**(num_layers - layer_id - 1)

            parameter_groups[group_name] = {
                'weight_decay': this_weight_decay,
                'params': [],
                'param_names': [],
                'lr_scale': scale,
                'group_name': group_name,
                'lr': scale * lr,
            }

        parameter_groups[group_name]['params'].append(param)
        parameter_groups[group_name]['param_names'].append(name)

    return parameter_groups