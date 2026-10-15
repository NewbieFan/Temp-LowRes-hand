
optimizer = dict(lr=1e-4, weight_decay=0.01,
                 paramwise_cfg=dict(num_layers=12, layer_decay_rate=0.85))

optimizer_config = dict(grad_clip=dict(max_norm=1., norm_type=2))

# model settings
model = dict(
    backbone=dict(
        img_size=(64, 64),
        patch_size=16,
        embed_dim=768,
        depth=12,
        num_heads=12,
        ratio=4,
        use_checkpoint=False,
        mlp_ratio=4,
        qkv_bias=True,
        drop_path_rate=0.3,
    ),
    keypoint_head=dict(
        in_channels=768, 
        num_FMs=8, 
        num_joints=21, 
        num_verts=778, 
        feature_shape=[8,8], 
        heatmap_shape=[32,32]))
