_base_ = [
    '../../_base_/default_runtime.py'
]
plugin = True
plugin_dir = 'plugin/RLRFusion/'

# TODO: please change to your own data root!
data_root = '/home/data/nuscenes/'
max_epoch = 20
samples_per_gpu = 8
workers_per_gpu = 8
work_dir = None
load_from = None
resume_from = None

dataset_type = 'NuScenesDatasetRadar'
class_names = [
    'car', 'truck',  'bus', 'trailer',
    'motorcycle', 'bicycle', 'pedestrian', ]
point_cloud_range = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
voxel_size_radar = [0.8, 0.8, 8]
voxel_size = [0.1, 0.1, 0.2]
input_modality = dict(
    use_lidar=True,
    use_camera=False,
    use_radar=True,
    use_map=False,
    use_external=False)

model = dict(
    type='Centerpoint_early_rcs_cat',
    use_LiDAR=True,
    use_Cam=False,
    use_Radar=True,
    middle_fusion=True,  # Middle fusion True!!
    pts_voxel_layer=dict(
        max_num_points=15,
        point_cloud_range=point_cloud_range,
        voxel_size=voxel_size,
        max_voxels=(90000, 120000),
        deterministic=False),

    pts_voxel_layer_radar=dict(  # PointPillars voxelization
        max_num_points=64,
        point_cloud_range=point_cloud_range,
        voxel_size=voxel_size_radar,
        max_voxels=(90000, 120000)),

    pts_voxel_encoder=dict(type='HardSimpleVFE', num_features=5),

    pts_voxel_encoder_radar=dict(
        type='PillarFusionFeatureNet_point',
        in_channels=10,
        feat_channels=[32],
        with_distance=False,
        voxel_size=voxel_size_radar,
        point_cloud_range=point_cloud_range,
        norm_cfg=dict(type='BN1d', eps=1e-3, momentum=0.01),
        legacy=False),
    pts_middle_encoder=dict(
        type='SparseEncoder_out',
        in_channels=5,
        sparse_shape=[41, 1024, 1024],
        output_channels=128,
        order=('conv', 'norm', 'act'),
        encoder_channels=((16, 16, 32), (32, 32, 64), (64, 64, 128), (128,
                                                                      128)),
        encoder_paddings=((0, 0, 1), (0, 0, 1), (0, 0, [0, 1, 1]), (0, 0)),
        block_type='basicblock'),

    pts_middle_encoder_radar=dict(
        type='PointPillarsScatter', in_channels=32, output_shape=[128, 128]),  # 51.2 * 2 / 0.2 = 512

    middle_fusion_layer=dict(
        type='rcs_middle_fusion_gamma', in_channels=256),

    pts_backbone=dict(
        type='SECOND',
        in_channels=256,  # C*D
        out_channels=[128, 256],
        layer_nums=[5, 5],
        layer_strides=[1, 2],
        norm_cfg=dict(type='BN', eps=1e-3, momentum=0.01),
        conv_cfg=dict(type='Conv2d', bias=False)),

    fusion_head=dict(layers=3, in_channels=512 + 32, out_channels=512, ),

    pts_neck=dict(
        type='SECONDFPN',
        in_channels=[128, 256],
        out_channels=[256, 256],
        upsample_strides=[1, 2],
        norm_cfg=dict(type='BN', eps=0.001, momentum=0.01),
        upsample_cfg=dict(type='deconv', bias=False),
        use_conv_for_no_stride=True),
    pts_bbox_head=dict(
        type='CenterHead',
        in_channels=512,
        tasks=[
            dict(num_class=1, class_names=['car']),
            dict(num_class=1, class_names=['truck']),
            dict(num_class=2, class_names=['bus', 'trailer']),
            dict(num_class=2, class_names=['motorcycle', 'bicycle']),
            dict(num_class=1, class_names=['pedestrian'])
        ],
        common_heads=dict(
            reg=(2, 2), height=(1, 2), dim=(3, 2), rot=(2, 2), vel=(2, 2)),
        share_conv_channel=64,
        bbox_coder=dict(
            type='CenterPointBBoxCoder',
            post_center_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
            max_num=500,
            score_threshold=0.1,
            out_size_factor=8,
            voxel_size=[0.1, 0.1],
            code_size=9,
            pc_range=[-51.2, -51.2]),
        separate_head=dict(
            type='DCNSeparateHead',
            init_bias=-2.19,
            final_kernel=3,
            dcn_config=dict(
                type='DCN',
                in_channels=64,
                out_channels=64,
                kernel_size=3,
                padding=1,
                groups=4)),
        loss_cls=dict(type='GaussianFocalLoss', reduction='mean'),
        loss_bbox=dict(type='L1Loss', reduction='mean', loss_weight=0.25),
        norm_bbox=True),
    # model training and testing settings
    train_cfg=dict(pts=dict(
        grid_size=[1024, 1024, 40],
        voxel_size=voxel_size,
        point_cloud_range=point_cloud_range,
        out_size_factor=8,
        dense_reg=1,
        gaussian_overlap=0.1,
        max_objs=500,
        min_radius=2,
        code_weights=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.2, 0.2],)),
    test_cfg=dict(pts=dict(
        post_center_limit_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
        max_per_img=500,
        pc_range=[-51.2, -51.2],
        max_pool_nms=False,
        min_radius=[4, 12, 10, 0.85, 0.175],
        score_threshold=0.1,
        out_size_factor=8,
        voxel_size=[0.1, 0.1],
        nms_type='rotate',
        pre_max_size=1000,
        post_max_size=83,
        nms_thr=0.2)))


file_client_args = dict(backend='disk')

db_sampler = dict(
    data_root=data_root,
    #info_path=data_root + 'sun_dbinfos_train.pkl',
    info_path=data_root + 'radar_nuscenes_5sweeps_dbinfos_train.pkl',
    rate=1.0,
    prepare=dict(
        filter_by_difficulty=[-1],
        filter_by_min_points=dict(car=5,truck=5,bus=5,trailer=5,motorcycle=5,bicycle=5,pedestrian=5)),
    classes=class_names,
    sample_groups=dict(car=2,truck=3,bus=4,trailer=6,motorcycle=6,bicycle=6,pedestrian=2,),
    points_loader=dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=5,
        use_dim=[0, 1, 2, 3, 4],
        file_client_args=file_client_args))

radar_use_dims = [0, 1, 2, 5, 8, 9, 14, 15, 18]  # the last dim may be the the dt?

train_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=5,
        use_dim=5,
        file_client_args=file_client_args),
    dict(
        type='LoadRadarPointsMultiSweeps_lirafusion',
        load_dim=18,
        sweeps_num=4,
        use_dim=radar_use_dims,
        max_num=1200, ),
    dict(
        type='LoadPointsFromMultiSweeps',
        sweeps_num=9,
        use_dim=[0, 1, 2, 3, 4],
        file_client_args=file_client_args,
        pad_empty_sweeps=True,
        remove_close=True),
    dict(type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True),
    dict(type='ObjectSample', db_sampler=db_sampler),
    dict(
        type='GlobalRotScaleTrans_radar',
        rot_range=[-0.3925, 0.3925],
        scale_ratio_range=[0.95, 1.05],
        translation_std=[0, 0, 0]),
    dict(
        type='RandomFlip3D_radar',
        sync_2d=False,
        flip_ratio_bev_horizontal=0.5,
        flip_ratio_bev_vertical=0.5),
    dict(type='PointsRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='ObjectRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='ObjectNameFilter', classes=class_names),
    dict(type='PointShuffle'),
    dict(type='DefaultFormatBundle3D', class_names=class_names),
    dict(type='fuse_lidar_radar'),
    dict(type='Radar_format'),
    dict(type='Collect3D', keys=['points', 'gt_bboxes_3d', 'gt_labels_3d', 'radar', 'fused_points'])
]
test_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=5,
        use_dim=5,
        file_client_args=file_client_args),
    dict(
        type='LoadRadarPointsMultiSweeps_lirafusion',
        load_dim=18,
        sweeps_num=4,
        use_dim=radar_use_dims,
        max_num=1200, ),
    dict(
        type='LoadPointsFromMultiSweeps',
        sweeps_num=9,
        use_dim=[0, 1, 2, 3, 4],
        file_client_args=file_client_args,
        pad_empty_sweeps=True,
        remove_close=True),
    # revise
    dict(
        type='MultiScaleFlipAug3D',
        img_scale=(1333, 800),
        pts_scale_ratio=1,
        flip=False,
        transforms=[
            dict(
                type='GlobalRotScaleTrans',
                rot_range=[0, 0],
                scale_ratio_range=[1.0, 1.0],
                translation_std=[0, 0, 0]),
            dict(type='RandomFlip3D', sync_2d=False),
            dict(
                type='DefaultFormatBundle3D',
                class_names=class_names,
                with_label=False),
            dict(type='fuse_lidar_radar'),
            dict(type='Radar_format'),

            dict(type='Collect3D', keys=['points', 'radar', 'fused_points'])
        ])
]

# construct a pipeline for data and gt loading in show function
# please keep its loading function consistent with test_pipeline (e.g. client)
eval_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=5,
        use_dim=5,
        file_client_args=file_client_args),
    dict(
        type='LoadPointsFromMultiSweeps',
        sweeps_num=9,
        use_dim=[0, 1, 2, 3, 4],
        file_client_args=file_client_args,
        pad_empty_sweeps=True,
        remove_close=True),
    dict(
        type='DefaultFormatBundle3D',
        class_names=class_names,
        with_label=False),
    dict(type='Collect3D', keys=['points'])
]

data = dict(
    samples_per_gpu=samples_per_gpu,
    workers_per_gpu=workers_per_gpu,
    train=dict(
        type='CBGSDataset',
        classes=class_names,
        dataset=dict(
            type=dataset_type,
            data_root=data_root,
            ann_file=data_root + 'radar_nuscenes_5sweeps_infos_train_radar_coor_1.pkl',  # changed to new pkl file
            pipeline=train_pipeline,
            classes=class_names,
            modality=input_modality,
            test_mode=False,
            use_valid_flag=True,
            # we use box_type_3d='LiDAR' in kitti and nuscenes dataset
            # and box_type_3d='Depth' in sunrgbd and scannet dataset.
            box_type_3d='LiDAR'),
    ),

    val=dict(
        type=dataset_type,
        data_root=data_root,
        pipeline=test_pipeline, classes=class_names, modality=input_modality,
        #ann_file=data_root + 'weather_rain_val.pkl',
        ann_file=data_root + 'radar_nuscenes_5sweeps_infos_val_radar_coor_1.pkl',
        box_type_3d='LiDAR'),
    test=dict(
        type=dataset_type,
        data_root=data_root,
        pipeline=test_pipeline, classes=class_names, modality=input_modality,
        #ann_file=data_root + 'weather_rain_val.pkl', ),
        ann_file=data_root + 'radar_nuscenes_5sweeps_infos_val_radar_coor_1.pkl', ),
        box_type_3d='LiDAR')

optimizer = dict(type='AdamW', lr=0.0001, weight_decay=0.01)
# max_norm=10 is better for SECOND
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))

lr_config = dict(
    policy='cyclic',
    target_ratio=(10, 0.0001),
    cyclic_times=1,
    step_ratio_up=0.4)

evaluation = dict(interval=20, pipeline=eval_pipeline)
runner = dict(type='EpochBasedRunner', max_epochs=max_epoch)

find_unused_parameters = True
