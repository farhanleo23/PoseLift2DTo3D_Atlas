import numpy as np

from common.arguments import parse_args
from common.camera import *
from common.loss import *
from common.generators import UnchunkedGenerator

import acl

import os
import sys
sys.path.append('..')
from model_processor import ModelProcessor
from acl_resource import AclResource

args = parse_args()
print(args)


model_path = 'videopose3d_1f.om'
acl_resource = AclResource()
acl_resource.init()
    
model_parameters = {'model_dir': model_path}
model_processor = ModelProcessor(acl_resource, model_parameters)
model = model_processor.model
print(model)

print('Loading dataset...')
dataset_path = 'data/data_3d_' + args.dataset + '.npz'
if args.dataset == 'h36m':
    from common.h36m_dataset import Human36mDataset
    dataset = Human36mDataset(dataset_path)
elif args.dataset.startswith('custom'):
    from common.custom_dataset import CustomDataset
    dataset = CustomDataset('data/data_2d_' + args.dataset + '_' + args.keypoints + '.npz')
else:
    raise KeyError('Invalid dataset')

print('Preparing data...')
for subject in dataset.subjects():
    for action in dataset[subject].keys():
        anim = dataset[subject][action]
        
        if 'positions' in anim:
            positions_3d = []
            for cam in anim['cameras']:
                pos_3d = world_to_camera(anim['positions'], R=cam['orientation'], t=cam['translation'])
                pos_3d[:, 1:] -= pos_3d[:, :1] # Remove global offset, but keep trajectory in first position
                positions_3d.append(pos_3d)
            anim['positions_3d'] = positions_3d

print('Loading 2D detections...')
keypoints = np.load('data/data_2d_' + args.dataset + '_' + args.keypoints + '.npz', allow_pickle=True)
keypoints_metadata = keypoints['metadata'].item()
keypoints_symmetry = keypoints_metadata['keypoints_symmetry']
kps_left, kps_right = list(keypoints_symmetry[0]), list(keypoints_symmetry[1])
joints_left, joints_right = list(dataset.skeleton().joints_left()), list(dataset.skeleton().joints_right())
keypoints = keypoints['positions_2d'].item()

for subject in dataset.subjects():
    assert subject in keypoints, 'Subject {} is missing from the 2D detections dataset'.format(subject)
    for action in dataset[subject].keys():
        assert action in keypoints[subject], 'Action {} of subject {} is missing from the 2D detections dataset'.format(action, subject)
        if 'positions_3d' not in dataset[subject][action]:
            continue
            
        for cam_idx in range(len(keypoints[subject][action])):
            
            # We check for >= instead of == because some videos in H3.6M contain extra frames
            mocap_length = dataset[subject][action]['positions_3d'][cam_idx].shape[0]
            assert keypoints[subject][action][cam_idx].shape[0] >= mocap_length
            
            if keypoints[subject][action][cam_idx].shape[0] > mocap_length:
                # Shorten sequence
                keypoints[subject][action][cam_idx] = keypoints[subject][action][cam_idx][:mocap_length]

        assert len(keypoints[subject][action]) == len(dataset[subject][action]['positions_3d'])
        
for subject in keypoints.keys():
    for action in keypoints[subject]:
        for cam_idx, kps in enumerate(keypoints[subject][action]):
            # Normalize camera frame
            cam = dataset.cameras()[subject][cam_idx]
            kps[..., :2] = normalize_screen_coordinates(kps[..., :2], w=cam['res_w'], h=cam['res_h'])
            keypoints[subject][action][cam_idx] = kps

subjects_train = args.subjects_train.split(',')
subjects_semi = [] if not args.subjects_unlabeled else args.subjects_unlabeled.split(',')
if not args.render:
    subjects_test = args.subjects_test.split(',')
else:
    subjects_test = [args.viz_subject]

action_filter = None if args.actions == '*' else args.actions.split(',')
if action_filter is not None:
    print('Selected actions:', action_filter)

filter_widths = [int(x) for x in args.architecture.split(',')]
pad = [ filter_widths[0] // 2 ]

causal_shift = [ (filter_widths[0]) // 2 if args.causal else 0 ]
next_dilation = filter_widths[0]
for i in range(1, len(filter_widths)):
    pad.append((filter_widths[i] - 1)*next_dilation // 2)
    causal_shift.append((filter_widths[i]//2 * next_dilation) if args.causal else 0)
    next_dilation *= filter_widths[i]

frames = 0
for f in pad:
    frames += f
receptive_field = 1 + 2*frames
pad = (receptive_field - 1)//2
if args.causal:
    print('INFO: Using causal convolutions')
    causal_shift = pad
else:
    causal_shift = 0


if args.render:
    print('Rendering...')
    
    input_keypoints = keypoints[args.viz_subject][args.viz_action][args.viz_camera].copy()
    ground_truth = None
    if args.viz_subject in dataset.subjects() and args.viz_action in dataset[args.viz_subject]:
        if 'positions_3d' in dataset[args.viz_subject][args.viz_action]:
            ground_truth = dataset[args.viz_subject][args.viz_action]['positions_3d'][args.viz_camera].copy()
    if ground_truth is None:
        print('INFO: this action is unlabeled. Ground truth will not be rendered.')
        
    gen = UnchunkedGenerator(None, None, [input_keypoints],
                             pad=pad, causal_shift=causal_shift, augment=args.test_time_augmentation,
                             kps_left=kps_left, kps_right=kps_right, joints_left=joints_left, joints_right=joints_right)
    
    new = None
    prediction = np.array([])
        
    for _, batch, batch_2d in gen.next_epoch():
        i = 0
        while(i+243 <= batch_2d.shape[1]):
            inputs_2d = batch_2d[:,i:i+243,:,:].astype('float32')
            predicted_3d_pos = model.execute([inputs_2d])[0]
            i = i + 1

            # Test-time augmentation (if enabled)
            if gen.augment_enabled():
                # Undo flipping and take average with non-flipped version
                predicted_3d_pos[1, :, :, 0] *= -1
                predicted_3d_pos[1, :, joints_left + joints_right] = predicted_3d_pos[1, :, joints_right + joints_left]
                predicted_3d_pos = np.mean(predicted_3d_pos, axis=0, keepdims=True)
                
            
            output = predicted_3d_pos.squeeze(axis = 0)    
            if(new == None):
                prediction = output
                new = 0
            else:
                prediction = np.concatenate((prediction , output) , axis = 0)  

    if args.viz_output is not None:

        # Invert camera transformation
        cam = dataset.cameras()[args.viz_subject][args.viz_camera]
        
        # If the ground truth is not available, take the camera extrinsic params from a random subject.
        # They are almost the same, and anyway, we only need this for visualization purposes.
        for subject in dataset.cameras():
            if 'orientation' in dataset.cameras()[subject][args.viz_camera]:
                rot = dataset.cameras()[subject][args.viz_camera]['orientation']
                break
        prediction = camera_to_world(prediction, R=rot, t=0)
        # We don't have the trajectory, but at least we can rebase the height
        prediction[:, :, 2] -= np.min(prediction[:, :, 2])
        
        anim_output = {'Reconstruction': prediction}
        
        input_keypoints = image_coordinates(input_keypoints[..., :2], w=cam['res_w'], h=cam['res_h'])
        
        from common.visualization import render_animation
        render_animation(input_keypoints, keypoints_metadata, anim_output,
                         dataset.skeleton(), dataset.fps(), args.viz_bitrate, cam['azimuth'], args.viz_output,
                         limit=args.viz_limit, downsample=args.viz_downsample, size=args.viz_size,
                         input_video_path=args.viz_video, viewport=(cam['res_w'], cam['res_h']),
                         input_video_skip=args.viz_skip)
