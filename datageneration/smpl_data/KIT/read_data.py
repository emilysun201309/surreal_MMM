import argparse
import os
import json
import xml.etree.cElementTree as ET
import logging

import numpy as np


def parse_motions(path):
    xml_tree = ET.parse(path)
    xml_root = xml_tree.getroot()
    xml_motions = xml_root.findall('Motion')
    motions = []

    if len(xml_motions) > 1:
        logging.warn('more than one <Motion> tag in file "%s", only parsing the first one', path)
    motions.append(_parse_motion(xml_motions[0], path))
    return motions


def _parse_motion(xml_motion, path):
    xml_joint_order = xml_motion.find('JointOrder')
    if xml_joint_order is None:
        raise RuntimeError('<JointOrder> not found')

    joint_names = []
    joint_indexes = []
    for idx, xml_joint in enumerate(xml_joint_order.findall('Joint')):
        name = xml_joint.get('name')
        if name is None:
            raise RuntimeError('<Joint> has no name')
        joint_indexes.append(idx)
        name,_ = name.split("_")
        if(name[-1:] == 'x'):
            joint_names.append((name[:-1],0))
        elif(name[-1:] == 'y'):
            joint_names.append((name[:-1],1))
        elif(name[-1:] == 'z'):
            joint_names.append((name[:-1],2))
        else:
            joint_names.append((name[:-1],name[-1:]))

    joint_frames = []
    root_pos_frames = []
    root_rot_frames = []
    xml_frames = xml_motion.find('MotionFrames')
    if xml_frames is None:
        raise RuntimeError('<MotionFrames> not found')
    for xml_frame in xml_frames.findall('MotionFrame'):
        print('frame',_parse_frame(xml_frame, joint_indexes)[1])
        joint_frames.append(_parse_frame(xml_frame, joint_indexes)[0])
        root_pos_frames.append(_parse_frame(xml_frame, joint_indexes)[1])
        root_rot_frames.append(_parse_frame(xml_frame, joint_indexes)[2])

    return joint_names, joint_frames,root_pos_frames,root_rot_frames


def _parse_frame(xml_frame, joint_indexes):
    n_joints = len(joint_indexes)
    xml_joint_pos = xml_frame.find('JointPosition')
    xml_root_pos = xml_frame.find('RootPosition')
    xml_root_rot = xml_frame.find('RootRotation')
    if xml_joint_pos is None:
        raise RuntimeError('<JointPosition> not found')
    joint_pos = _parse_list(xml_joint_pos, n_joints, joint_indexes)
    root_pos = _parse_list(xml_root_pos, 3)
    root_rot = _parse_list(xml_root_rot, 3)
    return joint_pos,root_pos,root_rot


def _parse_list(xml_elem, length, indexes=None):
    if indexes is None:
        indexes = range(length)
    elems = [float(x) for idx, x in enumerate(xml_elem.text.rstrip().split(' ')) if idx in indexes]
    if len(elems) != length:
        raise RuntimeError('invalid number of elements')
    return elems


def main():
    #input_path = args.input
    input_path = 'samples/'
    print('Scanning files ...')
    files = [f for f in os.listdir(input_path) if os.path.isfile(os.path.join(input_path, f)) and f[0] != '.']
    basenames = list(set([os.path.splitext(f)[0].split('_')[0] for f in files]))
    print('done, {} potential motions and their annotations found'.format(len(basenames)))
    print('')

    # Parse all files.
    print('Processing data in "{}" ...'.format(input_path))
    all_ids = []
    all_motions = []
    all_annotations = []
    all_metadata = []
    reference_joint_names = None

    all_root_pos = []
    all_root_rot = []
    for idx, basename in enumerate(basenames):
        print('  {}/{} ...'.format(idx + 1, len(basenames))),

        # Load motion.
        mmm_path = os.path.join(input_path, basename + '_mmm.xml')
        assert os.path.exists(mmm_path)
        joint_names, joint_frames,root_pos_frames,root_rot_frames = parse_motions(mmm_path)[0]
        if reference_joint_names is None:
            reference_joint_names = joint_names[:]
        elif reference_joint_names != joint_names:
            print('skipping, invalid joint_names {}'.format(joint_names))
            continue
        print(reference_joint_names)
        # Load annotation.
        '''
        annotations_path = os.path.join(input_path, basename + '_annotations.json')
        assert os.path.exists(annotations_path)
        with open(annotations_path, 'r') as f:
            annotations = json.load(f)

        # Load metadata.
        meta_path = os.path.join(input_path, basename + '_meta.json')
        assert os.path.exists(meta_path)
        with open(meta_path, 'r') as f:
            meta = json.load(f)

        assert len(annotations) == meta['nb_annotations']
        '''
        all_ids.append(int(basename))
        all_motions.append(np.array(joint_frames, dtype='float32'))
        #all_root_pos.append(np.array(root_pos_frames,dtype='float32'))
        #all_root_rot.append(np.array(root_rot_frames,dtype='float32'))
        root_pos_frames = np.array(root_pos_frames,dtype='float32')
        root_rot_frames = np.array(root_rot_frames,dtype='float32')
        np.save('01_root_pos.npy',root_pos_frames)
        np.save('01_root_rot.npy',root_rot_frames)
        #all_annotations.append(annotations)
        #all_metadata(meta)
        print('done')
    all_m = []
    for f in joint_frames:    
        all_joint_motions = zip(reference_joint_names, f)
        all_joint_motions = list(all_joint_motions)
        motions = []
        temp = np.zeros(3)
        j_prev = all_joint_motions[0][0][0]
        for ((j,l),p) in all_joint_motions:
            if j not in ['RMro','LMro','RF','LF']:
                if j != j_prev:
                    motions.append(temp)
                    temp = np.zeros(3)
                    j_prev = j
                if l == 0:
                    temp[0] = p
                elif l == 1:
                    temp[1] = p
                elif l == 2:
                    temp[2] = p
        motions.append(temp)
        all_m.append(motions)
    
    #print(all_joint_motions)
    #print(all_ids,all_motions)
    #assert len(all_motions) == len(all_annotations)
    assert len(all_motions) == len(all_ids)
    print('done, successfully processed {} motions and their annotations'.format(len(all_motions)))
    print('')

    all_m = np.asarray(all_m)
    print(all_m[0])
    np.save('01.npy', all_m)
    
    # At this point, you can do anything you want with the motion and annotation data.


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    #parser.add_argument('input', type=str)
    main()