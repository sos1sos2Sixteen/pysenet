from estimator import partNames, partIDs

_regions = [
    [
        ['point', 'nose'],
        ['line', ('nose', 'leftShoulder')],
        ['line', ('nose', 'rightShoulder')],
        ['line', ('leftShoulder', 'leftHip')],
        ['line', ('rightShoulder', 'rightHip')]
    ],  # others, none participants in evaluation
    [
        ['point', 'leftEye'],
        ['point', 'rightEye'],
        ['line', ('leftEye', 'rightEye')]
    ],  # head
    [
        ['point', 'leftShoulder'],
        ['point', 'leftElbow'],
        ['point', 'leftWrist'],
        ['line', ('leftShoulder', 'leftElbow')],
        ['line', ('leftElbow', 'leftWrist')]
    ],  # left arm
    [
        ['point', 'rightShoulder'],
        ['point', 'rightElbow'],
        ['point', 'rightWrist'],
        ['line', ('rightShoulder', 'rightElbow')],
        ['line', ('rightElbow', 'rightWrist')]
    ],  # right arm
    [
        ['point', 'leftHip'],
        ['point', 'leftKnee'],
        ['point', 'leftAnkle'],
        ['line', ('leftHip', 'leftKnee')],
        ['line', ('leftKnee', 'leftAnkle')]
    ],  # left leg
    [
        ['point', 'rightHip'],
        ['point', 'rightKnee'],
        ['point', 'rightAnkle'],
        ['line', ('rightHip', 'rightKnee')],
        ['line', ('rightKnee', 'rightAnkle')]
    ],  # right leg
    [
        ['line', ('leftShoulder', 'rightShoulder')]
    ],  # Shoulder line
    [
        ['line', ('leftHip', 'rightHip')]
    ],  # hip line
]


def transform_part_point(point):
    return [False, partIDs[point[1]]]

def transform_part_line(line):
    (genos, talos) = line[1]
    return [True, (partIDs[genos], partIDs[talos])]

def transform_feature(ft):
    if ft[0] == 'point':
        return transform_part_point(ft)
    else:
        return transform_part_line(ft)

def transform_region(region):
    res = []
    for ft in region:
        res.append(transform_feature(ft))
    return res

regions = []
for reg in _regions:
    regions.append(transform_region(reg))
