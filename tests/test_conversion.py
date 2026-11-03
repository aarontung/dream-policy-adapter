import base64
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from colosseum_policy_server import colosseum_pb2 as pb
from colosseum_policy_server.tensors import tensor_from_numpy
from dream_policy_adapter.adapter import DreamAdapter, observation_payload
from dream_policy_adapter.dream_policy import to_actions


def result():
    return {'per_key': {'action.left_arm': np.ones((48, 6)),
        'action.right_arm': np.full((48, 6), 2),
        'action.left_grip': np.full(48, .2), 'action.right_grip': np.full(48, .8)}}


def observation():
    obs = pb.Observation(instruction='put cup on plate', robot_time_ns=2_000_000_000)
    obs.state['joint_position'].CopyFrom(tensor_from_numpy(np.arange(12, dtype=np.float32)))
    obs.state['gripper_position'].CopyFrom(tensor_from_numpy(np.array([.2, .8], np.float32)))
    for name in ('head_image', 'left_image', 'right_image'):
        rgb = np.zeros((16, 16, 3), np.uint8)
        rgb[:, :, 0] = 255
        obs.sensors.add(sensor_id=name, encoding=pb.RAW_RGB, width=16, height=16, data=rgb.tobytes())
    return obs


def model():
    return SimpleNamespace(action_space='joint_position', action_dim=14, control_hz=30,
        max_horizon=48, endpoint='http://127.0.0.1:9120',
        backend_options={'adapter': 'dream_yam', 'contract_confirmed': True})


def test_output_order_and_gripper_inversion():
    a = to_actions(result())
    assert a.shape == (48, 14)
    assert a.dtype == np.float32
    np.testing.assert_allclose(a[0], [1]*6 + [.2] + [2]*6 + [.8])
    np.testing.assert_allclose(to_actions(result(), invert_grippers=True)[0, [6, 13]], [.8, .2])


@pytest.mark.parametrize('key,value', [
    ('action.left_arm', np.zeros((48, 7))),
    ('action.right_arm', np.zeros((47, 6))),
    ('action.left_grip', np.zeros((48, 1))),
    ('action.left_grip', np.full(48, 1.3)),
    ('action.right_arm', np.full((48, 6), np.nan)),
])
def test_bad_output(key, value):
    raw = result()
    raw['per_key'][key] = value
    with pytest.raises(ValueError):
        to_actions(raw)


def test_horizon_rejected_without_truncation():
    with pytest.raises(ValueError):
        to_actions(result(), max_horizon=47)


def test_observation_mapping_and_rgb():
    p = observation_payload(observation(), 30, True)
    assert p['timestamp'] == 2
    assert p['observation']['state.right_arm'] == list(range(6, 12))
    np.testing.assert_allclose(p['observation']['state.left_gripper'], [.8])
    assert set(p['frames']) == {'video.overhead', 'video.wrist_l', 'video.wrist_r'}
    bgr = cv2.imdecode(np.frombuffer(base64.b64decode(p['frames']['video.overhead']), np.uint8), cv2.IMREAD_COLOR)
    assert bgr[0, 0, 2] > 250 and bgr[0, 0, 0] < 5


@pytest.mark.parametrize('bad', ['missing_camera', 'duplicate_camera', 'bad_state', 'empty_instruction'])
def test_invalid_observation(bad):
    obs = observation()
    if bad == 'missing_camera':
        del obs.sensors[-1]
    elif bad == 'duplicate_camera':
        obs.sensors.add().CopyFrom(obs.sensors[0])
    elif bad == 'bad_state':
        obs.state['joint_position'].CopyFrom(tensor_from_numpy(np.full(12, np.inf)))
    else:
        obs.instruction = ''
    with pytest.raises(ValueError):
        observation_payload(obs, 30)


def test_contract_requires_confirmation():
    m = model()
    m.backend_options['contract_confirmed'] = False
    with pytest.raises(ValueError, match='Confirm'):
        DreamAdapter({}).validate(m)


def test_wrong_frequency_rejected():
    m = model()
    m.control_hz = 15
    with pytest.raises(ValueError):
        DreamAdapter({}).validate(m)


@pytest.mark.parametrize('invert', [False, True])
def test_small_gripper_overshoot_clipped_before_inversion(invert):
    raw = result()
    raw['per_key']['action.left_grip'][:] = -0.1
    raw['per_key']['action.right_grip'][:] = 1.2
    actions = to_actions(raw, invert_grippers=invert)
    np.testing.assert_array_equal(actions[:, 6], 1 if invert else 0)
    np.testing.assert_array_equal(actions[:, 13], 0 if invert else 1)
    np.testing.assert_array_equal(actions[:, :6], 1)
    np.testing.assert_array_equal(actions[:, 7:13], 2)
    assert raw['per_key']['action.left_grip'][0] == -0.1


@pytest.mark.parametrize('value', [-0.1001, 1.2001, -1, 100])
def test_large_gripper_overshoot_still_rejected(value, capsys):
    raw = result()
    raw['per_key']['action.right_grip'][:] = value
    with pytest.raises(ValueError, match='Dream grippers exceed'):
        to_actions(raw)
    assert 'right=[' in capsys.readouterr().out
