"""Colosseum plugin. The worker owns all Dream sessions."""
import base64
import uuid
from urllib.parse import urlsplit

import cv2
import httpx
import numpy as np

from colosseum_policy_server import colosseum_pb2 as pb
from colosseum_policy_server.tensors import tensor_to_numpy
from .dream_policy import validate_actions


def observation_payload(observation, control_hz, invert_grippers=False):
    if not observation.instruction.strip():
        raise ValueError("Instruction is empty")
    states = {}
    for key, size in (("joint_position", 12), ("gripper_position", 2)):
        if key not in observation.state:
            raise ValueError(f"Missing state: {key}")
        value = np.asarray(tensor_to_numpy(observation.state[key]), dtype=np.float32).reshape(-1)
        if value.shape != (size,) or not np.isfinite(value).all():
            raise ValueError(f"Invalid state: {key}")
        states[key] = value
    joints, grips = states["joint_position"], states["gripper_position"]
    if np.any((grips < 0) | (grips > 1)):
        raise ValueError("Observation grippers must be in [0, 1]")
    if invert_grippers:
        grips = 1 - grips
    images = {im.sensor_id: im for im in observation.sensors}
    if len(images) != len(observation.sensors):
        raise ValueError("Duplicate sensor IDs")
    frames = {}
    for source, target in (("head_image", "video.overhead"), ("left_image", "video.wrist_l"), ("right_image", "video.wrist_r")):
        image = images.get(source)
        if (image is None or image.encoding != pb.RAW_RGB or image.width < 1 or image.height < 1
                or len(image.data) != image.width * image.height * 3):
            raise ValueError(f"Expected RAW_RGB sensor: {source}")
        rgb = np.frombuffer(image.data, np.uint8).reshape(image.height, image.width, 3)
        # Preserve source resolution; SDK-specific resizing must be verified separately.
        ok, jpeg = cv2.imencode('.jpg', cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        if not ok:
            raise ValueError("JPEG encoding failed")
        frames[target] = base64.b64encode(jpeg).decode('ascii')
    return {"frames": frames, "timestamp": observation.robot_time_ns / 1e9 if observation.robot_time_ns else observation.control_step / control_hz,
            "observation": {"prompt": observation.instruction,
                "state.left_arm": joints[:6].tolist(), "state.right_arm": joints[6:].tolist(),
                "state.left_gripper": grips[:1].tolist(), "state.right_gripper": grips[1:].tolist()}}


class DreamAdapter:
    supported_robot_types = ('yam',)

    def __init__(self, options):
        self.timeout = float(options.get('http_timeout_seconds', 120))
        if not 0 < self.timeout <= 120:
            raise ValueError("Invalid HTTP timeout")
        self.client = self.model = self.session_id = None
        self.failed = False

    def validate(self, model):
        options = model.backend_options
        if (options.get('adapter') != 'dream_yam' or model.action_space != 'joint_position'
                or model.action_dim != 14 or model.control_hz != 30 or model.max_horizon != 48):
            raise ValueError("Dream YAM requires joint_position, 14 dimensions, 30 Hz, max_horizon 48")
        if options.get('contract_confirmed') is not True:
            raise ValueError("Confirm Dream absolute radians, joint order, gripper convention and 30 Hz before enabling contract_confirmed")
        if type(options.get('invert_grippers', False)) is not bool:
            raise ValueError("invert_grippers must be boolean")
        endpoint = urlsplit(model.endpoint)
        if endpoint.scheme != 'http' or endpoint.hostname != '127.0.0.1' or endpoint.port is None or endpoint.path not in ('', '/') or endpoint.query or endpoint.fragment or endpoint.username:
            raise ValueError("Worker endpoint must be http://127.0.0.1:PORT")

    async def _post(self, path, payload):
        response = await self.client.post(path, json=payload)
        response.raise_for_status()
        return response.json()

    async def start_session(self, model, request):
        self.validate(model)
        if self.client is not None:
            raise RuntimeError("Session already active")
        self.model, self.session_id, self.failed = model, uuid.uuid4().hex, False
        self.client = httpx.AsyncClient(base_url=model.endpoint, timeout=self.timeout, trust_env=False)
        try:
            health = await self.client.get('/health')
            health.raise_for_status()
            if health.json().get('protocol') != 'dream-policy-adapter-v1':
                raise RuntimeError("Unexpected service at worker endpoint")
            result = await self._post('/session/start', {
                'session_id': self.session_id, 'checkpoint': model.backend_options.get('checkpoint', 'dreamlabs/dream-1-beta'),
                'max_horizon': model.max_horizon, 'invert_grippers': model.backend_options.get('invert_grippers', False)})
            if result.get('session_id') != self.session_id:
                raise RuntimeError("Worker session mismatch")
        except BaseException:
            await self.end_session()
            raise

    async def infer(self, model, observation):
        if self.client is None or model != self.model or self.failed:
            raise RuntimeError("No usable prepared Dream session")
        try:
            payload = observation_payload(observation, model.control_hz, model.backend_options.get('invert_grippers', False))
            payload['session_id'] = self.session_id
            result = await self._post('/infer', payload)
            if result.get('session_id') != self.session_id:
                raise RuntimeError("Worker session mismatch")
            return validate_actions(result['actions'], model.max_horizon)
        except BaseException:
            # A timed-out SDK request may still run; never reuse its session.
            self.failed = True
            raise

    async def end_session(self):
        if self.client is None:
            return
        client, sid = self.client, self.session_id
        self.client = self.model = self.session_id = None
        try:
            response = await client.post('/session/end', json={'session_id': sid}, timeout=2)
            response.raise_for_status()
        except httpx.HTTPError:
            pass  # The worker never reuses a session; next start closes any old client.
        finally:
            await client.aclose()


def create_adapter(options):
    return DreamAdapter(options)
