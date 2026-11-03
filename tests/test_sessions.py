import asyncio
import threading
from types import SimpleNamespace

import httpx
import numpy as np
import pytest

from dream_policy_adapter.adapter import DreamAdapter
from dream_policy_adapter.dream_policy import DreamPolicy
from dream_policy_adapter.worker import SessionManager, make_server
from test_conversion import model, observation, result


class SDKError(Exception):
    retryable = True
    retry_after_s = .001


class FakeClient:
    instances = []

    def __init__(self):
        self.closed = False
        self.observations = []
        self.failures = 0
        self.calls = 0
        self.instances.append(self)

    def policy(self, spec):
        self.spec = spec

    def set_observation(self, frames, observation, timestamp):
        self.observations.append((frames, observation, timestamp))

    def get_action(self):
        self.calls += 1
        if self.calls <= self.failures:
            raise SDKError()
        return result()

    def close(self):
        self.closed = True


SDK = SimpleNamespace(DreamClient=FakeClient, DreamServerError=SDKError)


def factory(checkpoint, max_horizon, invert_grippers):
    return DreamPolicy(checkpoint, max_horizon, invert_grippers, sdk=SDK)


def test_sdk_retry_and_timestamp():
    policy = factory('test-checkpoint', 48, False)
    policy.client.failures = 1
    payload = {'timestamp': 0, 'frames': {}, 'observation': {}}
    assert policy.infer(payload).shape == (48, 14)
    assert policy.client.calls == 2
    assert len(policy.client.observations) == 1
    with pytest.raises(ValueError, match='timestamps'):
        policy.infer(payload)
    policy.close()
    assert policy.client.closed


def test_retry_limit():
    policy = factory('test', 48, False)
    policy.client.failures = 20
    with pytest.raises(SDKError):
        policy.infer({'timestamp': 0, 'frames': {}, 'observation': {}})
    assert policy.client.calls == 3


def test_session_replacement_and_stale_end():
    manager = SessionManager(factory)
    a, b = 'a'*32, 'b'*32
    def start(sid):
        manager.dispatch('/session/start', {'session_id': sid, 'checkpoint': 'test', 'max_horizon': 48, 'invert_grippers': False})
    start(a)
    previous = manager.policy.client
    start(b)
    assert previous.closed
    manager.dispatch('/session/end', {'session_id': a})
    assert manager.session_id == b
    with pytest.raises(ValueError):
        manager.dispatch('/infer', {'session_id': a})
    manager.close()


def test_worker_closes_failed_session():
    manager = SessionManager(factory)
    sid = 'a'*32
    manager.dispatch('/session/start', {'session_id': sid, 'checkpoint': 'test', 'max_horizon': 48, 'invert_grippers': False})
    client = manager.policy.client
    client.failures = 20
    with pytest.raises(SDKError):
        manager.dispatch('/infer', {'session_id': sid, 'timestamp': 0, 'frames': {}, 'observation': {}})
    assert client.closed and manager.session_id is None


def test_http_adapter_worker_integration():
    manager = SessionManager(factory)
    server = make_server(0, manager)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    async def run():
        m = model()
        m.endpoint = f'http://127.0.0.1:{server.server_port}'
        adapter = DreamAdapter({})
        try:
            await adapter.start_session(m, {'robot_type': 'yam'})
            client = manager.policy.client
            actions = await adapter.infer(m, observation())
            np.testing.assert_allclose(actions[0], [1]*6 + [.2] + [2]*6 + [.8])
            assert client.spec['ik'] == {'arm': 'yam', 'head': 'joint'}
            with pytest.raises(httpx.HTTPStatusError):
                await adapter.infer(m, observation())  # Timestamp repeated.
            with pytest.raises(RuntimeError, match='usable'):
                await adapter.infer(m, observation())
        finally:
            await adapter.end_session()
        assert client.closed
    try:
        asyncio.run(run())
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)
        manager.close()


def test_timeout_poisoning():
    async def run():
        adapter = DreamAdapter({})
        m = model()
        adapter.model = m
        adapter.session_id = 'a'*32
        async def handler(request):
            raise httpx.ReadTimeout('simulated', request=request)
        adapter.client = httpx.AsyncClient(transport=httpx.MockTransport(handler), base_url=m.endpoint)
        with pytest.raises(httpx.ReadTimeout):
            await adapter.infer(m, observation())
        assert adapter.failed
        with pytest.raises(RuntimeError):
            await adapter.infer(m, observation())
        await adapter.end_session()
    asyncio.run(run())


class SessionExpired(SDKError):
    retryable = False


@pytest.mark.parametrize('expire_again', [False, True])
def test_expired_session_rebuilds_only_once(expire_again):
    clients = []
    class ExpiringClient(FakeClient):
        def __init__(self):
            super().__init__()
            clients.append(self)
        def get_action(self):
            self.calls += 1
            if self is clients[0] or expire_again:
                raise SessionExpired()
            return result()
    sdk = SimpleNamespace(DreamClient=ExpiringClient, DreamServerError=SDKError,
                          DreamSessionExpired=SessionExpired)
    manager = SessionManager(lambda checkpoint, horizon, invert: DreamPolicy(checkpoint, horizon, invert, sdk=sdk))
    sid = 'a'*32
    manager.dispatch('/session/start', dict(session_id=sid, checkpoint='same-checkpoint', max_horizon=48, invert_grippers=True))
    payload = dict(session_id=sid, timestamp=123.5, frames={'head':'encoded'}, observation={'prompt':'task'})
    if expire_again:
        with pytest.raises(SessionExpired):
            manager.dispatch('/infer', payload)
        assert manager.policy is None and clients[1].closed
    else:
        actions = np.asarray(manager.dispatch('/infer', payload)['actions'])
        np.testing.assert_allclose(actions[0, [6, 13]], [.8, .2])
        assert manager.session_id == sid
        assert manager.policy.last_timestamp == 123.5
        with pytest.raises(ValueError, match='timestamps'):
            manager.policy.infer(payload)
    assert len(clients) == 2 and clients[0].closed
    assert clients[0].spec == clients[1].spec
    assert clients[0].observations == clients[1].observations
    assert clients[0].calls == clients[1].calls == 1
    manager.close()


def test_timeout_does_not_rebuild_client():
    class DreamTimeout(Exception):
        pass
    clients = []
    class TimedOutClient(FakeClient):
        def __init__(self):
            super().__init__()
            clients.append(self)
        def get_action(self):
            raise DreamTimeout()
    sdk = SimpleNamespace(DreamClient=TimedOutClient, DreamServerError=SDKError,
                          DreamSessionExpired=SessionExpired)
    policy = DreamPolicy('test', 48, sdk=sdk)
    with pytest.raises(DreamTimeout):
        policy.infer(dict(timestamp=0, frames={}, observation={}))
    assert len(clients) == 1
    policy.close()


@pytest.mark.parametrize('delay', [10.0, 30.0])
def test_long_retry_after_is_honored_with_bounded_attempts(monkeypatch, delay):
    monkeypatch.setattr(SDKError, 'retry_after_s', delay)
    waits = []
    monkeypatch.setattr('dream_policy_adapter.dream_policy.time.sleep', waits.append)
    policy = factory('test', 48, False)
    policy.client.failures = 2
    assert policy.infer(dict(timestamp=0, frames={}, observation={})).shape == (48, 14)
    assert waits == [delay, delay] and policy.client.calls == 3
    policy.client.failures = 100
    with pytest.raises(SDKError):
        policy.infer(dict(timestamp=1, frames={}, observation={}))
    assert policy.client.calls == 6 and waits == [delay] * 4
    policy.close()


@pytest.mark.parametrize('delay', [30.1, -1, float('nan'), float('inf')])
def test_invalid_retry_after_is_not_slept(monkeypatch, delay):
    monkeypatch.setattr(SDKError, 'retry_after_s', delay)
    waits = []
    monkeypatch.setattr('dream_policy_adapter.dream_policy.time.sleep', waits.append)
    policy = factory('test', 48, False)
    policy.client.failures = 100
    with pytest.raises(SDKError):
        policy.infer(dict(timestamp=0, frames={}, observation={}))
    assert not waits and policy.client.calls == 1
    policy.close()
