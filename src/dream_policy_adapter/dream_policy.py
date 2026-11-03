"""Dream SDK boundary, based on the user's working DreamClient example."""
import math
import time

import numpy as np


def validate_actions(value, max_horizon=48):
    actions = np.asarray(value, dtype=np.float32)
    if (actions.ndim != 2 or actions.shape[1] != 14
            or not 1 <= len(actions) <= max_horizon or not np.isfinite(actions).all()):
        raise ValueError("Expected finite (horizon, 14) actions within max_horizon")
    grips = actions[:, [6, 13]]
    # Tolerate normalized predictions in [-0.1, 1.2], then saturate; no unit conversion.
    if np.any((grips < np.float32(-0.1)) | (grips > np.float32(1.2))):
        message = ("Dream grippers exceed [-0.1, 1.2]: "
                   f"left=[{grips[:, 0].min():.6g}, {grips[:, 0].max():.6g}], "
                   f"right=[{grips[:, 1].min():.6g}, {grips[:, 1].max():.6g}]")
        # Only numeric action ranges; never log SDK exceptions or credentials.
        print(message, flush=True)
        raise ValueError(message)
    if np.any((grips < 0) | (grips > 1)):
        actions = actions.copy()
        actions[:, [6, 13]] = np.clip(grips, 0, 1)
    return actions


def to_actions(result, max_horizon=48, invert_grippers=False):
    values = result["per_key"]
    left = np.asarray(values["action.left_arm"], dtype=np.float32)
    right = np.asarray(values["action.right_arm"], dtype=np.float32)
    lg = np.asarray(values["action.left_grip"], dtype=np.float32)
    rg = np.asarray(values["action.right_grip"], dtype=np.float32)
    if (left.ndim != 2 or left.shape[1] != 6 or right.shape != left.shape
            or lg.shape != (len(left),) or rg.shape != lg.shape):
        raise ValueError("Invalid Dream arm/gripper shapes")
    actions = validate_actions(np.concatenate([left, lg[:, None], right, rg[:, None]], axis=1), max_horizon)
    if invert_grippers:
        actions[:, [6, 13]] = 1 - actions[:, [6, 13]]
    return actions


def load_sdk():
    import dream
    if not callable(getattr(dream, "DreamClient", None)):
        raise RuntimeError("Install the Dream SDK used by the supplied example: DreamClient is required; PolicyClient-only SDKs are incompatible")
    if not isinstance(getattr(dream, "DreamServerError", None), type):
        raise RuntimeError("Dream SDK must export DreamServerError")
    return dream


class DreamPolicy:
    def __init__(self, checkpoint, max_horizon, invert_grippers=False, sdk=None):
        self.checkpoint = checkpoint
        self.sdk = sdk or load_sdk()
        self.client = self.sdk.DreamClient()
        self.max_horizon = max_horizon
        self.invert_grippers = invert_grippers
        self.last_timestamp = None
        try:
            for name in ("policy", "set_observation", "get_action", "close"):
                if not callable(getattr(self.client, name, None)):
                    raise RuntimeError(f"Incompatible DreamClient: missing {name}")
            self.client.policy({"checkpoint": checkpoint, "ik": {"arm": "yam", "head": "joint"}})
        except BaseException:
            self.close()
            raise

    def infer(self, payload):
        expired = getattr(self.sdk, "DreamSessionExpired", ())
        try:
            return self._infer_once(payload)
        except expired:
            # The gateway restarted the episode. Discard its result and create
            # a fresh client; replay this request's observation exactly once.
            print("Dream session expired; rebuilding once for this observation", flush=True)
            self.close()
            self.__init__(self.checkpoint, self.max_horizon,
                          self.invert_grippers, sdk=self.sdk)
            # Outside the try block: another expiration propagates to the worker.
            return self._infer_once(payload)

    def _infer_once(self, payload):
        timestamp = float(payload["timestamp"])
        if not math.isfinite(timestamp) or timestamp < 0 or (self.last_timestamp is not None and timestamp <= self.last_timestamp):
            raise ValueError("Observation timestamps must increase within a session")
        self.client.set_observation(payload["frames"], payload["observation"], timestamp=timestamp)
        self.last_timestamp = timestamp
        # Submit each real observation once; do not fabricate a history window.
        for attempt in range(3):
            try:
                return to_actions(self.client.get_action(), self.max_horizon, self.invert_grippers)
            except self.sdk.DreamServerError as error:
                if not error.retryable or attempt == 2:
                    raise
                wait = float(error.retry_after_s or 1.0)
                if not math.isfinite(wait) or not 0 <= wait <= 30:
                    raise
                time.sleep(wait)

    def close(self):
        close = getattr(self.client, "close", None)
        if close is not None:
            close()
