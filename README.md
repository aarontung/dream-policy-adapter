# Dream Policy Adapter

A Dream-1 policy server integration for RoboColosseum, supporting YAM robots through the Dream API.

## Installation

Requires Python 3.10+, RoboColosseum Policy Server with plugin support, and a Dream SDK providing `DreamClient` and `DreamServerError`.

Replace `/PATH/TO/policy-server` with your Policy Server directory, e.g. `/home/ai/colosseum-policy-server`.

```bash
git clone git@github.com:aarontung/dream-policy-adapter.git
cd dream-policy-adapter
uv pip install --python /PATH/TO/policy-server/.venv/bin/python .
```

Follow the [Dream API setup guide](https://platform.dreamlabs.ai/setup-api) to install the Dream SDK and configure credentials in the same Python environment.

## Policy Server configuration

In your existing Policy Server config, merge the following backend settings and append the Dream entry to `models`. Keep your other settings and models.

```yaml
backend:
  name: auto
  options:
    vla:
      http_timeout_seconds: 120

models:
  - name: dream-1
    model_type: vla
    # Router matching placeholders, not a real repository or commit.
    url: https://huggingface.co/dreamlabs/dream-1
    revision: "0000000000000000000000000000000000000000"
    action_space: joint_position
    action_dim: 14
    control_hz: 30
    max_horizon: 48
    endpoint: http://127.0.0.1:9120
    launcher:
      - /PATH/TO/policy-server/.venv/bin/python
      - -m
      - dream_policy_adapter.worker
      - --port
      - "9120"
    backend_options:
      adapter: dream_yam
      robot_type: yam
      checkpoint: dreamlabs/dream-1-beta
      contract_confirmed: true
      invert_grippers: false
```

- `launcher`: replace the Python path with your actual path.
- `url` / `revision`: Router matching placeholders; keep them consistent with your Router configuration.

## Start

Replace `/PATH/TO/policy.yaml` with your existing config path:

```bash
/PATH/TO/policy-server/.venv/bin/colosseum-policy-local --config /PATH/TO/policy.yaml
```

The Policy Server starts the worker automatically. No Policy Server source changes are required.
