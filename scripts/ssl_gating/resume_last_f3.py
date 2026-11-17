"""Resume F3 while accepting Transformers' added null attention config field.

The original guarded trainer and saved artifacts remain unchanged. Only a
missing-versus-null serialization difference is normalized for its strict check.
The model constructor still explicitly selects the original attention backend.
"""

import json

from scripts.ssl_gating import train_last_f3 as trainer


_strict_verify_recovery = trainer.verify_recovery


def verify_recovery(state, config):
    saved_config = json.loads(json.dumps(state['config']))
    frontend = saved_config.get('frontend_config', {})
    expected = config.get('frontend_config', {})
    if ('attn_implementation' not in expected
            and 'attn_implementation' in frontend
            and frontend['attn_implementation'] is None):
        del frontend['attn_implementation']
    normalized = dict(state, config=saved_config)
    _strict_verify_recovery(normalized, config)


def main():
    trainer.verify_recovery = verify_recovery
    trainer.main()


if __name__ == '__main__':
    main()
