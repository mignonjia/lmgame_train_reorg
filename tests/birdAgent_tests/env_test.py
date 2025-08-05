#!/usr/bin/env python3
"""
Condensed sanity-checks for BirdEnv
"""

import sys, random
from datetime import datetime
from pathlib import Path

# ── project import setup ────────────────────────────────────────────────
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))          # repo packages

from agents.birdAgent.env import BirdEnv       # adjust if path differs

# ── logging helper (same as GSM8K template) ─────────────────────────────
def setup_logging():
    log_dir = Path(__file__).parent / "test_logs"
    log_dir.mkdir(exist_ok=True)

    ts       = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"bird_env_test_{ts}.log"

    class Tee:
        def __init__(self, fp):
            self.file   = open(fp, "w")
            self.stdout = sys.stdout
        def write(self, data):
            self.file.write(data); self.file.flush(); self.stdout.write(data)
        def flush(self):
            self.file.flush(); self.stdout.flush()
        def close(self):
            self.file.close()

    tee = Tee(log_file)
    sys.stdout = tee

    print(f"📝 BirdEnv test log started at {datetime.now()}")
    print(f"📄 Log file: {log_file}")
    print("=" * 60)
    return tee

# ── lightweight default config object ───────────────────────────────────
class BirdEnvCfg(dict):
    def __init__(self):
        super().__init__(
            dataset_path    = "birdsql/share-bam",
            split           = "train",
            max_steps       = 5,
            db_root         = "",          # let BirdEnv choose
            no_code_penalty = -0.5,
        )

def get_default_config():
    cfg = BirdEnvCfg()
    print("✅ Using default Bird configuration")
    for k, v in cfg.items():
        print(f"   {k}: {v}")
    return cfg

def make_env(cfg_dict):
    return BirdEnv(cfg_dict)

# ── individual tests ────────────────────────────────────────────────────
def test_env_creation_and_reset():
    print("🔍 Test 1: environment creation & reset")
    env = make_env(get_default_config())
    obs = env.reset(seed=42)
    assert isinstance(obs, str) and obs.startswith("[DB schema:")
    print("   Observation OK")
    env.close()

def test_step_logic():
    print("🔍 Test 2: step() with gold vs. dummy SQL")
    env = make_env(get_default_config())
    env.reset(seed=0)

    gold_sql_block = f"```sql\n{env.gold_sql}\n```"
    dummy_sql_block = "```sql\nSELECT 1;\n```"

    # gold → should succeed and finish
    _, r, done, info = env.step(gold_sql_block)
    assert done and r > 0 and info["success"]
    print(f"   Gold SQL matched → reward {r}")

    # reset & send dummy → should fail
    env.reset(seed=0)
    _, r, done, info = env.step(dummy_sql_block)
    assert not info["success"]
    print(f"   Dummy SQL mismatch → reward {r}")
    env.close()

def test_seeding_determinism():
    print("🔍 Test 3: seeding determinism")
    env = make_env(get_default_config())
    a = env.reset(seed=123)
    b = env.reset(seed=123)
    c = env.reset(seed=124)
    assert a == b and a != c
    print("   Same seed → same sample; diff seed → diff sample")
    env.close()

def test_info_structure():
    print("🔍 Test 4: info dict structure")
    env = make_env(get_default_config())
    env.reset(seed=0)
    _, _, _, info = env.step("```sql\nSELECT 1;\n```")
    assert set(info) == {"action_is_valid(code_block)", "success"}
    assert all(isinstance(v, bool) for v in info.values())
    print(f"   Info keys OK → {info}")
    env.close()

# ── main runner ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    tee = setup_logging()
    try:
        print("🚀 Starting condensed BirdEnv tests\n")
        test_env_creation_and_reset(); print()
        test_step_logic();             print()
        test_seeding_determinism();    print()
        test_info_structure();         print()
        print("=" * 60)
        print("🎉 All condensed tests passed!")
        print(f"✅ Completed at {datetime.now()}")
    except Exception as e:
        print("❌ Test run failed:", e)
        import traceback; traceback.print_exc()
    finally:
        tee.close()
        sys.stdout = tee.stdout
