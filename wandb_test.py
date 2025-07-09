#!/usr/bin/env python
"""Quick script to verify your Weights & Biases setup.

Usage:
  python wandb_test.py

It will:
1. Read WANDB_API_KEY and WANDB_ENTITY from the environment (set by env_setup.sh)
2. Log in to wandb (if not already logged-in)
3. Create a short dummy run in the specified project/ entity.
"""

import os
import random
import wandb

# 1. Pick up credentials from environment (env_setup.sh sets these)
WANDB_API_KEY = os.getenv("WANDB_API_KEY")
WANDB_ENTITY = os.getenv("WANDB_ENTITY", "yuxuan_zhang13-uc-san-diego")  # fallback to default

# 2. Log in (no-op if already logged in)
# wandb.login() automatically looks at the env var, but we call explicitly for clarity
wandb.login(key=WANDB_API_KEY)  # type: ignore[attr-defined]

# 3. Create a short run to test logging
EPOCHS = 5
LR = 0.01

run = wandb.init(  # type: ignore[attr-defined]
    project="lmgame-wandb-test",  # small throw-away project name
    entity=WANDB_ENTITY,
    config={
        "learning_rate": LR,
        "epochs": EPOCHS,
    },
)

print(f"⚙️  Logging to wandb entity: {WANDB_ENTITY}, project: {run.project}")
print(f"🔑  Using API key: {'SET' if WANDB_API_KEY else 'NOT SET (using cached creds)'}")

# Dummy training loop
offset = random.random() / 5
for epoch in range(1, EPOCHS + 1):
    acc = 1 - 2 ** -epoch - random.random() / epoch - offset
    loss = 2 ** -epoch + random.random() / epoch + offset
    print(f"epoch={epoch}, accuracy={acc:.4f}, loss={loss:.4f}")
    wandb.log({"accuracy": acc, "loss": loss, "epoch": epoch})  # type: ignore[attr-defined]

print("✅ wandb test run completed!")
run.finish()