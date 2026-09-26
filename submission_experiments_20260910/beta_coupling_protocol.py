"""Torch-free protocol constants for beta-control checks and workers."""

PROTOCOL_VERSION = "beta-coupling-control-v4"
SEEDS = (42, 123, 456, 789, 2024)
AVAILABILITY_BINS = ("M1", "M2", "M3", "M4")
BETA_ARMS = ("learned", "fixed_0_5", "observed_count", "random_hash")
PRIMARY_AVAILABILITY_CONDITIONS = ("M2", "M3", "M4")
