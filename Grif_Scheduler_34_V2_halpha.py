from __future__ import annotations

# ──────────────────────────────────────────────────────────────────────
# standard libs
# ──────────────────────────────────────────────────────────────────────
import argparse
import enum
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm

# ──────────────────────────────────────────────────────────────────────
# third-party
# ──────────────────────────────────────────────────────────────────────
import simpy
import yaml

# ──────────────────────────────────────────────────────────────────────
# globals / bookkeeping
# ──────────────────────────────────────────────────────────────────────
sample_loc: dict[int, str] = {}         # sid -> location string (Mixer07, Waste, etc.)
results: list[dict] = []               # event log
batch_dispatchers: dict[str, simpy.Store] = {}  # per-step batch queues (keyed by process+bundle+step)

# boat context
sample_boat: dict[int, int] = {}        # sid -> boat_id
boat_process: dict[int, str] = {}       # boat_id -> process name ("A", "B", ...)
boat_bundle: dict[int, int] = {}        # boat_id -> bundle index (0..N-1)

logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(message)s",
    level=logging.INFO,
)
log = logging.getLogger("sim")


# ──────────────────────────────────────────────────────────────────────
# recipe metadata
# ──────────────────────────────────────────────────────────────────────
REAGENT_SRC = {
    "Dispense Sample": "Sample",
    "Dispense Lysis":  "Lysis",
    "Dispense cRNA":   "cRNA",
    "Dispense IC":     "IC",
    "Dispense pK":     "pK",
    "Dispense Bind":   "Bind",
    "Dispense W1":     "Wash1",
    "Dispense W2":     "Wash2",
    "Dispense W3":     "Wash3",
    "Dispense Elute":  "Elute",
}

LARGE_TIP_STEPS = {
    "Dispense Sample",
    # add more if needed
}

BOAT_SIZE_DEFAULT = 8


# ──────────────────────────────────────────────────────────────────────
# config dataclasses
# ──────────────────────────────────────────────────────────────────────
class Prio(enum.IntEnum):
    """SimPy PriorityResource: lower number ⇒ higher priority."""
    CRITICAL = 0
    URGENT   = 1
    NORMAL   = 2
    NEW      = 3


@dataclass
class Step:
    name: str
    resource: str
    duration_s: int
    max_delay_after_s: int
    step_batch: int = 1
    tool_type: str = "none"     # "gripper" | "pipettor" | "boat_move" | "mixer" | "bulk_disp" | ...
    mutate_loc: bool = False
    cap_move: bool = False


@dataclass
class ResourceSpec:
    quantity: int
    batch_size: int = 1         # 1 ⇒ PriorityResource, >1 ⇒ global BatchingResource


@dataclass
class SimConfig:
    # high-level knobs
    simulation_duration_min: int
    samples_per_arrival_batch: int
    arrival_batch_interval_min: int
    total_samples_target: int

    # boat model
    boat_size: int
    boat_process_pattern: List[str]      # e.g. ["A","B"] repeating
    boat_bundle_pattern: Optional[List[int]]  # e.g. [0,1] repeating (optional)

    # resources and processes
    resources: Dict[str, ResourceSpec]
    processes: Dict[str, List[Step]]     # process_name -> steps
    fluidic_bundles: List[Dict[str, str]]  # bundle_idx -> {logical_resource: actual_resource}

    @property
    def simulation_duration_s(self) -> int:
        return self.simulation_duration_min * 60


# ──────────────────────────────────────────────────────────────────────
# defaults
# ──────────────────────────────────────────────────────────────────────
DEFAULT_RAW_CFG: dict = {
    "simulation_duration_min": 240,
    "samples_per_arrival_batch": 2,        # each arrival batch == one boat launch (by default)
    "arrival_batch_interval_min": 4.5,
    "total_samples_target": 2,

    "boat_size": 8,
    "boat_process_pattern": ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"],     # boat0->A, boat1->B, boat2->A, ...
    # optional override; if omitted we use round-robin bundles
    # "boat_bundle_pattern": [0, 1],

    "resources": {
        # NOTE: "Mixer" is interpreted as NUMBER OF MIXER STATIONS (tube residency limit)
        "Mixer":        {"quantity": 40, "batch_size": 1},

        # Example fluidics/tooling
        "PipSamp":      {"quantity": 1,  "batch_size": 2},
        "PipElut":      {"quantity": 1,  "batch_size": 2},
        "DispensorA": {"quantity": 1,  "batch_size": 2},
        "DispensorB": {"quantity": 1,  "batch_size": 2},
        "AspiratorA":   {"quantity": 1,  "batch_size": 2},
        "AspiratorB":   {"quantity": 1,  "batch_size": 2},
        "Sonicator":    {"quantity": 4,  "batch_size": 2},
        "BoatA":         {"quantity": 1,  "batch_size": 8},  # your existing "Boat" tool usage (clamp, etc.)
        "BoatB":         {"quantity": 1,  "batch_size": 8},  # your existing "Boat" tool usage (clamp, etc.)
        "BoatC":         {"quantity": 1,  "batch_size": 8},  # your existing "Boat" tool usage (clamp, etc.)
        "BoatD":         {"quantity": 1,  "batch_size": 8},  # your existing "Boat" tool usage (clamp, etc.)
        "BoatE":         {"quantity": 1,  "batch_size": 8},  # your existing "Boat" tool usage (clamp, etc.)
        "BoatMover":    {"quantity": 1,  "batch_size": 8},
        #"PCR":      {"quantity": 6,  "batch_size": 8},

    },

    # Fluidic bundle mapping:
    # Steps may request logical names like "Aspirator" and we resolve per-boat to actual.
    "fluidic_bundles": [
        {"Aspirator": "AspiratorA", "Dispensor": "DispensorA", "Boat": "BoatA"},
        {"Aspirator": "AspiratorB", "Dispensor": "DispensorB", "Boat": "BoatB"},
        {"Aspirator": "AspiratorA", "Dispensor": "DispensorA", "Boat": "BoatC"},
        {"Aspirator": "AspiratorB", "Dispensor": "DispensorB", "Boat": "BoatD"},
        {"Aspirator": "AspiratorA", "Dispensor": "DispensorA", "Boat": "BoatE"},
        {"Aspirator": "AspiratorB", "Dispensor": "DispensorB", "Boat": "BoatA"},
        {"Aspirator": "AspiratorA", "Dispensor": "DispensorA", "Boat": "BoatB"},
        {"Aspirator": "AspiratorB", "Dispensor": "DispensorB", "Boat": "BoatC"},
        {"Aspirator": "AspiratorA", "Dispensor": "DispensorA", "Boat": "BoatD"},
        {"Aspirator": "AspiratorB", "Dispensor": "DispensorB", "Boat": "BoatE"},
    ],

    # Two processes. You can diverge these step lists over time.
    # NOTE: Aspirator steps use resource="Aspirator" (LOGICAL), not Aspirator1/2.
    "processes": {
        "A": [
            {"name": "Consumable Placement","resource": "PipSamp",      "duration_s": 12,  "max_delay_after_s": 500,  "step_batch": 1, "tool_type": "gripper",   "mutate_loc": True},
            {"name": "Clamp",               "resource": "Boat",         "duration_s": 1,   "max_delay_after_s": 1000, "step_batch": 1, "tool_type": "gripper",   "mutate_loc": False},
            {"name": "UnCap",               "resource": "PipSamp",      "duration_s": 8,  "max_delay_after_s": 1000, "step_batch": 1, "tool_type": "gripper",   "mutate_loc": False},
             {"name": "Unclamp",             "resource": "Boat",         "duration_s": 1,   "max_delay_after_s": 1000, "step_batch": 1, "tool_type": "mixer",     "mutate_loc": False},
           {"name": "Dispense Sample",     "resource": "PipSamp",      "duration_s": 30,  "max_delay_after_s": 500,  "step_batch": 1, "tool_type": "pipettor",  "mutate_loc": False},
            {"name": "Dispense Lysis",      "resource": "Dispensor", "duration_s": 13,  "max_delay_after_s": 500,  "step_batch": 1, "tool_type": "bulk_disp", "mutate_loc": False},
            {"name": "Cover",               "resource": "PipSamp",      "duration_s": 8,  "max_delay_after_s": 1000, "step_batch": 1, "tool_type": "gripper",   "mutate_loc": False},

            # Example boat dynamics: non-gating drift (resource None)
            #{"name": "Boat Drift (non-gating)", "resource": "None",     "duration_s": 30,   "max_delay_after_s": 1000, "step_batch": 1, "tool_type": "boat_move", "mutate_loc": False},

            # Example gating move (requests BoatMover)
            #{"name": "Boat Index (gating)", "resource": "BoatMover",    "duration_s": 20,   "max_delay_after_s": 1000, "step_batch": 1, "tool_type": "boat_move", "mutate_loc": False},

            #{"name": "Clamp",               "resource": "Boat",         "duration_s": 5,   "max_delay_after_s": 1000, "step_batch": 1, "tool_type": "gripper",   "mutate_loc": False},
            {"name": "Mixer  preSon",           "resource": "Mixer",        "duration_s": 60,  "max_delay_after_s": 1000, "step_batch": 1, "tool_type": "mixer",     "mutate_loc": False},
            {"name": "Sonicate Incubate",   "resource": "Sonicator",    "duration_s": 180, "max_delay_after_s": 1000, "step_batch": 1, "tool_type": "mixer",     "mutate_loc": False},
            {"name": "Mixer  postSon",           "resource": "Mixer",        "duration_s": 60,  "max_delay_after_s": 1000, "step_batch": 1, "tool_type": "mixer",     "mutate_loc": False},
            {"name": "UnCover",             "resource": "PipSamp",      "duration_s": 8,  "max_delay_after_s": 1000, "step_batch": 1, "tool_type": "gripper",   "mutate_loc": False},
            #{"name": "UnClamp",             "resource": "Boat",         "duration_s": 5,   "max_delay_after_s": 1000, "step_batch": 1, "tool_type": "gripper",   "mutate_loc": False},
            {"name": "Dispense Bind",       "resource": "Dispensor", "duration_s": 13,  "max_delay_after_s": 1000, "step_batch": 1, "tool_type": "bulk_disp", "mutate_loc": False},
            {"name": "Bind Incubation",     "resource": "Mixer",        "duration_s": 300, "max_delay_after_s": 1000, "step_batch": 1, "tool_type": "mixer",     "mutate_loc": False},
            {"name": "Capture_SN",          "resource": "Mixer",        "duration_s": 15,  "max_delay_after_s": 1000, "step_batch": 1, "tool_type": "mixer",     "mutate_loc": False},
            {"name": "Aspirate SN",         "resource": "Aspirator",    "duration_s": 25,  "max_delay_after_s": 1000, "step_batch": 1, "tool_type": "aspirator", "mutate_loc": False},
            {"name": "Dispense W1",         "resource": "Dispensor", "duration_s": 15,  "max_delay_after_s": 1000, "step_batch": 1, "tool_type": "bulk_disp", "mutate_loc": False},
            {"name": "Mixer  W1",           "resource": "Mixer",        "duration_s": 60,  "max_delay_after_s": 1000, "step_batch": 1, "tool_type": "mixer",     "mutate_loc": False},
            {"name": "Capture_W1",          "resource": "Mixer",        "duration_s": 15,  "max_delay_after_s": 1000, "step_batch": 1, "tool_type": "mixer",     "mutate_loc": False},
            {"name": "Aspirate W1",         "resource": "Aspirator",    "duration_s": 25,  "max_delay_after_s": 1000, "step_batch": 1, "tool_type": "aspirator", "mutate_loc": False},
            {"name": "Dispense W2",         "resource": "Dispensor", "duration_s": 15,  "max_delay_after_s": 1000, "step_batch": 1, "tool_type": "bulk_disp", "mutate_loc": False},
            {"name": "Mixer  W2",           "resource": "Mixer",        "duration_s": 60,  "max_delay_after_s": 1000, "step_batch": 1, "tool_type": "mixer",     "mutate_loc": False},
            {"name": "Capture_W2",          "resource": "Mixer",        "duration_s": 15,  "max_delay_after_s": 1000, "step_batch": 1, "tool_type": "mixer",     "mutate_loc": False},
            {"name": "Aspirate W2",         "resource": "Aspirator",    "duration_s": 25,  "max_delay_after_s": 1000, "step_batch": 1, "tool_type": "aspirator", "mutate_loc": False},
            {"name": "Dispense W3",         "resource": "Dispensor", "duration_s": 15,  "max_delay_after_s": 1000, "step_batch": 1, "tool_type": "bulk_disp", "mutate_loc": False},
            {"name": "Mixer  W3",           "resource": "Mixer",        "duration_s": 60,  "max_delay_after_s": 1000, "step_batch": 1, "tool_type": "mixer",     "mutate_loc": False},
            {"name": "Capture_W3",          "resource": "Mixer",        "duration_s": 15,  "max_delay_after_s": 1000, "step_batch": 1, "tool_type": "mixer",     "mutate_loc": False},
            {"name": "Aspirate W3",         "resource": "Aspirator",    "duration_s": 25,  "max_delay_after_s": 1000, "step_batch": 1, "tool_type": "aspirator", "mutate_loc": False},
            #{"name": "Evaporation WF",      "resource": "Mixer",        "duration_s": 75,  "max_delay_after_s": 1000, "step_batch": 1, "tool_type": "mixer",     "mutate_loc": False},
            {"name": "Dispense Elute",      "resource": "Dispensor", "duration_s": 11,  "max_delay_after_s": 1000, "step_batch": 1, "tool_type": "bulk_disp", "mutate_loc": False},
            {"name": "Mixer  Elute",        "resource": "Mixer",        "duration_s": 300, "max_delay_after_s": 60,   "step_batch": 1, "tool_type": "mixer",     "mutate_loc": False},
            {"name": "Capture_Elute",       "resource": "Mixer",        "duration_s": 15,  "max_delay_after_s": 1000, "step_batch": 1, "tool_type": "mixer",     "mutate_loc": False},
            {"name": "Aspirate Elute",      "resource": "PipElut",      "duration_s": 30,  "max_delay_after_s": 1000, "step_batch": 1, "tool_type": "pipettor",  "mutate_loc": False},
            {"name": "Clamp",               "resource": "Boat",         "duration_s": 5,   "max_delay_after_s": 1000, "step_batch": 1, "tool_type": "mixer",     "mutate_loc": False},
            {"name": "cap",                 "resource": "PipElut",      "duration_s": 8,  "max_delay_after_s": 500,  "step_batch": 1, "tool_type": "gripper",   "mutate_loc": False},
            {"name": "Dispose Consumable",  "resource": "PipElut",      "duration_s": 12,  "max_delay_after_s": 1000, "step_batch": 1, "tool_type": "gripper",   "mutate_loc": True},
            #{"name": "PCR",  "resource": "PCR",      "duration_s": 2100,  "max_delay_after_s": 1000, "step_batch": 1, "tool_type": "gripper",   "mutate_loc": True},
        ],

        # Process B starts identical by default (edit freely later)
        "B": [],  # filled at load-time if empty -> clone of A
        "C": [],  # filled at load-time if empty -> clone of A
        "D": [],  # filled at load-time if empty -> clone of A
        "E": [],  # filled at load-time if empty -> clone of A
        "F": [],  # filled at load-time if empty -> clone of A
        "G": [],  # filled at load-time if empty -> clone of A
        "H": [],  # filled at load-time if empty -> clone of A
        "I": [],  # filled at load-time if empty -> clone of A
        "J": [],  # filled at load-time if empty -> clone of A
    },
}


# ──────────────────────────────────────────────────────────────────────
# config loader
# ──────────────────────────────────────────────────────────────────────
def load_config(path: Optional[Path]) -> SimConfig:
    if path and path.exists():
        log.info("Loading configuration from %s", path)
        with path.open() as fh:
            raw = yaml.safe_load(fh)
    else:
        if path:
            log.warning("%s not found – using built-in defaults", path)
        else:
            log.info("No YAML supplied – using built-in defaults")
        from copy import deepcopy
        raw = deepcopy(DEFAULT_RAW_CFG)

    # Backward-compat: if someone provides process_steps, treat as process "A"
    if "process_steps" in raw and "processes" not in raw:
        raw["processes"] = {"A": raw["process_steps"]}
        raw.setdefault("boat_process_pattern", ["A"])

    # Fill empty/missing processes with a clone of A (arbitrary count)
    if "processes" in raw and "A" in raw["processes"]:
        from copy import deepcopy

        # 1) ensure every process named in the pattern exists
        patt = raw.get("boat_process_pattern", ["A"])
        for pname in patt:
            raw["processes"].setdefault(pname, [])

        # 2) clone ANY empty process from A (not just B)
        for pname, steps in list(raw["processes"].items()):
            if pname != "A" and (not steps):          # [] or None
                raw["processes"][pname] = deepcopy(raw["processes"]["A"])
    missing = [p for p in raw.get("boat_process_pattern", ["A"]) if p not in raw["processes"]]
    if missing:
        log.warning("boat_process_pattern references missing processes: %s", missing)


    res_specs = {k: ResourceSpec(**v) for k, v in raw["resources"].items()}
    proc_steps: Dict[str, List[Step]] = {
        pname: [Step(**s) for s in steps]
        for pname, steps in raw["processes"].items()
    }

    return SimConfig(
        simulation_duration_min    = raw["simulation_duration_min"],
        samples_per_arrival_batch  = raw["samples_per_arrival_batch"],
        arrival_batch_interval_min = raw["arrival_batch_interval_min"],
        total_samples_target       = raw["total_samples_target"],
        boat_size                  = raw.get("boat_size", BOAT_SIZE_DEFAULT),
        boat_process_pattern       = raw.get("boat_process_pattern", ["A"]),
        boat_bundle_pattern        = raw.get("boat_bundle_pattern"),
        resources                  = res_specs,
        processes                  = proc_steps,
        fluidic_bundles            = raw.get("fluidic_bundles", []),
    )


# ──────────────────────────────────────────────────────────────────────
# seat pools + batching resources
# ──────────────────────────────────────────────────────────────────────
class SeatPool:
    """FIFO pool of named seats (Mixer01…MixerNN)."""
    def __init__(self, names: List[str]):
        self.free = names.copy()

    def acquire(self) -> str:
        if not self.free:
            return "SEAT_FULL"
        return self.free.pop(0)

    def release(self, seat: str) -> None:
        if seat and (seat not in self.free):
            self.free.append(seat)


class BatchingResource:
    """
    Global batching: collects spec.batch_size tokens, then runs on units(capacity=quantity).
    Token requires {done, dur}. Optional {prio}.
    """
    def __init__(self, env: simpy.Environment, name: str, spec: ResourceSpec):
        self.env, self.name, self.spec = env, name, spec
        self.store = simpy.Store(env)
        self.units = simpy.Resource(env, capacity=spec.quantity)
        env.process(self._dispatcher())

    def put(self, token: dict):
        return self.store.put(token)

    def _dispatcher(self):
        bid = 0
        while True:
            batch = []
            for _ in range(self.spec.batch_size):
                tok = yield self.store.get()
                batch.append(tok)
            bid += 1
            dur = max(tok["dur"] for tok in batch)
            self.env.process(self._run_batch(bid, batch, dur))

    def _run_batch(self, bid: int, batch: List[dict], dur: int):
        with self.units.request() as req:
            yield req
            yield self.env.timeout(dur)
        for tok in batch:
            tok["done"].succeed(self.env.now)


# ──────────────────────────────────────────────────────────────────────
# resource resolution (boat affinity)
# ──────────────────────────────────────────────────────────────────────
def resolve_resource(cfg: SimConfig, logical: str, sid: int) -> str:
    """
    Map a logical resource name (e.g., "Aspirator") to an actual resource name
    (e.g., "Aspirator1") based on the boat's bundle.
    """
    b = sample_boat[sid]
    bundle_idx = boat_bundle[b]
    if 0 <= bundle_idx < len(cfg.fluidic_bundles):
        return cfg.fluidic_bundles[bundle_idx].get(logical, logical)
    return logical


def is_none_resource(name: str) -> bool:
    return name.strip().lower() in {"none", "no", "null", "nil", ""}


# ──────────────────────────────────────────────────────────────────────
# build resources + step-batch dispatchers
# ──────────────────────────────────────────────────────────────────────
def build_resources(env: simpy.Environment, cfg: SimConfig) -> Dict[str, object]:
    res: Dict[str, object] = {}

    # MixerSeats count = Mixer station count (tube residency)
    mixer_qty = cfg.resources["Mixer"].quantity if "Mixer" in cfg.resources else 0
    res["MixerSeats"] = SeatPool([f"Mixer{i:02d}" for i in range(1, mixer_qty + 1)])

    # Build the raw tool resources
    for name, spec in cfg.resources.items():
        if spec.batch_size <= 1:
            res[name] = simpy.PriorityResource(env, capacity=spec.quantity)
        else:
            res[name] = BatchingResource(env, name, spec)

    # Build step-level batchers (if any steps have step_batch > 1)
    #
    # IMPORTANT: because a logical resource might resolve to multiple actual resources
    # (bundle 0 -> Aspirator1, bundle 1 -> Aspirator2), we create one dispatcher per:
    #   (process_name, step_index, step_name, actual_resource, bundle_idx)
    #
    # In your current config most steps have step_batch=1; this is here for completeness.
    for pname, steps in cfg.processes.items():
        for idx, st in enumerate(steps):
            if st.step_batch <= 1:
                continue

            for bundle_idx in range(max(1, len(cfg.fluidic_bundles))):
                # pretend a sample in this bundle; map logical->actual by bundle map directly
                actual = cfg.fluidic_bundles[bundle_idx].get(st.resource, st.resource) if cfg.fluidic_bundles else st.resource

                tool = res.get(actual)
                if not isinstance(tool, simpy.PriorityResource):
                    continue  # step-level batching only supports PriorityResource

                key = f"{pname}|B{bundle_idx}|{idx:02d}:{st.name}|{actual}"
                store = simpy.Store(env)
                batch_dispatchers[key] = store
                env.process(step_batcher(env, st, tool, store))

    return res


# ──────────────────────────────────────────────────────────────────────
# transfers / location
# ──────────────────────────────────────────────────────────────────────
def pipettor_transfer(step_name: str, sid: int) -> Tuple[str, str, str]:
    current_loc = sample_loc.get(sid, "UNKNOWN")

    is_asp = step_name.lower().startswith("aspirate")
    is_dsp = step_name.lower().startswith("dispense")
    tip_size = "LargePip" if step_name in LARGE_TIP_STEPS else "SmallPip"

    if is_asp:
        src = current_loc
        dst = "ElutionPlate" if step_name == "Aspirate Elute" else "LiquidWaste"
    elif is_dsp:
        src = REAGENT_SRC.get(step_name, "Reservoir:UNKNOWN")
        dst = current_loc
    else:
        src = dst = current_loc

    return src, dst, tip_size


def location_transfer(step: Step, sid: int, res: Dict[str, object]) -> Tuple[str, str]:
    current = sample_loc.get(sid, "UNKNOWN")
    src = dst = current

    if step.name == "Consumable Placement":
        src = "TubeRack"
        # seat was reserved before this step runs
        dst = sample_loc.get(sid, current)

    elif step.name == "Dispose Consumable":
        src = current
        dst = "Waste"
        if step.mutate_loc:
            sample_loc[sid] = dst

    return src, dst


def perform_step_transfer(step: Step, sid: int, res: Dict[str, object]) -> Tuple[str | None, str | None, str | None]:
    if step.tool_type == "gripper":
        src, dst = location_transfer(step, sid, res)
        return src, dst, "gripper"
    if step.tool_type == "pipettor":
        src, dst, tip = pipettor_transfer(step.name, sid)
        return src, dst, tip
    # boat_move, mixer, bulk_disp, aspirator, etc. are not location-mutating here
    return None, None, None


# ──────────────────────────────────────────────────────────────────────
# logging
# ──────────────────────────────────────────────────────────────────────
def log_event(kind: str,
              sid: int,
              step: str,
              resource: str | None,
              start: float,
              end: float,
              source: str | None = None,
              dest: str | None = None,
              tool_type: str | None = None) -> None:
    results.append({
        "sample_id": sid,
        "boat_id": sample_boat.get(sid, -1),
        "process": boat_process.get(sample_boat.get(sid, -1), "UNK"),
        "bundle": boat_bundle.get(sample_boat.get(sid, -1), -1),
        "step_name": step,
        "resource": resource,
        "start": start,
        "end": end,
        "duration": end - start,
        "kind": kind,
        "source": source,
        "destination": dest,
        "tool_type": tool_type,
    })


# ──────────────────────────────────────────────────────────────────────
# plots
# ──────────────────────────────────────────────────────────────────────
def plot_resource_gantt(df: pd.DataFrame) -> None:
    work = df[(df["kind"] == "work") & df["resource"].notna()].copy()
    if work.empty:
        log.warning("No work rows to plot (resource gantt)")
        return

    resources = sorted(work["resource"].unique())
    max_t = work["end"].max()

    fig_h = max(4, 0.4 * len(resources))
    fig, ax = plt.subplots(figsize=(14, fig_h))
    cmap = cm.get_cmap("tab20", len(resources))
    y_pos = {r: i for i, r in enumerate(resources)}

    for r in resources:
        block = work[work["resource"] == r]
        clr = cmap(y_pos[r])
        ax.barh([y_pos[r]] * len(block),
                block["duration"],
                left=block["start"],
                height=0.6,
                color=clr, edgecolor="black", linewidth=0.3)

    ax.set_xlim(0, max_t * 1.05)
    ax.set_yticks(range(len(resources)))
    ax.set_yticklabels(resources)
    ax.set_xlabel("Time (s)")
    ax.set_title("Resource-centric Gantt")
    ax.grid(axis="x", linestyle=":", alpha=0.4)
    plt.tight_layout()
    plt.show()


def plot_sample_gantt(df: pd.DataFrame) -> None:
    if df.empty:
        log.warning("No results data to plot")
        return

    work  = df[(df["kind"] == "work") & df["resource"].notna()].copy()
    waits = df[df["kind"] == "wait"].copy()
    if work.empty:
        log.warning("No work rows to plot")
        return

    # ✅ Collapse resource families to one legend/color entry
    def _group_resource(r):
        if not isinstance(r, str):
            return r
        if r.startswith("Mixer"):
            return "Mixer"
        # collapse BoatA/BoatB/... but keep BoatMover distinct if you use it
        if r.startswith("Boat") and r != "BoatMover":
            return "Boat"
        return r

    work["resource_group"] = work["resource"].apply(_group_resource)

    samples   = sorted(df["sample_id"].unique())
    groups    = sorted(work["resource_group"].unique())
    y_pos     = {sid: i for i, sid in enumerate(samples)}
    max_t     = work["end"].max()

    fig_h = max(4, 0.5 * len(samples))
    fig, ax = plt.subplots(figsize=(18, fig_h))

    cmap  = cm.get_cmap("tab20", len(groups))
    gclr  = {g: cmap(i) for i, g in enumerate(groups)}

    # waits
    ax.barh(
        [y_pos[s] for s in waits["sample_id"]],
        waits["duration"], left=waits["start"],
        height=0.25, color="lightgrey", alpha=0.6, edgecolor="none"
    )

    # work (color by group)
    ax.barh(
        [y_pos[s] for s in work["sample_id"]],
        work["duration"], left=work["start"],
        height=0.55,
        color=[gclr[g] for g in work["resource_group"]],
        edgecolor="black", linewidth=0.3
    )

    ax.set_xlim(0, max_t * 1.05)
    ax.set_yticks(range(len(samples)))
    ax.set_yticklabels([f"S{sid}" for sid in samples])
    ax.set_xlabel("Time (s)")
    ax.set_title("Sample-centric Gantt – grey = waiting, colour = work (Mixers+Boats collapsed)")
    ax.grid(axis="x", linestyle=":", alpha=0.4)

    # legend
    handles = [plt.Rectangle((0,0), 1, 1, color="lightgrey", alpha=0.6, label="Wait")]
    handles += [plt.Rectangle((0,0), 1, 1, color=gclr[g], label=g) for g in groups]
    ax.legend(handles=handles, fontsize="x-small",
              ncol=min(5, len(handles)), loc="upper right", framealpha=0.9)

    plt.tight_layout()
    plt.show()



# ──────────────────────────────────────────────────────────────────────
# step-level batching on PriorityResource
# ──────────────────────────────────────────────────────────────────────
def _wait_for_request(req):
    yield req


def step_batcher(env: simpy.Environment,
                 step: Step,
                 tool: simpy.PriorityResource,
                 store: simpy.Store,
                 timeout_s: int = 0):
    """
    Mini-batching on a PriorityResource.
    Tokens must include: {sid, done, holds_station(bool)}.
    """
    while True:
        first = yield store.get()
        batch = [first]
        t0 = env.now

        # build batch up to step.step_batch (optional timeout)
        while len(batch) < step.step_batch:
            remaining = timeout_s - (env.now - t0)
            if remaining <= 0:
                break
            get_ev = store.get()
            tmo_ev = env.timeout(remaining)
            yielded = yield get_ev | tmo_ev
            if get_ev in yielded:
                batch.append(yielded[get_ev])
                t0 = env.now
            else:
                break

        esc = Prio.CRITICAL if any(tok.get("holds_station") for tok in batch) else Prio.NORMAL
        req = tool.request(priority=esc)
        wait_ev = env.process(_wait_for_request(req))

        while len(batch) < step.step_batch:
            nxt_ev = store.get()
            yielded = yield nxt_ev | wait_ev
            if wait_ev in yielded:
                if nxt_ev in yielded:
                    batch.append(yielded[nxt_ev])
                else:
                    nxt_ev.cancel()
                break
            batch.append(yielded[nxt_ev])

        yield wait_ev
        yield env.timeout(step.duration_s)
        tool.release(req)

        for tok in batch:
            tok["done"].succeed(env.now)


def flush_batchers(env: simpy.Environment) -> simpy.events.Event:
    yield env.timeout(1)
    for key, store in batch_dispatchers.items():
        while store.items:
            tok = yield store.get()
            tok["done"].succeed(env.now)


# ──────────────────────────────────────────────────────────────────────
# core sample logic
# ──────────────────────────────────────────────────────────────────────
def sample_process(env: simpy.Environment, sid: int, cfg: SimConfig, res: Dict[str, object]):
    """
    Model:
      - Boat = group of cfg.boat_size samples.
      - Each boat has:
          * a process type (A/B/...)
          * a fluidic bundle (pins logical resources to actual tools)
      - Mixer station (res["Mixer"]) and a named Mixer seat are acquired BEFORE "Consumable Placement"
        and held until AFTER "Dispose Consumable".
      - Steps with resource == "Mixer" do not request again (station already held).
      - boat_move steps:
          * if resource == "BoatMover": gating (requests BoatMover)
          * if resource == "None": non-gating (just time)
    """
    arrival = env.now
    b = sample_boat[sid]
    proc = boat_process[b]
    bundle_idx = boat_bundle[b]

    steps = cfg.processes[proc]

    station_req = None
    resident_seat: str | None = None

    for idx, step in enumerate(steps):
        logical_res = step.resource
        actual_res = resolve_resource(cfg, logical_res, sid)

        # ── acquire station + seat just before placement step begins ─────
        if step.name == "Consumable Placement" and station_req is None:
            t_wait = env.now
            station_req = res["Mixer"].request(priority=Prio.NORMAL)
            yield station_req
            if (env.now - t_wait) > 1e-6:
                log_event("wait", sid, "WAIT-Station", "Mixer", t_wait, env.now)

            resident_seat = res["MixerSeats"].acquire()
            sample_loc[sid] = resident_seat

        # ── PATH 1: step-level batching (PriorityResource only) ─────────
        batch_key = f"{proc}|B{bundle_idx}|{idx:02d}:{step.name}|{actual_res}"
        if step.step_batch > 1 and batch_key in batch_dispatchers:
            done = env.event()
            t_wait = env.now
            batch_dispatchers[batch_key].put(
                {"sid": sid, "done": done, "holds_station": bool(station_req)}
            )
            yield done

            wait_end = env.now - step.duration_s
            if (wait_end - t_wait) > 1e-6:
                log_event("wait", sid, f"WAIT-{step.name}", actual_res, t_wait, wait_end)

            src, dst, ttype = perform_step_transfer(step, sid, res)
            log_event("work", sid, step.name, actual_res,
                      env.now - step.duration_s, env.now, src, dst, ttype)
        else:
            # ── PATH 2/3: normal PriorityResource or global BatchingResource ─
            # Special: local mixer work (station already held)
            if actual_res == "Mixer" and station_req is not None:
                yield env.timeout(step.duration_s)
                src, dst, ttype = perform_step_transfer(step, sid, res)
                # optionally log by seat instead of generic Mixer
                log_event("work", sid, step.name, resident_seat or "Mixer",
                          env.now - step.duration_s, env.now, src, dst, ttype)

            # Special: boat_move (gating vs non-gating)
            elif step.tool_type == "boat_move":
                t_wait = env.now
                if is_none_resource(actual_res):
                    # non-gating: no request, just time
                    yield env.timeout(step.duration_s)
                    log_event("work", sid, step.name, None,
                              env.now - step.duration_s, env.now, None, None, "boat_move")
                else:
                    # gating: request the mover
                    tool = res[actual_res]
                    if not isinstance(tool, simpy.PriorityResource):
                        raise TypeError(f"BoatMover must be PriorityResource; got {type(tool)}")
                    req = tool.request(priority=Prio.CRITICAL if station_req else Prio.NORMAL)
                    yield req
                    if (env.now - t_wait) > 1e-6:
                        log_event("wait", sid, f"WAIT-{step.name}", actual_res, t_wait, env.now)
                    yield env.timeout(step.duration_s)
                    tool.release(req)
                    log_event("work", sid, step.name, actual_res,
                              env.now - step.duration_s, env.now, None, None, "boat_move")

            else:
                tool = res.get(actual_res)

                # PriorityResource
                if isinstance(tool, simpy.PriorityResource):
                    prio = Prio.CRITICAL if station_req else Prio.NORMAL
                    req = tool.request(priority=prio)
                    t_wait = env.now
                    yield req
                    if (env.now - t_wait) > 1e-6:
                        log_event("wait", sid, f"WAIT-{step.name}", actual_res, t_wait, env.now)

                    yield env.timeout(step.duration_s)
                    src, dst, ttype = perform_step_transfer(step, sid, res)
                    log_event("work", sid, step.name, actual_res,
                              env.now - step.duration_s, env.now, src, dst, ttype)
                    tool.release(req)

                # global batching resource (BatchingResource)
                elif isinstance(tool, BatchingResource):
                    done = env.event()
                    t_wait = env.now
                    tool.put({"sid": sid, "done": done, "dur": step.duration_s})
                    yield done

                    wait_end = env.now - step.duration_s
                    if (wait_end - t_wait) > 1e-6:
                        log_event("wait", sid, f"WAIT-{step.name}", actual_res, t_wait, wait_end)

                    src, dst, ttype = perform_step_transfer(step, sid, res)
                    log_event("work", sid, step.name, actual_res,
                              env.now - step.duration_s, env.now, src, dst, ttype)
                else:
                    raise KeyError(f"Resource '{actual_res}' not found for step '{step.name}'")

        # ── release station + seat after disposal completes ──────────────
        if step.name == "Dispose Consumable":
            if station_req is not None:
                res["Mixer"].release(station_req)
                station_req = None
            if resident_seat is not None:
                res["MixerSeats"].release(resident_seat)
                resident_seat = None

    log_event("complete", sid, "COMPLETE", None, arrival, env.now)
    log.info("S%02d completed (boat=%d proc=%s bundle=%d cycle=%.1fs)",
             sid, b, proc, bundle_idx, env.now - arrival)


# ──────────────────────────────────────────────────────────────────────
# arrivals / boat assignment
# ──────────────────────────────────────────────────────────────────────
def assign_boat_context(cfg: SimConfig, boat_id: int) -> None:
    # process type
    patt = cfg.boat_process_pattern or ["A"]
    boat_process[boat_id] = patt[boat_id % len(patt)]

    # bundle selection
    if cfg.boat_bundle_pattern:
        bp = cfg.boat_bundle_pattern
        boat_bundle[boat_id] = bp[boat_id % len(bp)]
    else:
        # default: round-robin bundles
        nb = max(1, len(cfg.fluidic_bundles))
        boat_bundle[boat_id] = boat_id % nb


def sample_generator(env: simpy.Environment, cfg: SimConfig, res: Dict[str, object]):
    sid = 0
    boat_id = -1
    boat_size = max(1, cfg.boat_size)

    while sid < cfg.total_samples_target:
        if sid:
            yield env.timeout(cfg.arrival_batch_interval_min * 60)

        # each arrival batch constitutes a new boat by default
        boat_id += 1
        assign_boat_context(cfg, boat_id)

        launched = 0
        for _ in range(cfg.samples_per_arrival_batch):
            sid += 1
            if sid > cfg.total_samples_target:
                break

            sample_boat[sid] = boat_id
            launched += 1

            # OPTIONAL: enforce boat_size exactly (if you ever set samples_per_arrival_batch != boat_size)
            # If you want strict boat groups of size boat_size, you can rework this generator accordingly.

            env.process(sample_process(env, sid, cfg, res))

        log.info("Boat %d launched %d samples (up to S%02d) @ t=%.1fs  proc=%s bundle=%d",
                 boat_id, launched, sid, env.now,
                 boat_process[boat_id], boat_bundle[boat_id])


# ──────────────────────────────────────────────────────────────────────
# export + diagnostics
# ──────────────────────────────────────────────────────────────────────
def export_results(df: pd.DataFrame, cfg: SimConfig,
                   xlsx_fname="sim_results.xlsx",
                   csv_fname="sim_results_step_details.csv") -> None:
    if df.empty:
        log.warning("No results to export")
        return

    summary = {
        "Total samples": df["sample_id"].nunique(),
        "Total rows": len(df),
        "Sim time (s)": df["end"].max(),
    }
    cfg_dict = {
        "duration_min": cfg.simulation_duration_min,
        "arrival_batch": cfg.samples_per_arrival_batch,
        "arrival_int_min": cfg.arrival_batch_interval_min,
        "target": cfg.total_samples_target,
        "boat_size": cfg.boat_size,
        "process_pattern": ",".join(cfg.boat_process_pattern),
        "bundle_count": len(cfg.fluidic_bundles),
    }

    with pd.ExcelWriter(xlsx_fname, engine="openpyxl") as xl:
        df.to_excel(xl, sheet_name="Step Details", index=False)
        pd.DataFrame(list(summary.items()), columns=["Metric", "Value"]).to_excel(
            xl, sheet_name="Summary", index=False
        )
        pd.DataFrame(list(cfg_dict.items()), columns=["Param", "Value"]).to_excel(
            xl, sheet_name="Config", index=False
        )
    log.info("Excel exported → %s", xlsx_fname)

    df.to_csv(csv_fname, index=False)
    log.info("CSV exported → %s", csv_fname)


def run_diagnostics(env: simpy.Environment, resources: Dict[str, object], df: pd.DataFrame):
    # samples not complete
    last_row = (df.sort_values(["sample_id", "end"])
                  .groupby("sample_id").tail(1))
    stuck = last_row[last_row["step_name"] != "COMPLETE"]
    if stuck.empty:
        log.info("✓ every sample reached COMPLETE")
    else:
        log.warning("⚠ %d samples never finished", len(stuck))
        for _, row in stuck.iterrows():
            log.warning("  S%02d stuck after %-30s @ %.0fs",
                        row.sample_id, row.step_name, row.end)

    # resource utilization
    log.info("— final resource utilisation —")
    for name, r in resources.items():
        if isinstance(r, simpy.PriorityResource):
            log.info("%-14s held=%d queued=%d", name, r.count, len(r.queue))
        elif isinstance(r, BatchingResource):
            log.info("%-14s busy=%d unitQ=%d store_wait=%d",
                     name, r.units.count, len(r.units.queue), len(r.store.items))


# ──────────────────────────────────────────────────────────────────────
# main
# ──────────────────────────────────────────────────────────────────────
def main() -> None:
    ap = argparse.ArgumentParser(description="Medical-device throughput sim (boats + affinity)")
    ap.add_argument("-c", "--config", type=Path, help="Path to sim_config.yaml (optional)")
    ap.add_argument("--log", choices=["debug", "info", "warning", "error"],
                    default="info", help="Console log level")
    args = ap.parse_args()
    logging.getLogger("sim").setLevel(args.log.upper())

    cfg = load_config(args.config)
    env = simpy.Environment()
    res = build_resources(env, cfg)

    env.process(sample_generator(env, cfg, res))

    log.info("▶ starting simulation (%d min)…", cfg.simulation_duration_min)
    t0 = time.time()

    # run for configured duration
    env.run(until=cfg.simulation_duration_s)

    # flush any leftover step-batch tokens, then drain queue
    flush_ev = env.process(flush_batchers(env))
    env.run(until=flush_ev)
    env.run()

    log.info("⏹ finished @ t=%.1fs (wall=%.2fs)", env.now, time.time() - t0)

    df = pd.DataFrame(results)
    if df.empty:
        log.warning("No results collected.")
        return

    plot_resource_gantt(df)
    plot_sample_gantt(df)
    export_results(df, cfg)
    run_diagnostics(env, res, df)


if __name__ == "__main__":
    main()
