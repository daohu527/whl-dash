#!/usr/bin/env python3

# Copyright 2026 The WheelOS Team. All Rights Reserved.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Created Date: 2026-03-23
# Author: daohu527


import math
import os
import types
from typing import List, Dict, Any

import numpy as np
import pandas as pd
from cyber_record.record import Record


class RecordDataLoader:
    """Reads protobuf records, extracts scalar metrics dynamically, and exposes a pandas-eval compatible namespace."""

    def __init__(self, record_path: str, topics: List[str] = None):
        self.record_path = os.path.abspath(record_path)
        self.target_topics = topics or [
            "/apollo/control",
            "/apollo/canbus/chassis",
            "/apollo/localization/pose",
        ]
        self.master_df = pd.DataFrame()
        self.eval_env = {}
        self.available_signals = []
        self._load()

    def _get_record_files(self, path: str) -> List[str]:
        if os.path.isfile(path):
            return [path]
        res = []
        if os.path.isdir(path):
            for name in sorted(os.listdir(path)):
                f = os.path.join(path, name)
                if os.path.isfile(f) and ".record" in name:
                    res.append(f)
        if not res:
            raise FileNotFoundError(f"No records found in {path}")
        return res

    def _topic_alias(self, topic: str) -> str:
        return topic.strip("/").split("/")[-1]

    def _flatten_msg(self, msg, prefix="", max_depth=4, depth=0) -> Dict[str, Any]:
        res = {}
        if msg is None:
            return res
        try:
            for field_desc, val in msg.ListFields():
                key = f"{prefix}.{field_desc.name}" if prefix else field_desc.name
                if field_desc.label == field_desc.LABEL_REPEATED:
                    if field_desc.type == field_desc.TYPE_MESSAGE:
                        if depth < max_depth:
                            for i, item in enumerate(val):
                                res.update(
                                    self._flatten_msg(
                                        item, f"{key}[{i}]", max_depth, depth + 1
                                    )
                                )
                        else:
                            res[key] = len(val)
                    else:
                        res[key] = len(val)
                elif field_desc.type == field_desc.TYPE_MESSAGE:
                    if depth < max_depth:
                        res.update(self._flatten_msg(val, key, max_depth, depth + 1))
                else:
                    if isinstance(val, (int, float, bool)):
                        if isinstance(val, float) and not math.isfinite(val):
                            res[key] = np.nan
                        else:
                            res[key] = val
        except Exception:
            pass
        return res

    def _load(self):
        rows_by_topic = {}
        record_files = self._get_record_files(self.record_path)
        for f in record_files:
            rec = Record(f)
            for topic, msg, t in rec.read_messages(self.target_topics):
                alias = self._topic_alias(topic)
                row = self._flatten_msg(msg)

                # Derive timestamp
                ts = t / 1e9
                if hasattr(msg, "header") and msg.header.timestamp_sec > 0:
                    ts = msg.header.timestamp_sec
                elif (
                    hasattr(msg, "measurement_time")
                    and getattr(msg, "measurement_time", 0) > 0
                ):
                    ts = getattr(msg, "measurement_time")

                row["timestamp_sec"] = float(ts)
                rows_by_topic.setdefault(alias, []).append(row)

        if not rows_by_topic:
            raise RuntimeError(f"No targeted messages found in {self.record_path}")

        dfs = {}
        for topic, rows in rows_by_topic.items():
            df = pd.DataFrame(rows).sort_values("timestamp_sec").reset_index(drop=True)
            # Qualify column names (e.g. control.throttle)
            df.columns = [
                f"{topic}.{c}" if c != "timestamp_sec" else c for c in df.columns
            ]
            dfs[topic] = df

        # Master timeline: merge over the sequence of all unique timestamps
        all_ts = np.unique(
            np.concatenate([df["timestamp_sec"].values for df in dfs.values()])
        )
        all_ts.sort()

        master = pd.DataFrame({"timestamp_sec": all_ts})
        for topic, df in dfs.items():
            master = pd.merge_asof(
                master, df, on="timestamp_sec", direction="nearest", tolerance=0.1
            )

        t0 = float(master["timestamp_sec"].iloc[0])
        master["relative_time_sec"] = master["timestamp_sec"] - t0

        self.master_df = master
        self.available_signals = sorted(
            [
                c
                for c in master.columns
                if c not in ["timestamp_sec", "relative_time_sec"]
            ]
        )
        self.eval_env = self._build_eval_env(master)

    def _build_eval_env(self, df: pd.DataFrame) -> dict:
        env = {
            "np": np,
            "math": math,
            "relative_time_sec": df["relative_time_sec"].values,
        }

        # Custom analytical functions
        def deriv(x):
            if len(np.asarray(x)) < 2: return np.zeros_like(x)
            return np.gradient(np.asarray(x), df["relative_time_sec"].values)

        def smooth(x, window=10):
            x_arr = np.asarray(x)
            if len(x_arr) < window: return x_arr
            kernel = np.ones(window)/window
            return np.convolve(x_arr, kernel, mode='same')

        def rad2deg(x):
            return np.degrees(np.asarray(x))

        def deg2rad(x):
            return np.radians(np.asarray(x))

        env["deriv"] = deriv
        env["smooth"] = smooth
        env["rad2deg"] = rad2deg
        env["deg2rad"] = deg2rad
        env["abs"] = np.abs
        env["max"] = np.maximum
        env["min"] = np.minimum

        tree = {}

        # Build nested dict from dot-separated columns
        for col in df.columns:
            if col in ["timestamp_sec", "relative_time_sec"]:
                continue
            parts = col.split(".")
            curr = tree
            for p in parts[:-1]:
                if p not in curr:
                    curr[p] = {}
                curr = curr[p]
            curr[parts[-1]] = pd.to_numeric(df[col], errors="coerce").values

        # Convert dict branches to namespaces for proper object.path access in eval()
        def to_ns(t):
            ns = types.SimpleNamespace()
            for k, v in t.items():
                if isinstance(v, dict):
                    setattr(ns, k, to_ns(v))
                else:
                    setattr(ns, k, v)
            return ns

        for k, v in tree.items():
            if isinstance(v, dict):
                env[k] = to_ns(v)
            else:
                env[k] = v
        return env

    def evaluate(self, expr: str) -> np.ndarray:
        try:
            expr = expr.strip()
            if not expr:
                return None
            val = eval(expr, {"__builtins__": {}}, self.eval_env)
            return np.asarray(val, dtype=float)
        except Exception as e:
            print(f"Eval warning on '{expr}': {e}")
            return None
