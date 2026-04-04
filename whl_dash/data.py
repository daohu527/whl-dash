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
        self.target_topics = topics
        self.master_df = pd.DataFrame()
        self.eval_env = {}
        self.available_signals = []
        
        # New components for Spatial Layers (Frames)
        self.raw_frames = {}  # Dict[topic, List[Tuple[float, Any]]]
        self.frame_eval_envs = {} # Cache environments if needed
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
                    # Never flatten repeated elements into thousands of columns for scalar plots.
                    # It creates unmanageable pandas dataframes.
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
        
        # Auto-discover target topics if none provided
        if not self.target_topics:
            self.target_topics = []
            files = self._get_record_files(self.record_path)
            if files:
                first_rec = Record(files[0])
                try:
                    channels = list(first_rec._reader.channels.keys())
                except Exception:
                    channels = []
                
                # For performance, only auto-collect core AD topics
                core_topics = [
                    '/apollo/localization/pose',
                    '/apollo/control',
                    '/apollo/canbus/chassis',
                    '/apollo/planning',
                    '/apollo/prediction',
                    '/apollo/perception/obstacles',
                    '/apollo/routing',
                ]
                for c in channels:
                    if any(c.startswith(t) for t in core_topics):
                        self.target_topics.append(c)
        record_files = self._get_record_files(self.record_path)
        for f in record_files:
            rec = Record(f)
            for topic, msg, t in rec.read_messages(self.target_topics):
                if any(k in topic for k in ['/sensor/camera', '/sensor/lidar', 'pointcloud', 'image']):
                    continue
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
                
                # Store raw message for Frame inspection
                # Exclude pointclouds, images, radar, and large arrays to save RAM
                if not any(k in topic for k in ['/sensor/camera', '/sensor/lidar', 'pointcloud', 'image']):
                    self.raw_frames.setdefault(alias, []).append((float(ts), msg))

        # Sort frames by time
        for topic in self.raw_frames:
            self.raw_frames[topic].sort(key=lambda x: x[0])

        if not rows_by_topic:
            raise RuntimeError(f"No messages found in {self.record_path}")

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

    def get_spatial_frame(self, time_sec: float, layer_config) -> tuple:
        """
        Returns (x_list, y_list) for a given spatial layer config at the specified time.
        Config has: layer_type ('track' or 'frame'), topic, array_base, x_expr, y_expr
        """
        if layer_config.layer_type == 'track':
            # Fast path: extract from master_df using existing evaluate
            x = self.evaluate(layer_config.x_expr)
            y = self.evaluate(layer_config.y_expr)
            if x is not None and y is not None:
                # Decimate if needed, handled in dashboard
                return x, y
            return [], []
            
        elif layer_config.layer_type == 'frame':
            alias = self._topic_alias(layer_config.topic)
            frames = self.raw_frames.get(alias)
            if not frames:
                return [], []
            
            # Find the closest frame by time
            import bisect
            times = [f[0] for f in frames]
            t_base = self.master_df['timestamp_sec'].iloc[0] if not self.master_df.empty else 0
            abs_time = time_sec + t_base
            
            idx = bisect.bisect_left(times, abs_time)
            if idx >= len(times):
                idx = len(times) - 1
            if idx > 0 and abs(times[idx-1] - abs_time) < abs(times[idx] - abs_time):
                idx = idx - 1
                
            msg_dict = frames[idx][1]
            
            # Extract nested array
            array_obj = msg_dict
            if layer_config.array_base:
                parts = layer_config.array_base.split('.')
                for p in parts:
                    if p:
                        if hasattr(array_obj, p):
                            array_obj = getattr(array_obj, p)
                        elif isinstance(array_obj, dict):
                            array_obj = array_obj.get(p, [])
            
            if not isinstance(array_obj, list):
                if array_obj:  # Support scalar object
                    array_obj = [array_obj]
                else:
                    return [], []
                    
            x_list, y_list = [], []
            
            def eval_item(item, expr):
                if not expr: return 0.0
                try:
                    curr = item
                    for p in expr.split('.'):
                        if hasattr(curr, p):
                            curr = getattr(curr, p)
                        elif isinstance(curr, dict):
                            curr = curr.get(p, 0.0)
                        else:
                            return 0.0
                    return float(curr)
                except Exception:
                    return 0.0

            for item in array_obj:
                x_list.append(eval_item(item, layer_config.x_expr))
                y_list.append(eval_item(item, layer_config.y_expr))
                
            return np.array(x_list), np.array(y_list)
        
        return [], []
