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


import json
import os
from dataclasses import dataclass, asdict
from typing import List, Dict


@dataclass
class RowConfig:
    title: str
    signals: List[str]  # e.g., ["control.throttle", "np.sin(chassis.speed_mps)"]


@dataclass
class DashboardTemplate:
    name: str
    rows: List[RowConfig]


class TemplateManager:
    """Manages creation, loading, and saving of dashboard templates."""

    def __init__(self, path: str):
        self.path = path
        self.templates: Dict[str, DashboardTemplate] = {}
        self.load()

    def load(self):
        if not os.path.exists(self.path):
            self._init_defaults()
            self.save()
        else:
            try:
                with open(self.path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                for k, v in data.items():
                    rows = [
                        RowConfig(
                            title=r.get("title", ""), signals=r.get("signals", [])
                        )
                        for r in v.get("rows", [])
                    ]
                    self.templates[k] = DashboardTemplate(name=k, rows=rows)
                if "control_feedback" not in self.templates:
                    self._init_defaults()
                    self.save()
            except Exception:
                self._init_defaults()

    def save(self):
        with open(self.path, "w", encoding="utf-8") as f:
            data = {k: asdict(v) for k, v in self.templates.items()}
            json.dump(data, f, indent=2, ensure_ascii=False)

    def _init_defaults(self):
        self.templates["control_feedback"] = DashboardTemplate(
            "control_feedback",
            [
                RowConfig(
                    "Speed + auto mode", ["chassis.speed_mps", "chassis.driving_mode"]
                ),
                RowConfig(
                    "Throttle", ["control.throttle", "chassis.throttle_percentage"]
                ),
                RowConfig("Brake", ["control.brake", "chassis.brake_percentage"]),
                RowConfig(
                    "Steering",
                    ["control.steering_target", "chassis.steering_percentage"],
                ),
            ],
        )
        self.templates["lat_control"] = DashboardTemplate(
            "lat_control",
            [
                RowConfig(
                    "Scenario",
                    ["chassis.speed_mps", "control.debug.simple_lat_debug.curvature"],
                ),
                RowConfig(
                    "Core error",
                    [
                        "control.debug.simple_lat_debug.lateral_error",
                        "control.debug.simple_lat_debug.heading_error",
                    ],
                ),
                RowConfig(
                    "Steering output",
                    [
                        "control.debug.simple_lat_debug.steer_angle",
                        "control.debug.simple_lat_debug.steer_angle_feedforward",
                        "control.debug.simple_lat_debug.steer_angle_feedback",
                    ],
                ),
                RowConfig(
                    "Feedback attribution",
                    [
                        "control.debug.simple_lat_debug.steer_angle_lateral_contribution",
                        "control.debug.simple_lat_debug.steer_angle_heading_contribution",
                    ],
                ),
            ],
        )
        self.templates["lon_control"] = DashboardTemplate(
            "lon_control",
            [
                RowConfig(
                    "Speed tracking",
                    [
                        "control.debug.simple_lon_debug.speed_reference",
                        "chassis.speed_mps",
                    ],
                ),
                RowConfig(
                    "Acceleration cmd",
                    [
                        "control.debug.simple_lon_debug.acceleration_cmd",
                        "control.debug.simple_lon_debug.acceleration_lookup",
                    ],
                ),
                RowConfig(
                    "Core errors",
                    [
                        "control.debug.simple_lon_debug.station_error",
                        "control.debug.simple_lon_debug.speed_error",
                    ],
                ),
                RowConfig(
                    "Actuator",
                    [
                        "control.debug.simple_lon_debug.throttle_cmd",
                        "control.debug.simple_lon_debug.brake_cmd",
                    ],
                ),
            ],
        )
        self.templates["pose_analysis"] = DashboardTemplate(
            "pose_analysis",
            [
                RowConfig(
                    "Position x,y", ["pose.pose.position.x", "pose.pose.position.y"]
                ),
                RowConfig("Heading", ["pose.pose.heading"]),
                RowConfig(
                    "Linear Velocity XYZ",
                    [
                        "pose.pose.linear_velocity.x",
                        "pose.pose.linear_velocity.y",
                        "np.sqrt(pose.pose.linear_velocity.x**2 + pose.pose.linear_velocity.y**2) * 3.6  # XY speed km/h",
                    ],
                ),
            ],
        )
