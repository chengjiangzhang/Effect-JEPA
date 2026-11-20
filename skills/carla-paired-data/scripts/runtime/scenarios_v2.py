"""Offline CARLA/OpenDRIVE scene plans for four controlled interaction families.

The plans borrow interaction categories, not official Bench2Drive scripts.
No server connection, world mutation, model score, or observed future is used
to choose a scene. Every vehicle path is an actual CARLA driving-lane route.
"""
from __future__ import annotations

import hashlib
import itertools
import math
from pathlib import Path

import carla
import numpy as np


FAMILIES = ("rear_following", "adjacent_parallel", "unprotected_intersection", "merging")
# Audited, geographically distinct main-road entry locations. Ordering preserves
# the pilot's scene16/17/18 roads while replacing scene15 and scene19. Selection
# is topology/scene-content based, never conditional on future reaction size.
MERGE_ANCHORS = (("Town04", 1061), ("Town04", 764), ("Town04", 1082),
                 ("Town04", 1176), ("Town04", 1593))
# At this junction both connector routes share nearly 30 m of centreline
# before their OpenDRIVE endpoints. Anchor arrival times at the physical join,
# otherwise the nominal approach vehicles are already overlapping at spawn.
MERGE_CENTERLINE_ANCHORS = {("Town04", 1061)}
MERGE_SELECTION_NOTES = (
    {"map": "Town04", "junction_id": 459,
     "reason": "Excluded service-road/car-wash entry: one same-direction driving lane on both approaches and exit; not a multi-lane main-road merge."},
    {"map": "Town04", "junction_id": 1593,
     "reason": "Selected with corrected roles: the continuous four-lane main road bends while its two-lane entry is straighter. Lane-bundle topology, not smallest turn angle, identifies ego's main road."},
    {"map": "Town04", "junction_id": 1061,
     "reason": "Selected with physical-join arrival anchor: independent single-lane ramp into four-lane main road. OpenDRIVE connector endpoints occur about 29 m after centreline convergence; the old endpoint anchor caused overlapping initial vehicles."},
    {"map": "Town06", "junction_id": None,
     "reason": "Excluded: local OpenDRIVE files exist but live get_available_maps/load_world confirmed the rendered Town06 map is not installed. Offline geometry alone does not establish collectability."},
)


def angle(a, b):
    return abs((float(a) - float(b) + 180) % 360 - 180)


def wp_key(w):
    return int(w.road_id), int(w.section_id), int(w.lane_id), round(float(w.s), 3)


def row(w):
    t = w.transform
    return [float(t.location.x), float(t.location.y), float(t.location.z), float(t.rotation.yaw)]


def choose_step(w, distance, backward=False):
    options = list(w.previous(distance) if backward else w.next(distance))
    options = [o for o in options if o.lane_type == carla.LaneType.Driving]
    if not options:
        return None
    return min(options, key=lambda o: (angle(o.transform.rotation.yaw, w.transform.rotation.yaw), wp_key(o)))


def extend(w, length, backward=False):
    values = []
    current = w
    for _ in range(int(math.ceil(length))):
        current = choose_step(current, 1.0, backward)
        if current is None:
            return None
        values.append(current)
    return values


def trace_connector(start, end):
    points = [start]
    current = start
    for _ in range(180):
        if current.transform.location.distance(end.transform.location) < 1.2:
            points.append(end)
            return points
        options = list(current.next(1.0))
        if not options:
            return None
        current = min(options, key=lambda w: (0 if w.road_id == end.road_id else 1,
                     w.transform.location.distance(end.transform.location), wp_key(w)))
        points.append(current)
    return None


def path_from_connector(start, end):
    middle = trace_connector(start, end)
    before, after = extend(start, 70, True), extend(end, 100)
    if middle is None or before is None or after is None:
        return None
    points = list(reversed(before)) + middle + after
    values = np.asarray([row(w) for w in points])
    # Keep path continuous and remove duplicate endpoint samples.
    keep = np.r_[True, np.linalg.norm(np.diff(values[:, :3], axis=0), axis=1) > .01]
    values = values[keep]
    if np.max(np.linalg.norm(np.diff(values[:, :3], axis=0), axis=1)) > 2.1:
        return None
    return values.tolist()


def cumulative(route):
    values = np.asarray(route)
    return np.r_[0., np.cumsum(np.linalg.norm(np.diff(values[:, :2], axis=0), axis=1))]


def nearest_s(route, point):
    values = np.asarray(route)
    idx = int(np.argmin(np.linalg.norm(values[:, :2] - np.asarray(point)[:2], axis=1)))
    return float(cumulative(route)[idx])


def point_at_s(route, s):
    values = np.asarray(route)
    distance = cumulative(route)
    if s < 2 or s > distance[-1] - 10:
        raise ValueError("Requested vehicle placement leaves insufficient route")
    index = int(np.searchsorted(distance, s))
    fraction = (s - distance[index-1]) / max(distance[index] - distance[index-1], 1e-6)
    xyz = values[index-1, :3] * (1-fraction) + values[index, :3] * fraction
    yaw = values[index-1, 3] + fraction * ((values[index, 3]-values[index-1, 3]+180) % 360 - 180)
    return {"position": [float(xyz[0]), float(xyz[1]), float(xyz[2] + .4)],
            "rotation_deg": [0., 0., float(yaw)]}


def parallel_candidates(cmap, map_name):
    output = []
    for w in sorted(cmap.generate_waypoints(12), key=wp_key):
        if w.is_junction or w.lane_type != carla.LaneType.Driving:
            continue
        adjacent = []
        for side, other in ((-1, w.get_left_lane()), (1, w.get_right_lane())):
            if other is not None and other.lane_type == carla.LaneType.Driving and not other.is_junction and angle(other.transform.rotation.yaw, w.transform.rotation.yaw) < 3:
                adjacent.append((side, other))
        if not adjacent:
            continue
        side, other = adjacent[0]
        before, after = extend(w, 35, True), extend(w, 105)
        obefore, oafter = extend(other, 35, True), extend(other, 105)
        if any(value is None for value in (before, after, obefore, oafter)):
            continue
        own_route = [row(v) for v in list(reversed(before)) + [w] + after]
        adjacent_route = [row(v) for v in list(reversed(obefore)) + [other] + oafter]
        future = np.asarray(own_route)[35:105]
        if max(angle(yaw, w.transform.rotation.yaw) for yaw in future[:, 3]) > 18:
            continue
        separation = np.linalg.norm(np.asarray(own_route)[35:105, :2] - np.asarray(adjacent_route)[35:105, :2], axis=1)
        if separation.min() < 2.8 or separation.max() > 4.5:
            continue
        origin = row(w)
        if any(np.linalg.norm(np.asarray(origin[:2]) - np.asarray(v["origin"][:2])) < 32 for v in output):
            continue
        output.append({"map": map_name, "origin": origin[:3], "road_id": int(w.road_id), "lane_id": int(w.lane_id),
                       "lane_width": float(w.lane_width), "own_route": own_route, "adjacent_route": adjacent_route,
                       "adjacent_side": side, "own_anchor_s": nearest_s(own_route, origin),
                       "adjacent_anchor_s": nearest_s(adjacent_route, row(other))})
    return output


def same_direction_driving_lanes(waypoint):
    """Count physical driving lanes, excluding opposite traffic and shoulders."""
    result = set()
    yaw = waypoint.transform.rotation.yaw
    for side in ("get_left_lane", "get_right_lane"):
        current, visited = waypoint, set()
        while current is not None:
            identity = int(current.road_id), int(current.section_id), int(current.lane_id)
            if identity in visited:
                break
            visited.add(identity)
            if current.lane_type == carla.LaneType.Driving and angle(current.transform.rotation.yaw, yaw) < 30:
                result.add(identity[2])
            current = getattr(current, side)()
    return sorted(result)


def merge_path_topology(start, end):
    incoming, outgoing = list(start.previous(3)), list(end.next(3))
    if len(incoming) != 1 or len(outgoing) != 1:
        return None
    incoming, outgoing = incoming[0], outgoing[0]
    return {"incoming_road_id": int(incoming.road_id), "incoming_lane_id": int(incoming.lane_id),
            "incoming_is_junction": bool(incoming.is_junction),
            "incoming_same_direction_lanes": same_direction_driving_lanes(incoming),
            "outgoing_road_id": int(outgoing.road_id), "outgoing_lane_id": int(outgoing.lane_id),
            "outgoing_same_direction_lanes": same_direction_driving_lanes(outgoing),
            "connector_road_id": int(start.road_id), "connector_lane_id": int(start.lane_id),
            "connector_turn_deg": float(angle(start.transform.rotation.yaw, end.transform.rotation.yaw))}


def orient_main_road_merge(a, b):
    """Main road preserves its wider lane bundle; the entry need not be curved.

    A shallow angle/shared exit alone also accepts service drives and two turns
    entering the same road. Requiring a continuous wider multi-lane bundle
    excludes those cases and handles a curved main road without swapping roles.
    """
    for main, entry in ((a,b), (b,a)):
        mt, et = main.get("topology"), entry.get("topology")
        if mt is None or et is None:
            continue
        nmain, nentry = len(mt["incoming_same_direction_lanes"]), len(et["incoming_same_direction_lanes"])
        if (not mt["incoming_is_junction"] and not et["incoming_is_junction"]
                and mt["incoming_road_id"] != et["incoming_road_id"]
                and nmain >= 2 and nmain == len(mt["outgoing_same_direction_lanes"])
                and 1 <= nentry < nmain):
            return main, entry
    return None


def physical_merge_join(main_route, entry_route, distance_m=.35, heading_deg=5., sustained_samples=12):
    """First sustained centreline convergence inside the entry connector.

    Routes have a 70-sample incoming prefix and 100-sample outgoing suffix.
    Project each entry point onto the main polyline, rather than comparing
    nearest sampled points (whose longitudinal discretization can be 0.5 m).
    The heading and sustained-match tests reject a transient road crossing.
    """
    main, entry = np.asarray(main_route), np.asarray(entry_route)
    segments = main[1:, :3]-main[:-1, :3]
    lengths_squared = (segments*segments).sum(axis=1)
    if len(entry) <= 170 or not len(segments) or np.any(lengths_squared <= 1e-12):
        raise ValueError("Physical join requires continuous non-degenerate extended routes")
    projections, valid = [], []
    for point in entry:
        fractions = np.clip(((point[:3]-main[:-1,:3])*segments).sum(axis=1)/lengths_squared, 0., 1.)
        closest = main[:-1,:3]+fractions[:,None]*segments
        distances = np.linalg.norm(closest-point[:3],axis=1)
        index = int(np.argmin(distances))
        yaw = main[index,3]+fractions[index]*((main[index+1,3]-main[index,3]+180)%360-180)
        heading = angle(point[3],yaw)
        projections.append((closest[index],float(distances[index]),float(heading)))
        valid.append(distances[index] <= distance_m and heading <= heading_deg)
    for index in range(70,len(entry)-100):
        if len(valid[index:index+sustained_samples]) == sustained_samples and all(valid[index:index+sustained_samples]):
            projected,distance,heading = projections[index]
            return ((entry[index,:3]+projected)/2).tolist(), {
                "type": "first_sustained_centerline_join", "entry_route_index": index,
                "max_distance_m": float(distance_m), "max_heading_difference_deg": float(heading_deg),
                "sustained_route_samples": int(sustained_samples),
                "measured_distance_m": distance, "measured_heading_difference_deg": heading}
    raise ValueError("No sustained physical centreline join inside the declared entry connector")


def junction_candidates(cmap, map_name, kind):
    junctions = {w.junction_id: w.get_junction() for w in cmap.generate_waypoints(8) if w.is_junction}
    results = []
    for jid, junction in sorted(junctions.items()):
        paths = []
        for start, end in sorted(junction.get_waypoints(carla.LaneType.Driving), key=lambda pair: wp_key(pair[0])):
            route = path_from_connector(start, end)
            exits = list(end.next(3))
            if route is None or len(exits) != 1:
                continue
            paths.append({"start": start, "end": end, "route": route,
                          "exit": (int(exits[0].road_id), int(exits[0].lane_id)),
                          "turn": angle(start.transform.rotation.yaw, end.transform.rotation.yaw),
                          "topology": merge_path_topology(start, end) if kind == "merging" else None})
        candidates = []
        for a, b in itertools.combinations(paths, 2):
            if a["start"].road_id == b["start"].road_id:
                continue
            pa, pb = np.asarray(a["route"]), np.asarray(b["route"])
            if kind == "unprotected_intersection":
                if a["turn"] > 18 or b["turn"] > 18:
                    continue
                crossing_angle = angle(a["start"].transform.rotation.yaw, b["start"].transform.rotation.yaw)
                if not 65 <= crossing_angle <= 115:
                    continue
                # Only connector interiors define this conflict, not remote roads.
                ia0, ib0 = 70, 70
                ia1, ib1 = len(pa) - 100, len(pb) - 100
                distances = np.linalg.norm(pa[ia0:ia1, None, :2] - pb[None, ib0:ib1, :2], axis=-1)
                if not distances.size or distances.min() > .85:
                    continue
                ia, ib = np.unravel_index(np.argmin(distances), distances.shape)
                ia, ib = ia + ia0, ib + ib0
                if abs(pa[ia, 2] - pb[ib, 2]) > .5:
                    continue
                conflict = (pa[ia, :3] + pb[ib, :3]) / 2
                score = abs(crossing_angle - 90)
            else:
                if a["exit"] != b["exit"]:
                    continue
                oriented = orient_main_road_merge(a,b)
                if oriented is None:
                    continue
                a,b = oriented
                approach_angle = angle(a["start"].transform.rotation.yaw, b["start"].transform.rotation.yaw)
                if not 5 <= approach_angle <= 65:
                    continue
                if a["start"].transform.location.distance(b["start"].transform.location) < 5:
                    continue
                # The two paths must really converge onto the same outgoing lane.
                end_a, end_b = np.asarray(row(a["end"])), np.asarray(row(b["end"]))
                # CARLA returns connector endpoint samples up to ~2 m before
                # the shared outgoing lane; their longitudinal offsets differ.
                # Same outgoing road/lane topology is the actual merge test.
                if np.linalg.norm(end_a[:3] - end_b[:3]) > 4:
                    continue
                conflict = (end_a[:3] + end_b[:3]) / 2
                join_evidence = None
                if (map_name,int(jid)) in MERGE_CENTERLINE_ANCHORS:
                    physical,join_evidence = physical_merge_join(a["route"],b["route"])
                    join_evidence["connector_endpoint_xyz"] = conflict.tolist()
                    join_evidence["distance_from_connector_endpoint_m"] = float(np.linalg.norm(np.asarray(physical)-conflict))
                    conflict = np.asarray(physical)
                score = a["turn"] + abs(approach_angle - 25) * .1
            candidates.append({"score": score, "map": map_name, "junction_id": int(jid),
                               "ego_route": a["route"], "npc_route": b["route"], "conflict_xyz": conflict.tolist(),
                               "ego_conflict_s": nearest_s(a["route"], conflict),
                               "npc_conflict_s": nearest_s(b["route"], conflict),
                               "ego_connector_road_id": int(a["start"].road_id),
                               "npc_connector_road_id": int(b["start"].road_id),
                               "outgoing_lane": list(a["exit"]), "geometry": kind})
            if kind == "merging":
                candidates[-1]["main_road_topology"] = a["topology"]
                candidates[-1]["entry_road_topology"] = b["topology"]
                candidates[-1]["approach_angle_deg"] = float(approach_angle)
                if join_evidence is not None:
                    candidates[-1]["physical_join_anchor"] = join_evidence
        if candidates:
            results.append(min(candidates, key=lambda value: value["score"]))
    return results


def vehicle(role, route, spawn_s, speed, desired_speed=None, priority=0, color="210,210,210"):
    return {"role": role, "blueprint_id": "vehicle.tesla.model3",
            "blueprint_attributes": {"role_name": role, "color": color},
            "transform": point_at_s(route, spawn_s), "route": route, "spawn_route_s": float(spawn_s),
            "initial_speed_mps": float(speed), "desired_speed_mps": float(speed if desired_speed is None else desired_speed),
            "priority": int(priority)}


def spawn_footprints_overlap(a, b):
    """Separating-axis test with conservative 5.6 x 2.3 m rectangles."""
    axes = []
    for value in (a, b):
        yaw = math.radians(value["transform"]["rotation_deg"][2])
        forward = np.array([math.cos(yaw), math.sin(yaw)])
        axes.append((forward, np.array([-forward[1], forward[0]])))
    delta = np.asarray(a["transform"]["position"][:2])-np.asarray(b["transform"]["position"][:2])
    for axis in (*axes[0], *axes[1]):
        radius = sum(2.8*abs(axis@forward)+1.15*abs(axis@right) for forward, right in axes)
        if abs(delta@axis) >= radius:
            return False
    return True


def finish_config(family, index, seed, vehicles, geometry, history_s, driving_warmup_s):
    return {"version": "paired-multicam-v2", "scene_id": f"scene_{index:04d}", "scene_index": index,
            "family": family, "map": geometry["map"], "seed": seed + index, "base_seed": seed,
            "vehicle_roles": [v["role"] for v in vehicles], "vehicles": vehicles,
            "geometry": geometry, "history_s": float(history_s), "driving_warmup_s": float(driving_warmup_s),
            "duration_s": 4.0, "dt_s": .05, "saved_fps": 2.0,
            "branches": ["maintain", "brake", "accelerate", "left", "right", "maintain_repeat"],
            "action": {"lateral_offset_m": 1.0, "lateral_onset_s": .2, "lateral_duration_s": 2.5,
                       "braking_deceleration_m_s2": 4.5, "acceleration_m_s2": 1.5,
                       "acceleration_speed_increase_cap_m_s": 5.0,
                       "reference": "measured ego XY speed at shared t0; offsets along route local normal"},
            "controller": {"type": "custom deterministic IDM and geometric arrival-time yielding with physical VehicleControl",
                           "idm": {"max_acceleration": 2.0, "comfortable_brake": 2.8, "time_headway": 1.25, "minimum_gap": 2.5},
                           "lateral_margin_m": .65, "conflict_time_gap_s": 1.7, "yield_lookahead_s": 3.0},
            "traffic_lights": "disable/freeze Off for controlled unprotected interaction; custom right-of-way is recorded",
            "weather": {"cloudiness": float((index % 3) * 10), "precipitation": 0., "sun_altitude_angle": 50.,
                        "sun_azimuth_angle": float(20 + (index % 4) * 35), "fog_density": 0., "wetness": 0.},
            "provenance": {"official_bench2drive": False,
                           "description": "custom CARLA route-based interaction families; not official Bench2Drive route/scenario replay",
                           "selection": "offline driving-lane topology and declared parameter variants only, no model or observed future scores"}}


def local_lane_geometry(waypoints, center, radius=120.):
    """Group sampled road centre lines without joining different lane sections."""
    groups = {}
    center = np.asarray(center)[:2]
    for waypoint in waypoints:
        value = row(waypoint)
        if np.linalg.norm(np.asarray(value[:2])-center) > radius:
            continue
        key = (int(waypoint.road_id), int(waypoint.lane_id), int(waypoint.section_id))
        if key not in groups:
            groups[key] = {"road_id": key[0], "lane_id": key[1], "section_id": key[2],
                           "is_junction": bool(waypoint.is_junction), "width": float(waypoint.lane_width),
                           "samples": []}
        groups[key]["samples"].append((float(waypoint.s), value))
    result = []
    for key, value in sorted(groups.items()):
        samples = sorted(value.pop("samples"), key=lambda item: item[0])
        value["points"] = [point for _, point in samples]
        value["opendrive_s"] = [s for s, _ in samples]
        result.append(value)
    return result


def build_scenes(opendrive_root, seed=20261013, count_per_family=5, history_s=2.0, driving_warmup_s=1.0):
    """Return JSON-serializable scene plans, using local XODR without any server."""
    if not 1 <= count_per_family <= len(MERGE_ANCHORS):
        raise ValueError("This pilot has exactly five reviewed merge anchors; do not silently invent more locations")
    root = Path(opendrive_root)
    maps = {}
    for name in ("Town10HD", "Town05", "Town03", "Town04"):
        path = root / f"{name}.xodr"
        if path.exists():
            maps[name] = carla.Map(name, path.read_text())
    if not maps:
        raise FileNotFoundError(f"No supported Town OpenDRIVE files under {root}")
    rng = np.random.default_rng(seed)
    parallels = []
    for name in ("Town10HD", "Town05", "Town04"):
        if name in maps:
            candidates = parallel_candidates(maps[name], name)
            # Geographically spaced candidates; alternate maps when available.
            parallels.extend(candidates[i] for i in rng.permutation(len(candidates)))
    if len(parallels) < count_per_family * 2:
        raise RuntimeError(f"Only {len(parallels)} independent parallel-road anchors available")
    crosses = []
    for name in ("Town10HD", "Town03", "Town05"):
        if name in maps:
            crosses.extend(junction_candidates(maps[name], name, "unprotected_intersection"))
    merges = []
    for name in ("Town04",):
        if name in maps:
            merges.extend(junction_candidates(maps[name], name, "merging"))
    merge_by_id = {(c["map"],c["junction_id"]):c for c in merges}
    missing = [anchor for anchor in MERGE_ANCHORS[:count_per_family] if anchor not in merge_by_id]
    if missing:
        raise RuntimeError(f"Reviewed main-road merge anchors unavailable or failed topology checks: {missing}")
    if len(crosses) < count_per_family or len(merges) < count_per_family:
        raise RuntimeError(f"Need {count_per_family} distinct junctions each; crossing={len(crosses)}, merge={len(merges)}")
    scenes = []
    map_lane_samples = {}
    used_junctions = {family: set() for family in ("unprotected_intersection", "merging")}
    pre_t0 = history_s + driving_warmup_s
    for family in FAMILIES:
        for variant in range(count_per_family):
            index = len(scenes)
            if family in ("rear_following", "adjacent_parallel"):
                sample = parallels[variant + (count_per_family if family == "adjacent_parallel" else 0)]
                route, adjacent = sample["own_route"], sample["adjacent_route"]
                own_s, adj_s = sample["own_anchor_s"], sample["adjacent_anchor_s"]
                speed = 5.5 + .4 * (variant % 4)
                front_gap = 28 + 2 * (variant % 3)
                rear_gap = 13 + (variant % 3)
                adjacent_front = 7 if family == "adjacent_parallel" else 22
                vehicles = [vehicle("ego", route, own_s, speed, priority=0, color="195,35,30"),
                            vehicle("lead", route, own_s + front_gap, speed + .5, priority=0, color="220,220,220"),
                            vehicle("rear", route, own_s - rear_gap, speed, speed + 1.2, priority=1, color="35,80,190"),
                            vehicle("adjacent_front", adjacent, adj_s + adjacent_front, speed, speed + .5, priority=1, color="220,170,35"),
                            vehicle("adjacent_rear", adjacent, adj_s - 9, speed, speed + 1.0, priority=1, color="30,160,160")]
                geometry = {k: v for k, v in sample.items() if k not in ("own_route", "adjacent_route")}
                geometry.update(type="same-direction adjacent driving lanes", conflicts=[])
            else:
                speed = (4.8 if family == "unprotected_intersection" else 6.0) + .25 * (variant % 3)
                npc_speed = speed + .3
                npc_role = "cross_traffic" if family == "unprotected_intersection" else "merging_traffic"
                rejected_geometry = []
                sample = None
                # A failed declared merge must fail the plan, not silently move
                # to a different site in response to collection outcomes.
                candidates = crosses if family == "unprotected_intersection" else [merge_by_id[MERGE_ANCHORS[variant]]]
                for proposed in candidates:
                    identity = proposed["map"], proposed["junction_id"]
                    if identity in used_junctions[family]:
                        continue
                    ego_route, npc_route = proposed["ego_route"], proposed["npc_route"]
                    ego_s = proposed["ego_conflict_s"] - speed * (pre_t0 + 2.5)
                    npc_s = proposed["npc_conflict_s"] - npc_speed * (pre_t0 + 2.6)
                    try:
                        vehicles = [vehicle("ego", ego_route, ego_s, speed, priority=0, color="195,35,30"),
                                    vehicle("rear", ego_route, ego_s - 13, speed, speed + .7, priority=1, color="35,80,190"),
                                    vehicle(npc_role, npc_route, npc_s, npc_speed, priority=2, color="220,170,35"),
                                    vehicle(npc_role + "_rear", npc_route, npc_s - 14, npc_speed, npc_speed + .5, priority=3, color="30,160,160")]
                    except ValueError:
                        rejected_geometry.append({"map":identity[0], "junction_id":identity[1], "reason":"insufficient approach route"})
                        continue
                    if any(spawn_footprints_overlap(a,b) for a,b in itertools.combinations(vehicles,2)):
                        rejected_geometry.append({"map":identity[0], "junction_id":identity[1], "reason":"initial vehicle footprint overlap"})
                        continue
                    sample = proposed
                    used_junctions[family].add(identity)
                    break
                if sample is None:
                    raise RuntimeError(f"Insufficient non-overlapping initial approaches for {family}/{variant}: {rejected_geometry}")
                geometry = {k: v for k, v in sample.items() if k not in ("ego_route", "npc_route")}
                geometry["rejected_geometry_candidates"] = rejected_geometry
                geometry["conflicts"] = [{"roles": ["ego", npc_role], "point": sample["conflict_xyz"],
                                          "route_s": {"ego": sample["ego_conflict_s"], npc_role: sample["npc_conflict_s"]},
                                          "priority_role": "ego", "kind": family}]
                geometry["nominal_conflict_after_t0_s"] = {"ego": 2.5, npc_role: 2.6}
                if family == "merging":
                    geometry["selection"] = {"anchor": list(MERGE_ANCHORS[variant]),
                        "reason": "Distinct audited multi-lane through main road plus an independent narrower entry; ego follows main road; shared outgoing road/lane confirmed.",
                        "candidate_decisions": list(MERGE_SELECTION_NOTES)}
            positions = np.array([v["transform"]["position"] for v in vehicles])
            minimum = min(np.linalg.norm(a[:2]-b[:2]) for a, b in itertools.combinations(positions, 2))
            if any(spawn_footprints_overlap(a, b) for a, b in itertools.combinations(vehicles, 2)):
                raise RuntimeError(f"Conservative spawn rectangles overlap for {family}/{variant}; minimum centres {minimum:.2f} m")
            config = finish_config(family, index, seed, vehicles, geometry, history_s, driving_warmup_s)
            source = root / (config["map"] + ".xodr")
            config["provenance"]["opendrive_sha256"] = hashlib.sha256(source.read_bytes()).hexdigest()
            config["provenance"]["geometry_min_spawn_separation_m"] = float(minimum)
            map_name = config["map"]
            if map_name not in map_lane_samples:
                map_lane_samples[map_name] = maps[map_name].generate_waypoints(2.)
            ego = vehicles[0]
            nominal_t0 = point_at_s(ego["route"], ego["spawn_route_s"]+speed*pre_t0)["position"]
            config["local_map_lanes"] = local_lane_geometry(map_lane_samples[map_name], nominal_t0)
            config["local_map_geometry"] = {"center_xyz": nominal_t0, "radius_m": 120., "sample_distance_m": 2.,
                                            "point_order": "increasing OpenDRIVE s within road/lane/section"}
            config["map_geometry_path"] = f"maps/{map_name}.xodr"
            scenes.append(config)
    return scenes


# A separate city profile: keep build_scenes and its historical geometry intact.
# These locations are declared from road topology, not selected from simulated
# reactions or model scores. The first ten anchors preserve the prior Town10HD
# following/parallel locations while making their identity explicit.
CITY_PARALLEL_ANCHORS = (
    (21, 4, 57.436, 16.844), (10, -2, 31.810, -67.910),
    (6, -2, -78.972, -57.094), (14, -2, -41.661, 51.706),
    (20, 4, 19.306, 16.738), (10, -2, 68.142, -67.835),
    (18, -2, -88.686, 27.938), (0, 1, 102.930, -9.382),
    (2, -2, 109.503, 53.293), (20, 4, -16.694, 16.638),
)
# (junction, ego incoming road/lane, ego outgoing road/lane,
#            NPC incoming road/lane, NPC outgoing road/lane).
CITY_CROSSING_ANCHORS = (
    (23, (6, -2), (7, -2), (7, 1), (22, -1)),
    (134, (11, -1), (12, -1), (12, 1), (9, -1)),
    (189, (14, -2), (15, -2), (15, 1), (20, -2)),
    (532, (3, 2), (2, 2), (2, -1), (21, 4)),
    (719, (5, -1), (6, -1), (6, 1), (18, -2)),
)
CITY_MERGE_ANCHORS = (
    (664, (2, 2), (1, 2), (11, 1), (1, 2)),
    (841, (20, -2), (21, -2), (9, -1), (21, -2)),
    (895, (13, -2), (14, -2), (12, -1), (14, -2)),
    (23, (6, -2), (7, -2), (22, 2), (7, -2)),
    (532, (3, 1), (2, 1), (21, -2), (2, 1)),
)


def city_connector(junction, incoming, outgoing):
    """Resolve one declared lane connection; never silently choose another."""
    matches = []
    for start, end in sorted(junction.get_waypoints(carla.LaneType.Driving),
                             key=lambda pair: wp_key(pair[0])):
        topology = merge_path_topology(start, end)
        if topology is None:
            continue
        identity = ((topology['incoming_road_id'], topology['incoming_lane_id']),
                    (topology['outgoing_road_id'], topology['outgoing_lane_id']))
        if identity != (tuple(incoming), tuple(outgoing)):
            continue
        route = path_from_connector(start, end)
        if route is None or topology['incoming_is_junction']:
            raise ValueError(f'Declared city connection has no independent continuous approach: {identity}')
        matches.append({'route': route, 'topology': topology,
                        'start_yaw': float(start.transform.rotation.yaw),
                        'signed_turn_deg': float((end.transform.rotation.yaw-start.transform.rotation.yaw+180) % 360-180)})
    if len(matches) != 1:
        raise ValueError(f'Declared city connection must be unique: {incoming}->{outgoing}; found {len(matches)}')
    return matches[0]


def city_route_point(route, distance):
    values, along = np.asarray(route), cumulative(route)
    if not 0 <= distance <= along[-1]:
        raise ValueError('City geometry check exceeds the declared route')
    return np.array([np.interp(distance, along, values[:, k]) for k in range(3)])


def city_project_s(route, point):
    values = np.asarray(route)
    segments = np.diff(values[:, :2], axis=0)
    squared = (segments*segments).sum(axis=1)
    fractions = np.clip(((np.asarray(point)[:2]-values[:-1, :2])*segments).sum(axis=1)/squared, 0., 1.)
    nearest = values[:-1, :2]+fractions[:, None]*segments
    index = int(np.argmin(np.linalg.norm(nearest-np.asarray(point)[:2], axis=1)))
    return float(cumulative(route)[index]+fractions[index]*math.sqrt(squared[index]))


def city_crossing_geometry(ego, npc):
    """A through route crossed by an opposing left turn, then different exits."""
    et, nt = ego['topology'], npc['topology']
    if (et['incoming_road_id'] == nt['incoming_road_id']
            or (et['outgoing_road_id'], et['outgoing_lane_id']) == (nt['outgoing_road_id'], nt['outgoing_lane_id'])
            or abs(ego['signed_turn_deg']) > 18
            or not -120 < npc['signed_turn_deg'] < -45
            or angle(ego['start_yaw'], npc['start_yaw']) < 150):
        raise ValueError('City crossing requires ego through, opposing NPC left turn, and separate exits')
    p, q = np.asarray(ego['route']), np.asarray(npc['route'])
    ps, qs = cumulative(p), cumulative(q)
    hits = []
    cross = lambda a, b: float(a[0]*b[1]-a[1]*b[0])
    # Only connector interiors, excluding the 70 m incoming/100 m outgoing
    # extensions. Exact segment intersections avoid sample-distance artefacts.
    for i in range(70, len(p)-101):
        a = p[i+1, :2]-p[i, :2]
        for j in range(70, len(q)-101):
            b = q[j+1, :2]-q[j, :2]
            denominator = cross(a, b)
            if abs(denominator) < 1e-8:
                continue
            delta = q[j, :2]-p[i, :2]
            u, v = cross(delta, b)/denominator, cross(delta, a)/denominator
            if not (0 <= u <= 1 and 0 <= v <= 1):
                continue
            xyz_p = p[i, :3]+u*(p[i+1, :3]-p[i, :3])
            xyz_q = q[j, :3]+v*(q[j+1, :3]-q[j, :3])
            tangent_angle = math.degrees(math.acos(float(np.clip(a@b/(np.linalg.norm(a)*np.linalg.norm(b)), -1., 1.))))
            if abs(xyz_p[2]-xyz_q[2]) > .5 or not 30 <= tangent_angle <= 150:
                continue
            es, ns = float(ps[i]+u*np.linalg.norm(a)), float(qs[j]+v*np.linalg.norm(b))
            separation = float(np.linalg.norm(city_route_point(p, es+20)[:2]-city_route_point(q, ns+20)[:2]))
            if separation < 8:
                continue
            hits.append((es, ns, ((xyz_p+xyz_q)/2).tolist(), tangent_angle, separation))
    if not hits:
        raise ValueError('No same-height transverse crossing followed by separated exits')
    es, ns, point, tangent, separation = min(hits)
    return point, es, ns, {'type': 'opposing_left_turn_across_ego_through_route',
        'intersection_method': 'exact XY polyline segment intersection inside both connectors',
        'crossing_tangent_angle_deg': float(tangent), 'separate_outgoing_lanes': True,
        'distance_along_each_route_after_conflict_m': 20., 'route_separation_after_conflict_m': separation,
        'minimum_required_separation_m': 8., 'npc_must_yield_to': 'ego'}


def city_merge_geometry(ego, npc):
    """Right-turn entry joins the same lane; this is not a highway ramp."""
    et, nt = ego['topology'], npc['topology']
    if (et['incoming_road_id'] == nt['incoming_road_id']
            or (et['outgoing_road_id'], et['outgoing_lane_id']) != (nt['outgoing_road_id'], nt['outgoing_lane_id'])
            or abs(ego['signed_turn_deg']) > 18 or not 45 < npc['signed_turn_deg'] < 120):
        raise ValueError('City entry merge requires a distinct right-turn approach and the same outgoing lane')
    point, evidence = physical_merge_join(ego['route'], npc['route'])
    es, ns = city_project_s(ego['route'], point), city_project_s(npc['route'], point)
    distances = [float(np.linalg.norm(city_route_point(ego['route'], es+d)-city_route_point(npc['route'], ns+d)))
                 for d in np.arange(0., 31., 1.)]
    if max(distances) > .4:
        raise ValueError('City merge does not remain on a shared downstream path for 30 m')
    entry_s = cumulative(npc['route'])
    index = evidence['entry_route_index']
    evidence.update(type='urban_right_turn_entry_merge', not_highway_ramp=True,
                    sustained_match_length_m=float(entry_s[index+11]-entry_s[index]),
                    downstream_check_length_m=30., downstream_sample_distance_m=1.,
                    maximum_downstream_route_separation_m=max(distances),
                    maximum_allowed_downstream_route_separation_m=.4,
                    npc_must_yield_to='ego')
    return point, es, ns, evidence


def build_city_scenes(opendrive_root, seed=20261013, count_per_family=5,
                      history_s=2.0, driving_warmup_s=1.0):
    """Twenty declared Town10HD city cases; only offline map geometry is read.

    Crossings: ego goes through while an opposing NPC turns left and yields.
    Merges: an NPC turns right from a real street into ego's outgoing lane.
    No official Bench2Drive scenario, highway ramp or outcome-based selection
    is claimed. The existing five-action/history/repeat protocol is retained.
    """
    if not 1 <= count_per_family <= 5:
        raise ValueError('The city pilot declares five locations per family')
    source = Path(opendrive_root)/'Town10HD.xodr'
    cmap = carla.Map('Town10HD', source.read_text())
    map_hash = hashlib.sha256(source.read_bytes()).hexdigest()
    parallels = parallel_candidates(cmap, 'Town10HD')
    selected = []
    for road, lane, x, y in CITY_PARALLEL_ANCHORS:
        matches = [c for c in parallels if c['road_id'] == road and c['lane_id'] == lane
                   and np.linalg.norm(np.asarray(c['origin'][:2])-[x, y]) < .002]
        if len(matches) != 1:
            raise ValueError(f'Declared city parallel anchor unavailable: {(road, lane, x, y)}')
        selected.append(matches[0])
    if min(np.linalg.norm(np.asarray(a['origin'][:2])-b['origin'][:2])
           for a, b in itertools.combinations(selected, 2)) < 32:
        raise ValueError('City following/parallel anchors must be distinct and at least 32 m apart')
    junctions = {w.junction_id: w.get_junction() for w in cmap.generate_waypoints(4) if w.is_junction}
    lane_samples = cmap.generate_waypoints(2.)
    pre_t0 = history_s+driving_warmup_s
    scenes = []
    for family in FAMILIES:
        for variant in range(count_per_family):
            index = len(scenes)
            if family in ('rear_following', 'adjacent_parallel'):
                anchor_index = variant+(5 if family == 'adjacent_parallel' else 0)
                sample = selected[anchor_index]
                route, adjacent = sample['own_route'], sample['adjacent_route']
                own_s, adj_s = sample['own_anchor_s'], sample['adjacent_anchor_s']
                speed = 5.5+.4*(variant % 4)
                adjacent_front = 7 if family == 'adjacent_parallel' else 22
                vehicles = [vehicle('ego', route, own_s, speed, priority=0, color='195,35,30'),
                    vehicle('lead', route, own_s+(28+2*(variant % 3)), speed+.5, priority=0, color='220,220,220'),
                    vehicle('rear', route, own_s-(13+(variant % 3)), speed, speed+1.2, priority=1, color='35,80,190'),
                    vehicle('adjacent_front', adjacent, adj_s+adjacent_front, speed, speed+.5, priority=1, color='220,170,35'),
                    vehicle('adjacent_rear', adjacent, adj_s-9, speed, speed+1., priority=1, color='30,160,160')]
                geometry = {k: v for k, v in sample.items() if k not in ('own_route', 'adjacent_route')}
                geometry.update(type='same-direction adjacent city driving lanes', conflicts=[],
                    selection={'predeclared': True, 'anchor': list(CITY_PARALLEL_ANCHORS[anchor_index]),
                               'minimum_anchor_separation_m': 32., 'outcome_based_selection': False})
            else:
                merging = family == 'merging'
                declared = (CITY_MERGE_ANCHORS if merging else CITY_CROSSING_ANCHORS)[variant]
                jid, ego_in, ego_out, npc_in, npc_out = declared
                if jid not in junctions:
                    raise ValueError(f'Declared Town10HD junction unavailable: {jid}')
                ego = city_connector(junctions[jid], ego_in, ego_out)
                npc = city_connector(junctions[jid], npc_in, npc_out)
                point, ego_conflict, npc_conflict, evidence = (
                    city_merge_geometry(ego, npc) if merging else city_crossing_geometry(ego, npc))
                speed = 4.8+.25*(variant % 3)
                # Urban turning traffic starts more slowly than the prior
                # highway approaches; the ego intervention law is unchanged.
                npc_speed = 4.0+.15*(variant % 3)
                npc_role = 'merging_traffic' if merging else 'cross_traffic'
                own_s = ego_conflict-speed*(pre_t0+2.5)
                npc_s = npc_conflict-npc_speed*(pre_t0+2.6)
                vehicles = [vehicle('ego', ego['route'], own_s, speed, priority=0, color='195,35,30'),
                    vehicle('rear', ego['route'], own_s-13, speed, speed+.7, priority=1, color='35,80,190'),
                    vehicle(npc_role, npc['route'], npc_s, npc_speed, priority=2, color='220,170,35'),
                    vehicle(npc_role+'_rear', npc['route'], npc_s-14, npc_speed, npc_speed+.5, priority=3, color='30,160,160')]
                geometry = {'map': 'Town10HD', 'junction_id': int(jid), 'geometry': family,
                    'type': evidence['type'], 'conflict_xyz': point,
                    'ego_conflict_s': ego_conflict, 'npc_conflict_s': npc_conflict,
                    'ego_connector_road_id': ego['topology']['connector_road_id'],
                    'npc_connector_road_id': npc['topology']['connector_road_id'],
                    'main_road_topology': ego['topology'], 'entry_road_topology': npc['topology'],
                    'ego_turn_signed_deg': ego['signed_turn_deg'], 'npc_turn_signed_deg': npc['signed_turn_deg'],
                    'nominal_conflict_after_t0_s': {'ego': 2.5, npc_role: 2.6},
                    'conflicts': [{'roles': ['ego', npc_role], 'point': point,
                        'route_s': {'ego': ego_conflict, npc_role: npc_conflict},
                        'priority_role': 'ego', 'kind': family}],
                    'selection': {'predeclared': True, 'junction_id': int(jid),
                        'ego_incoming_lane': list(ego_in), 'ego_outgoing_lane': list(ego_out),
                        'npc_incoming_lane': list(npc_in), 'npc_outgoing_lane': list(npc_out),
                        'outcome_based_selection': False,
                        'description': 'Right-turn city street entry into the same main-road lane; not a highway ramp.' if merging
                            else 'Opposing NPC left turn across ego through route, yielding to ego; routes leave on separate exits.'}}
                geometry['physical_join_anchor' if merging else 'crossing_evidence'] = evidence
                if merging:
                    geometry['outgoing_lane'] = list(ego_out)
            for a, b in itertools.combinations(vehicles, 2):
                if spawn_footprints_overlap(a, b):
                    raise ValueError(f'Declared city spawn footprints overlap: {family}/{variant}/{a["role"]}/{b["role"]}')
            config = finish_config(family, index, seed, vehicles, geometry, history_s, driving_warmup_s)
            config['scene_profile'] = 'town10_city_v1'
            config['controller']['leader_geometry'] = 'route_projection'
            config['provenance'].update(opendrive_sha256=map_hash,
                description='Custom Town10HD city interactions; not official Bench2Drive replay. City entries are right-turn street merges, not highway ramps.',
                selection='Predeclared road/lane connections and geometric validity only; no observed futures or model scores.',
                geometry_min_spawn_separation_m=float(min(np.linalg.norm(np.asarray(a['transform']['position'][:2])-b['transform']['position'][:2])
                    for a, b in itertools.combinations(vehicles, 2))),
                spawn_footprint_check={'shape': 'oriented rectangle', 'length_m': 5.6, 'width_m': 2.3,
                                       'all_pairs_disjoint': True, 'applies_to_blueprint': 'vehicle.tesla.model3'})
            nominal_t0 = point_at_s(vehicles[0]['route'], vehicles[0]['spawn_route_s']+speed*pre_t0)['position']
            config['local_map_lanes'] = local_lane_geometry(lane_samples, nominal_t0)
            config['local_map_geometry'] = {'center_xyz': nominal_t0, 'radius_m': 120., 'sample_distance_m': 2.,
                                          'point_order': 'increasing OpenDRIVE s within road/lane/section'}
            config['map_geometry_path'] = 'maps/Town10HD.xodr'
            scenes.append(config)
    return scenes


# Corrected bilateral-lane-change preflight: these anchors are declared from
# Town05's OpenDRIVE geometry before any simulation or future is observed.
# This is deliberately a TWO-family, ten-scene subset. Crossing and merging
# require their own real connector geometry and must not be invented by labels.
LANE_CHANGE_FAMILIES = ('rear_following', 'adjacent_parallel')
LANE_CHANGE_ANCHORS = (
    (34, -2, 60.), (34, 3, 220.), (37, -2, 80.),
    (37, 3, 740.), (37, -2, 220.),
    (37, 3, 600.), (37, -2, 400.), (37, 3, 420.),
    (38, -2, 70.), (38, 3, 240.),
)


def paired_adjacent_routes(source_waypoints):
    """Map-derived left/right centres at each source waypoint's own station.

    Independent calls to extend() on three lanes would pair unequal arc-length
    stations on curves. Instead query get_left_lane/get_right_lane for every
    source waypoint. Missing, opposite, shoulder and discontinuous targets are
    rejected, not extrapolated. Returned evidence is JSON serializable.
    """
    points = list(source_waypoints)
    if len(points) < 2:
        raise ValueError('Paired lane routes require at least two source stations')
    source = np.asarray([row(w) for w in points], dtype=float)
    if not np.isfinite(source).all() or any(w.is_junction or w.lane_type != carla.LaneType.Driving for w in points):
        raise ValueError('Source route must remain on non-junction Driving lanes')
    source_keys = [[int(w.road_id), int(w.section_id), int(w.lane_id)] for w in points]
    source_widths = [float(w.lane_width) for w in points]
    if min(source_widths) <= 0:
        raise ValueError('Source lane widths must be positive')
    source_lengths = np.linalg.norm(np.diff(source[:, :3], axis=0), axis=1)
    if np.any(source_lengths <= 1e-5) or np.any(source_lengths > 2.1):
        raise ValueError('Source lane route is degenerate or discontinuous')
    normal = np.column_stack((-np.sin(np.radians(source[:, 3])), np.cos(np.radians(source[:, 3]))))
    forward = np.column_stack((np.cos(np.radians(source[:, 3])), np.sin(np.radians(source[:, 3]))))
    result = {}
    for side, method, sign, permission in (('left', 'get_left_lane', -1, carla.LaneChange.Left),
                                           ('right', 'get_right_lane', 1, carla.LaneChange.Right)):
        targets = []
        for index, w in enumerate(points):
            target = getattr(w, method)()
            if target is None or target.is_junction or target.lane_type != carla.LaneType.Driving:
                raise ValueError(f'{side}: no actual adjacent Driving lane at station {index}')
            if angle(target.transform.rotation.yaw, w.transform.rotation.yaw) > 5:
                raise ValueError(f'{side}: adjacent lane runs in a different direction at station {index}')
            if (int(target.road_id), int(target.section_id)) != (int(w.road_id), int(w.section_id)):
                raise ValueError(f'{side}: adjacent lane is not in the same road section')
            if int(target.lane_id) == int(w.lane_id) or abs(float(target.s)-float(w.s)) > 1e-3:
                raise ValueError(f'{side}: source/target stations do not match')
            if not (int(w.lane_change) & int(permission)):
                raise ValueError(f'{side}: source lane marking disallows this lane change')
            targets.append(target)
        target = np.asarray([row(w) for w in targets], dtype=float)
        target_widths = [float(w.lane_width) for w in targets]
        if not np.isfinite(target).all() or min(target_widths) <= 0:
            raise ValueError(f'{side}: invalid adjacent route or lane widths')
        displacement = target[:, :2]-source[:, :2]
        lateral = np.einsum('ij,ij->i', displacement, normal)
        longitudinal = np.einsum('ij,ij->i', displacement, forward)
        centre_distance = (np.asarray(source_widths)+target_widths)/2
        if np.any(sign*lateral <= 0) or np.max(np.abs(sign*lateral-centre_distance)) > .15:
            raise ValueError(f'{side}: lane centres do not match the declared adjacent side and widths')
        if np.max(np.abs(longitudinal)) > .15:
            raise ValueError(f'{side}: target samples do not share longitudinal source stations')
        segment_lengths = np.linalg.norm(np.diff(target[:, :3], axis=0), axis=1)
        if np.any(segment_lengths <= 1e-5) or np.any(segment_lengths > 2.1):
            raise ValueError(f'{side}: adjacent route is degenerate or discontinuous')
        result[side] = {
            'available': True, 'target_route': target.tolist(),
            'width_m': float(min(target_widths)),
            'evidence': {
                'method': f'CARLA waypoint.{method} at every source OpenDRIVE station',
                'source_lane_keys': source_keys,
                'target_lane_keys': [[int(w.road_id), int(w.section_id), int(w.lane_id)] for w in targets],
                'source_opendrive_s_m': [float(w.s) for w in points],
                'target_opendrive_s_m': [float(w.s) for w in targets],
                'source_lane_widths_m': source_widths,
                'target_lane_widths_m': target_widths,
                'same_direction': True, 'target_lane_type': 'Driving',
                'station_count': len(points), 'lane_change_allowed_at_all_stations': True,
                'source_lane_change_permissions': [str(w.lane_change) for w in points],
                'max_source_target_heading_difference_deg': float(max(angle(a[3], b[3]) for a, b in zip(source, target))),
                'min_signed_lateral_centre_separation_m': float(np.min(sign*lateral)),
                'max_same_station_longitudinal_offset_m': float(np.max(np.abs(longitudinal))),
            },
        }
    return source.tolist(), result


def lane_change_anchor_geometry(cmap, anchor, before_m=45., after_m=140.):
    """Validate a predeclared spawn anchor without observing simulated futures."""
    road, lane, s = anchor
    waypoint = cmap.get_waypoint_xodr(int(road), int(lane), float(s))
    if waypoint is None:
        raise ValueError(f'Declared lane-change anchor is unavailable: {anchor}')
    before = extend(waypoint, before_m, True)
    after = extend(waypoint, after_m)
    if before is None or after is None:
        raise ValueError(f'Declared anchor has insufficient actual route: {anchor}')
    source_points = list(reversed(before))+[waypoint]+after
    route, targets = paired_adjacent_routes(source_points)
    spawn_index = len(before)
    return route, targets, spawn_index, {
        'map': 'Town05', 'origin': row(waypoint)[:3],
        'road_id': int(road), 'lane_id': int(lane),
        'lane_width': float(waypoint.lane_width), 'conflicts': [],
        'type': 'three or more same-direction Driving lanes with bilateral reachable adjacent lanes',
        'selection': {'predeclared': True, 'anchor': [int(road), int(lane), float(s)],
                      'outcome_based_selection': False,
                      'geometry_only': True, 'source_distance_before_spawn_m': float(before_m),
                      'source_distance_after_spawn_m': float(after_m),
                      'requires_both_adjacent_lanes_throughout_saved_route': True},
    }


def build_lane_change_scenes(opendrive_root, seed=20261014, count_per_family=5,
                             history_s=2.0, driving_warmup_s=1.0):
    """Ten real Town05 bilateral-lane-change cases for the first two families.

    The twenty-scene/four-family target is NOT fulfilled by this subset.
    Each config declares the missing intersection and merging families. Seven
    physical vehicles use fixed real lane routes; NPCs receive no ego branch
    identity. Slots/gaps/speeds are fixed before rollout, never reaction-filtered.
    """
    if not 1 <= count_per_family <= 5:
        raise ValueError('The lane-change preflight declares five anchors per supported family')
    if history_s < 0 or driving_warmup_s < 0:
        raise ValueError('History and driving warmup durations cannot be negative')
    source = Path(opendrive_root)/'Town05.xodr'
    cmap = carla.Map('Town05', source.read_text())
    map_hash = hashlib.sha256(source.read_bytes()).hexdigest()
    lane_samples = cmap.generate_waypoints(2.)
    scenes = []
    for family_index, family in enumerate(LANE_CHANGE_FAMILIES):
        for variant in range(count_per_family):
            anchor = LANE_CHANGE_ANCHORS[5*family_index+variant]
            route, targets, spawn_index, geometry = lane_change_anchor_geometry(cmap, anchor)
            own_s = float(cumulative(route)[spawn_index])
            left_route, right_route = (targets[side]['target_route'] for side in ('left', 'right'))
            left_s, right_s = (float(cumulative(r)[spawn_index]) for r in (left_route, right_route))
            speed = 6.0+.35*(variant % 4)
            # Two additional vehicles fill BOTH adjacent lanes. Their rear cars
            # may react when ego physically enters their lane, through the same
            # current-state IDM policy in every branch, never a branch trigger.
            front_gap = (24. if family == 'rear_following' else 10.)+2.*(variant % 2)
            rear_gap = 11.+float(variant % 3)
            vehicles = [
                vehicle('ego', route, own_s, speed, priority=0, color='195,35,30'),
                vehicle('lead', route, own_s+31.+2.*(variant % 3), speed+.4, priority=0, color='220,220,220'),
                vehicle('rear', route, own_s-14.-float(variant % 2), speed, speed+1.0, priority=1, color='35,80,190'),
                vehicle('left_front', left_route, left_s+front_gap, speed, speed+.5, priority=1, color='120,120,125'),
                vehicle('left_rear', left_route, left_s-rear_gap, speed, speed+1.0, priority=1, color='35,75,120'),
                vehicle('right_front', right_route, right_s+front_gap+2., speed, speed+.5, priority=1, color='155,155,150'),
                vehicle('right_rear', right_route, right_s-rear_gap-1., speed, speed+1.0, priority=1, color='85,40,40'),
            ]
            for a, b in itertools.combinations(vehicles, 2):
                if spawn_footprints_overlap(a, b):
                    raise ValueError(f'Lane-change spawn footprints overlap: {anchor}/{a["role"]}/{b["role"]}')
            config = finish_config(family, len(scenes), seed, vehicles, geometry, history_s, driving_warmup_s)
            config['scene_profile'] = 'lane_change_v1'
            config['version'] = 'paired-multicam-lane-change-v3'
            config['action'].pop('lateral_offset_m')
            config['action'].update(lateral_mode='adjacent_lane_route', lane_changes=targets,
                reference='measured ego XY speed at shared t0; left/right smoothly track actual same-direction adjacent lane centres',
                requested_lateral_semantics='left/right lane change or clear adjacent-lane cut-in; actual trajectory and bbox require independent acceptance')
            config['controller']['leader_geometry'] = 'route_projection'
            geometry['traffic_design'] = {
                'vehicle_count_including_ego': 7, 'additional_vehicles_vs_old_following_parallel': 2,
                'primary_reactive_candidates': ['rear', 'left_rear', 'right_rear'],
                'adjacent_front_route_gap_m': {'left': front_gap, 'right': front_gap+2.},
                'adjacent_rear_route_gap_m': {'left': rear_gap, 'right': rear_gap+1.},
                'gaps_defined_at': 'declared spawn before common warmup and history',
                'reaction_guaranteed': False, 'future_outcome_filtering': False,
            }
            # Reserve the entire warmup/history/future plus a 20 m route margin.
            # The accelerated branch has an upper bound using uncapped a*t^2/2;
            # this is deliberately conservative relative to the capped policy.
            required = {}
            total_s = float(history_s+driving_warmup_s+config['duration_s'])
            for v in vehicles:
                travel = max(v['initial_speed_mps'], v['desired_speed_mps'])*total_s
                if v['role'] == 'ego':
                    travel += .5*config['action']['acceleration_m_s2']*config['duration_s']**2
                available = float(cumulative(v['route'])[-1]-v['spawn_route_s'])
                if available < travel+20.:
                    raise ValueError(f'Insufficient route for complete accelerated horizon: {anchor}/{v["role"]}')
                required[v['role']] = {'available_distance_m': available, 'conservative_travel_m': travel,
                                       'required_extra_margin_m': 20.}
            geometry['route_horizon_checks'] = required
            config['provenance'].update(
                opendrive_sha256=map_hash,
                description='Town05 real same-direction adjacent-lane-change preflight; custom CARLA interaction cases, not official Bench2Drive replay.',
                selection='Predeclared road/lane/s anchors and geometry validity only. No observed future response or model score is used.',
                requested_benchmark_scene_count=20, requested_families=list(FAMILIES),
                implemented_families=list(LANE_CHANGE_FAMILIES),
                pending_families=['unprotected_intersection', 'merging'],
                scope_status='two_family_lane_change_preflight_not_complete_twenty_scene_benchmark',
                geometry_min_spawn_separation_m=float(min(np.linalg.norm(np.asarray(a['transform']['position'][:2])-b['transform']['position'][:2]) for a, b in itertools.combinations(vehicles, 2))),
                spawn_footprint_check={'shape': 'oriented rectangle', 'length_m': 5.6, 'width_m': 2.3,
                                       'all_pairs_disjoint': True, 'actual_blueprint_bbox_requires_post_collection_audit': True})
            nominal_t0 = point_at_s(route, own_s+speed*(history_s+driving_warmup_s))['position']
            config['local_map_lanes'] = local_lane_geometry(lane_samples, nominal_t0)
            config['local_map_geometry'] = {'center_xyz': nominal_t0, 'radius_m': 120., 'sample_distance_m': 2.,
                                          'point_order': 'increasing OpenDRIVE s within road/lane/section'}
            config['map_geometry_path'] = 'maps/Town05.xodr'
            scenes.append(config)
    return scenes


# Full twenty-configuration geometry plan. Simulation acceptance is separate.
# Five declared parameter variants distributed over TWO geographical junctions.
# All parameters are specified before simulation; selection sees only map data.
LANE_CHANGE_JUNCTION_VARIANTS = (
    {'junction': 2014, 'ego_speed': 5.2, 'npc_speed': 3.8, 'ego_arrival_s': 2.4, 'npc_arrival_s': 2.6},
    {'junction': 2328, 'ego_speed': 5.5, 'npc_speed': 4.0, 'ego_arrival_s': 2.6, 'npc_arrival_s': 2.8},
    {'junction': 2014, 'ego_speed': 5.8, 'npc_speed': 4.1, 'ego_arrival_s': 2.2, 'npc_arrival_s': 2.5},
    {'junction': 2328, 'ego_speed': 6.0, 'npc_speed': 4.2, 'ego_arrival_s': 2.8, 'npc_arrival_s': 2.7},
    {'junction': 2014, 'ego_speed': 5.6, 'npc_speed': 3.9, 'ego_arrival_s': 2.7, 'npc_arrival_s': 2.9},
)
LANE_CHANGE_CONNECTIONS = {
    2014: {'ego': ((36, 3), (35, 3)), 'right': ((36, 4), (35, 4)),
           'cross': ((35, -1), (13, -1)), 'entry': ((13, 2), (35, 4))},
    2328: {'ego': ((38, 3), (37, 3)), 'right': ((38, 4), (37, 4)),
           'cross': ((37, -1), (27, -1)), 'entry': ((27, 2), (37, 4))},
}


def lane_change_connector_points(junction, incoming, outgoing):
    """Resolve exact CARLA waypoint objects for one explicitly named connector."""
    matches = []
    for start, end in sorted(junction.get_waypoints(carla.LaneType.Driving), key=lambda p: wp_key(p[0])):
        topo = merge_path_topology(start, end)
        if topo is None:
            continue
        identity = ((topo['incoming_road_id'], topo['incoming_lane_id']),
                    (topo['outgoing_road_id'], topo['outgoing_lane_id']))
        if identity != (tuple(incoming), tuple(outgoing)):
            continue
        before, middle, after = extend(start, 70, True), trace_connector(start, end), extend(end, 100)
        if before is None or middle is None or after is None:
            raise ValueError(f'Actual connector route cannot cover required approach/exit: {identity}')
        points = list(reversed(before))+middle+after
        values = np.asarray([row(p) for p in points])
        keep = np.r_[True, np.linalg.norm(np.diff(values[:, :3], axis=0), axis=1) > .01]
        points = [p for p, keep_point in zip(points, keep) if keep_point]
        matches.append(points)
    if len(matches) != 1:
        raise ValueError(f'Actual connector is not unique: {incoming}->{outgoing}: {len(matches)}')
    return matches[0]


def paired_connector_targets(points):
    """Actual get_left/right results, including explicit controlled junctions.

    Neither nearest-lane projection nor synthetic extrapolation is used to fill
    a missing target. Junction markings can be NONE even when lane_change says
    Both; this records those facts without claiming public-road legality.
    """
    source = np.asarray([row(p) for p in points], float)
    segment = np.linalg.norm(np.diff(source[:, :3], axis=0), axis=1)
    if not np.isfinite(source).all() or np.any(segment <= 1e-5) or np.any(segment > 2.1):
        raise ValueError('Connector source route must be finite and continuous')
    if any(p.lane_type != carla.LaneType.Driving or p.lane_width <= 0 for p in points):
        raise ValueError('Connector source includes a non-driving or zero-width lane')
    normal = np.c_[-np.sin(np.radians(source[:, 3])), np.cos(np.radians(source[:, 3]))]
    forward = np.c_[np.cos(np.radians(source[:, 3])), np.sin(np.radians(source[:, 3]))]
    source_widths = np.array([p.lane_width for p in points], float)
    changes = {}
    for side, method, sign, perm in (('left', 'get_left_lane', -1, carla.LaneChange.Left),
                                     ('right', 'get_right_lane', 1, carla.LaneChange.Right)):
        targets = []
        for i, p in enumerate(points):
            q = getattr(p, method)()
            if q is None or q.lane_type != carla.LaneType.Driving or q.lane_width <= 0:
                raise ValueError(f'{side}: missing actual Driving neighbour at station {i}')
            if (p.road_id, p.section_id) != (q.road_id, q.section_id) or p.lane_id == q.lane_id or abs(p.s-q.s) > 1e-3:
                raise ValueError(f'{side}: actual source/target station correspondence failed')
            if angle(p.transform.rotation.yaw, q.transform.rotation.yaw) > 5:
                raise ValueError(f'{side}: neighbour has opposite or divergent traffic direction')
            if not (int(p.lane_change) & int(perm)):
                raise ValueError(f'{side}: map lane_change metadata does not permit the move')
            targets.append(q)
        target = np.asarray([row(q) for q in targets], float)
        width = np.array([q.lane_width for q in targets], float)
        delta = target[:, :2]-source[:, :2]
        signed = np.einsum('ij,ij->i', delta, normal)*sign
        longitudinal = np.einsum('ij,ij->i', delta, forward)
        spacing = (source_widths+width)/2
        if np.max(np.abs(signed-spacing)) > .15 or np.max(np.abs(longitudinal)) > .15:
            raise ValueError(f'{side}: actual neighbour centres do not form matched adjacent stations')
        lengths = np.linalg.norm(np.diff(target[:, :3], axis=0), axis=1)
        if np.any(lengths <= 1e-5) or np.any(lengths > 2.1):
            raise ValueError(f'{side}: actual neighbour route is discontinuous')
        changes[side] = {'available': True, 'target_route': target.tolist(), 'width_m': float(width.min()),
            'evidence': {
                'method': f'original CARLA waypoint.{method} at every original source station',
                'source_lane_keys': [[int(p.road_id), int(p.section_id), int(p.lane_id)] for p in points],
                'target_lane_keys': [[int(q.road_id), int(q.section_id), int(q.lane_id)] for q in targets],
                'source_opendrive_s_m': [float(p.s) for p in points],
                'target_opendrive_s_m': [float(q.s) for q in targets],
                'source_lane_widths_m': source_widths.tolist(), 'target_lane_widths_m': width.tolist(),
                'same_direction': True, 'target_lane_type': 'Driving', 'station_count': len(points),
                'source_is_junction': [bool(p.is_junction) for p in points],
                'source_lane_change_permissions': [str(p.lane_change) for p in points],
                'source_boundary_marking_type': [str(getattr(p, side+'_lane_marking').type) for p in points],
                'map_lane_change_metadata_permits_all_stations': True,
                'public_road_legal_permission_verified': False,
                'protocol': 'controlled unprotected junction; lane-change legality and painted junction lane boundaries are not established',
                'min_signed_lateral_centre_separation_m': float(signed.min()),
            }}
    return source.tolist(), changes


def check_lane_change_future_budget(config):
    action = config['action']
    lateral_checks = {}
    for name, profile in action.get('profiles', {}).items():
        if profile.get('kind') != 'lane_change':
            continue
        direction = profile['direction']
        onset = float(profile.get('onset_s', action.get('lateral_onset_s', .2)))
        duration = float(profile.get('duration_s', action.get('lateral_duration_s', 2.5)))
        hold = .5
        if onset < 0 or duration <= 0 or onset + duration + hold > config['duration_s'] + 1e-9:
            raise ValueError(f'Insufficient lane-change and hold time: {config["scene_id"]}/{name}')
        if not action.get('lane_changes', {}).get(direction, {}).get('available'):
            raise ValueError(f'Lane-change target unavailable: {config["scene_id"]}/{name}')
        lateral_checks[name] = {'direction': direction, 'onset_s': onset, 'duration_s': duration,
                                'required_hold_s': hold, 'available_future_s': config['duration_s'],
                                'note': 'Static budget only; actual completion requires trajectory and bounding-box audit.'}
    config['geometry']['lane_change_time_checks'] = lateral_checks
    total_s = config['history_s']+config['driving_warmup_s']+config['duration_s']
    checks = {}
    for v in config['vehicles']:
        travel = max(v['initial_speed_mps'], v['desired_speed_mps'])*total_s
        if v['role'] == 'ego':
            travel += .5*config['action']['acceleration_m_s2']*config['duration_s']**2
        available = float(cumulative(v['route'])[-1]-v['spawn_route_s'])
        if available < travel+20:
            raise ValueError(f'Incomplete future route budget: {config["scene_id"]}/{v["role"]}')
        checks[v['role']] = {'available_distance_m': available, 'conservative_travel_m': travel,
                             'required_extra_margin_m': 20.}
    config['geometry']['route_horizon_checks'] = checks


def _convex_polygons_overlap(a, b):
    """Separating-axis test including boundary contact, for map/footprint quads."""
    for polygon in (a, b):
        edges = np.roll(polygon, -1, axis=0)-polygon
        axes = np.c_[-edges[:, 1], edges[:, 0]]
        for axis in axes:
            if np.linalg.norm(axis) < 1e-9:
                continue
            pa, pb = a@axis, b@axis
            if pa.max() < pb.min()-1e-9 or pb.max() < pa.min()-1e-9:
                return False
    return True


def crossing_priority_zone(main_route, lane_changes, crossing_route, priority_roles,
                           yield_role='cross_traffic', sample_step_m=.1, mode='crossing'):
    """Map-only conflict corridor for a yielding vehicle crossing all three lanes.

    yield_route_s bounds the ACTOR CENTRE while its conservative oriented body
    touches any actual main-road lane. The controller must not subtract another
    half vehicle length from this entry. main_zone_s contains the longitudinal
    footprint sweep; priority actors are tested using their own live boxes.
    No ego action, branch, observed response, or future trajectory is consulted.
    """
    main, crossing = np.asarray(main_route, float), np.asarray(crossing_route, float)
    if not 0 < sample_step_m <= .25:
        raise ValueError('Priority-zone sampling step must be in (0, .25] metres')
    if mode not in ('crossing', 'merging'):
        raise ValueError('Priority zone must be a crossing or right-lane merge')
    if main.ndim != 2 or crossing.ndim != 2 or main.shape[1] != 4 or crossing.shape[1] != 4:
        raise ValueError('Priority-zone routes must be Nx4 map centreline poses')
    if not np.isfinite(main).all() or not np.isfinite(crossing).all():
        raise ValueError('Priority-zone routes must be finite')
    roles = list(priority_roles)
    if not roles or len(set(roles)) != len(roles) or yield_role in roles:
        raise ValueError('Priority-zone roles must be unique and exclude yielding role')
    distances = cumulative(crossing)
    lengths = np.diff(distances)
    if np.any(lengths <= 1e-6):
        raise ValueError('Priority-zone route has degenerate segments')
    # CARLA 0.9.16 native_v2 measurements include local bbox offsets. The largest
    # longitudinal bound is Charger 2.503913+0.005352 m; lateral is Tesla
    # 1.081725+0.000010 m. This fixed envelope covers all four supported models.
    half_length, half_width = 2.8, 1.2
    yaw = np.unwrap(np.radians(crossing[:, 3]))
    radius = math.hypot(half_length, half_width)
    max_yaw_per_m = float(np.max(abs(np.diff(yaw))/lengths))
    # Every continuous pose is within half a sample interval in translation and
    # rotation of a sampled pose. Inflate both rectangle axes by that bound.
    sampling_guard = sample_step_m/2*(1.+radius*max_yaw_per_m)
    extent = np.array([half_length+sampling_guard, half_width+sampling_guard])
    widths = np.asarray(lane_changes['left']['evidence']['source_lane_widths_m'], float)
    lanes = [('main', main, widths)]
    for side in ('left', 'right'):
        spec = lane_changes[side]
        route = np.asarray(spec['target_route'], float)
        lane_width = np.asarray(spec['evidence']['target_lane_widths_m'], float)
        if spec.get('available') is not True or route.shape != main.shape:
            raise ValueError('Priority zone requires both actual matched neighbour routes')
        lanes.append((side, route, lane_width))
    if mode == 'merging':
        lanes = [lanes[2]]
        main = lanes[0][1]
        lateral_bounds = [-float(lanes[0][2].max())/2, float(lanes[0][2].max())/2]
    else:
        normal = np.c_[-np.sin(np.radians(main[:, 3])), np.cos(np.radians(main[:, 3]))]
        outer = []
        for name, route, width in lanes:
            offset = np.einsum('ij,ij->i', route[:, :2]-main[:, :2], normal)
            outer.extend([float((offset-width/2).min()), float((offset+width/2).max())])
        lateral_bounds = [min(outer), max(outer)]
    polygons, heights, lane_names = [], [], []
    for name, route, width in lanes:
        if width.shape != (len(route),) or not np.isfinite(width).all() or np.any(width <= 0):
            raise ValueError('Priority zone requires positive measured lane widths')
        normal = np.c_[-np.sin(np.radians(route[:, 3])), np.cos(np.radians(route[:, 3]))]
        lo, hi = route[:, :2]-normal*width[:, None]/2, route[:, :2]+normal*width[:, None]/2
        for i in range(len(route)-1):
            polygons.append(np.array([lo[i], lo[i+1], hi[i+1], hi[i]]))
            heights.append((min(route[i, 2], route[i+1, 2]), max(route[i, 2], route[i+1, 2])))
            lane_names.append(name)
    polygons, heights = np.asarray(polygons), np.asarray(heights)
    lower, upper = polygons.min(axis=1), polygons.max(axis=1)
    stations = np.linspace(0., distances[-1], int(math.ceil(distances[-1]/sample_step_m))+1)
    centers = np.column_stack([np.interp(stations, distances, crossing[:, k]) for k in range(3)])
    angles = np.interp(stations, distances, yaw)
    signs = np.array([[-1., -1.], [1., -1.], [1., 1.], [-1., 1.]])
    hits, swept_boxes, hit_lanes, contained_stations = [], [], set(), []
    for station, center, angle_r in zip(stations, centers, angles):
        forward = np.array([math.cos(angle_r), math.sin(angle_r)])
        right = np.array([-forward[1], forward[0]])
        box = center[:2]+signs[:, :1]*extent[0]*forward+signs[:, 1:]*extent[1]*right
        candidates = np.flatnonzero(((upper >= box.min(axis=0)-1e-8) & (lower <= box.max(axis=0)+1e-8)).all(axis=1)
                                   & (heights[:, 0] <= center[2]+.5) & (heights[:, 1] >= center[2]-.5))
        touched = [j for j in candidates if _convex_polygons_overlap(box, polygons[j])]
        if touched:
            hits.append(float(station))
            swept_boxes.append(box)
            hit_lanes.update(lane_names[j] for j in touched)
            if mode == 'merging':
                # A merging path stays on the main road. Its conflict interval
                # ends only after a full footprint is aligned inside that lane
                # for at least one conservative vehicle length of shared path.
                inside = []
                for corner in box:
                    progress = city_project_s(main, corner)
                    sample = city_route_point(main, progress)
                    inside.append(np.linalg.norm(corner-sample[:2]) <= min(lanes[0][2])/2)
                main_s = city_project_s(main, center[:2])
                main_yaw = float(np.interp(main_s, cumulative(main), np.unwrap(np.radians(main[:, 3]))))
                aligned = abs((angle_r-main_yaw+math.pi) % (2*math.pi)-math.pi) <= math.radians(5.)
                contained_stations.append(bool(all(inside) and aligned))
    expected_lanes = {'right'} if mode == 'merging' else {'main', 'left', 'right'}
    if hit_lanes != expected_lanes:
        raise ValueError('Crossing footprint must traverse all declared actual main-road corridors')
    entry = max(0., min(hits)-sample_step_m/2)
    if mode == 'merging':
        run_start, exit_s = None, None
        for station, contained in zip(hits, contained_stations):
            run_start = (station if run_start is None else run_start) if contained else None
            if run_start is not None and station-run_start >= 2*half_length:
                exit_s = station+sample_step_m/2
                break
        if exit_s is None:
            raise ValueError('Merge needs sustained full-body alignment inside its actual right lane')
        swept_boxes = [box for station, box in zip(hits, swept_boxes) if station <= exit_s]
    else:
        exit_s = min(float(distances[-1]), max(hits)+sample_step_m/2)
    if entry <= sample_step_m or exit_s >= distances[-1]-sample_step_m:
        raise ValueError('Crossing route must approach and leave the complete priority corridor')
    # Enclose the swept boxes in a world-axis rectangle before projection. Its
    # full edges, not just corners, are projected so a gently bending map route
    # cannot hide an interior longitudinal extreme. This is conservative.
    corners = np.asarray(swept_boxes).reshape(-1, 2)
    low, high = corners.min(axis=0), corners.max(axis=0)
    rectangle = np.array([[low[0], low[1]], [high[0], low[1]], high, [low[0], high[1]]])
    projected = []
    for a, b in zip(rectangle, np.roll(rectangle, -1, axis=0)):
        for fraction in np.linspace(0., 1., int(math.ceil(np.linalg.norm(b-a)/sample_step_m))+1):
            projected.append(city_project_s(main, a+fraction*(b-a)))
    main_guard = sample_step_m
    main_interval = [max(0., min(projected)-main_guard),
                     min(float(cumulative(main)[-1]), max(projected)+main_guard)]
    return {'yield_role': yield_role, 'priority_roles': roles, 'main_route': main.tolist(),
            'main_zone_s': main_interval, 'yield_route_s': [entry, exit_s],
            'main_lateral_bounds_m': lateral_bounds,
            'stop_margin_m': 3., 'clearance_time_s': 4., 'lateral_lookahead_s': 1.,
            'evidence': {
                'method': 'oriented maximum-native-vehicle footprint sweep against full map-derived lane ribbons',
                'mode': mode, 'lane_corridors': sorted(expected_lanes),
                'mainroad_priority': 'all six declared main-road actors, subject to live footprint/corridor overlap',
                'body_half_length_m': half_length, 'body_half_width_m': half_width,
                'supported_blueprints': ['vehicle.tesla.model3', 'vehicle.dodge.charger_2020',
                                         'vehicle.lincoln.mkz_2020', 'vehicle.mercedes.coupe_2020'],
                'measured_bbox_with_offset_upper_bound_m': [2.509265, 1.081736],
                'footprint_source': 'CARLA 0.9.16 static bbox extent plus absolute local offset, native_v2 measurements; independent of future outcomes',
                'sample_step_m': sample_step_m, 'continuous_pose_inflation_m': sampling_guard,
                'yield_interval_semantics': 'actor-centre entry/exit with yielding body already included; for merge exit requires full-body alignment over one vehicle length of shared lane; stop_margin is additional, do not subtract half-length again',
                'main_interval_semantics': 'longitudinal envelope of crossing body sweep; do not add priority-body half-length here because controller projects live priority boxes',
                'main_projection_padding_m': main_guard, 'height_tolerance_m': .5,
                'admission_window': '4 seconds is a minimum heuristic constant-speed clearance window; live control may enlarge it using remaining route distance and observed speed, not a safety guarantee',
                'scope': 'fixed map routes, lane widths and body envelope; actual off-route tracking, contacts and dynamics still require capture audit',
                'uses_ego_branch_or_future': False}}


def build_extended_lane_change_scenes(opendrive_root, seed=20261014, count_per_family=5,
                                      history_s=2., driving_warmup_s=1.):
    if __package__:
        from .controllers_v2 import validate_lane_change_routes
    else:
        from controllers_v2 import validate_lane_change_routes

    if not 1 <= count_per_family <= 5:
        raise ValueError('Only five predeclared variants per family exist')
    source = Path(opendrive_root)/'Town05.xodr'
    cmap = carla.Map('Town05', source.read_text())
    lane_samples = cmap.generate_waypoints(2.)
    junctions = {p.junction_id: p.get_junction() for p in cmap.generate_waypoints(4) if p.is_junction}
    scenes = build_lane_change_scenes(opendrive_root, seed, count_per_family, history_s, driving_warmup_s)
    pre = history_s+driving_warmup_s
    for family in ('unprotected_intersection', 'merging'):
        for variant_index, variant in enumerate(LANE_CHANGE_JUNCTION_VARIANTS[:count_per_family]):
            jid = variant['junction']
            connection = LANE_CHANGE_CONNECTIONS[jid]
            junction = junctions[jid]
            ego = city_connector(junction, *connection['ego'])
            route, changes = paired_connector_targets(lane_change_connector_points(junction, *connection['ego']))
            np.testing.assert_allclose(route, ego['route'], atol=1e-6)
            merge = family == 'merging'
            npc = city_connector(junction, *connection['entry' if merge else 'cross'])
            if merge:
                right_main = city_connector(junction, *connection['right'])
                point, right_join_s, npc_s, evidence = city_merge_geometry(right_main, npc)
                # Project only the physical join point to define a longitudinal
                # arrival station on centre-lane ego's route. This is NOT a
                # claim that ego's reference lane shares the merge centreline.
                ego_s = city_project_s(route, point)
                evidence.update(type='right_adjacent_lane_entry_merge',
                    merges_into_ego_reference_lane=False,
                    merges_into_ego_right_lane=True,
                    reference_ego_outgoing_lane=list(connection['ego'][1]),
                    merged_outgoing_lane=list(connection['right'][1]),
                    public_road_legal_permission_verified=False,
                    npc_must_yield_to=['right_front', 'right_rear'],
                    ego_right_interaction='ego may physically enter this shared right lane; NPC observes current vehicles only')
                npc_role = 'merging_traffic'
            else:
                point, ego_s, npc_s, evidence = city_crossing_geometry(ego, npc)
                npc_role = 'cross_traffic'
            speed, npc_speed = variant['ego_speed'], variant['npc_speed']
            own_spawn = ego_s-speed*(pre+variant['ego_arrival_s'])
            npc_spawn = npc_s-npc_speed*(pre+variant['npc_arrival_s'])
            # Actual target-route distance of the same predeclared source
            # spawn station; neighbour routes may have different arc lengths.
            own_lengths = cumulative(route)
            left_s = float(np.interp(own_spawn, own_lengths, cumulative(changes['left']['target_route'])))
            right_s = float(np.interp(own_spawn, own_lengths, cumulative(changes['right']['target_route'])))
            left, right = changes['left']['target_route'], changes['right']['target_route']
            cars = [
                vehicle('ego', route, own_spawn, speed, priority=0, color='195,35,30'),
                vehicle('rear', route, own_spawn-14, speed, speed+.8, priority=1, color='35,80,190'),
                vehicle(npc_role, npc['route'], npc_spawn, npc_speed, priority=2, color='140,140,145'),
                vehicle('left_front', left, left_s+22, speed, speed+.4, priority=1, color='190,190,185'),
                vehicle('left_rear', left, left_s-12, speed, speed+.8, priority=1, color='35,75,120'),
                vehicle('right_front', right, right_s+24, speed, speed+.4, priority=1, color='155,155,150'),
                vehicle('right_rear', right, right_s-13, speed, speed+.8, priority=1, color='80,40,40'),
            ]
            for a, b in itertools.combinations(cars, 2):
                if spawn_footprints_overlap(a, b):
                    raise ValueError(f'Initial physical footprints overlap: {family}/{variant_index}/{a["role"]}/{b["role"]}')
            if merge:
                right_conflict_s = city_project_s(right, point)
                # Priority uses cars actually travelling in the merge lane.
                # Ego's different reference lane is not a fictitious conflict.
                conflicts = [{'roles': [priority, role], 'point': point,
                              'route_s': {priority: right_conflict_s, role: npc_s},
                              'priority_role': priority, 'kind': 'right_adjacent_lane_entry_merge'}
                             for role in (npc_role,)
                             for priority in ('right_front', 'right_rear')]
            else:
                conflicts = [{'roles': ['ego', role], 'point': point,
                              'route_s': {'ego': ego_s, role: npc_s},
                              'priority_role': 'ego', 'kind': family}
                             for role in (npc_role,)]
            geometry = {'map': 'Town05', 'junction_id': int(jid), 'geometry': family,
                'type': evidence['type'], 'conflict_xyz': point,
                'ego_conflict_s': float(ego_s), 'npc_conflict_s': float(npc_s),
                'ego_connector_road_id': ego['topology']['connector_road_id'],
                'npc_connector_road_id': npc['topology']['connector_road_id'],
                'main_road_topology': ego['topology'], 'entry_road_topology': npc['topology'],
                'ego_turn_signed_deg': ego['signed_turn_deg'], 'npc_turn_signed_deg': npc['signed_turn_deg'],
                'conflicts': conflicts,
                'nominal_conflict_after_t0_s': {'ego_longitudinal_station': variant['ego_arrival_s'], npc_role: variant['npc_arrival_s']},
                'selection': {'predeclared': True, 'junction_id': int(jid), 'parameter_variant': variant_index,
                    'declared_variant': dict(variant), 'geographic_junction_count_in_family': 2,
                    'ego_incoming_lane': list(connection['ego'][0]), 'ego_outgoing_lane': list(connection['ego'][1]),
                    'npc_incoming_lane': list(connection['entry' if merge else 'cross'][0]),
                    'npc_outgoing_lane': list(connection['entry' if merge else 'cross'][1]),
                    'outcome_based_selection': False, 'public_road_legal_permission_verified': False},
                'traffic_design': {'vehicle_count_including_ego': 7, 'previous_junction_vehicle_count': 4,
                    'additional_vehicles_vs_old_junction': 3,
                    'reaction_guaranteed': False, 'future_outcome_filtering': False,
                    'entry_merges_into': 'ego right adjacent lane, not ego reference lane' if merge else None},
                'physical_join_anchor' if merge else 'crossing_evidence': evidence}
            if merge:
                geometry['right_main_road_topology'] = right_main['topology']
                geometry['outgoing_lane'] = list(connection['right'][1])
            geometry['priority_zone'] = crossing_priority_zone(
                route, changes, npc['route'], [v['role'] for v in cars if v['role'] != npc_role],
                npc_role, mode='merging' if merge else 'crossing')
            config = finish_config(family, len(scenes), seed, cars, geometry, history_s, driving_warmup_s)
            config['scene_profile'] = 'lane_change_v1'
            config['version'] = 'paired-multicam-lane-change-v3'
            config['action'].pop('lateral_offset_m')
            config['action'].update(lateral_mode='adjacent_lane_route', lane_changes=changes,
                reference='measured ego XY speed at shared t0; left/right track actual same-direction adjacent lane centres',
                requested_lateral_semantics='left/right lane change or clear cut-in; actual execution requires independent trajectory and bbox audit')
            config['controller']['leader_geometry'] = 'route_projection'
            config['provenance'].update(opendrive_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                description='Controlled Town05 junction geometry, not official Bench2Drive replay or a public-road legality guarantee.',
                selection='Predeclared two-junction parameter variants, no observed future or reaction filtering.',
                spawn_footprint_check={'shape': 'oriented rectangle', 'length_m': 5.6, 'width_m': 2.3,
                    'all_pairs_disjoint': True, 'actual_blueprint_bbox_requires_post_collection_audit': True})
            check_lane_change_future_budget(config)
            nominal_t0 = point_at_s(route, own_spawn+speed*pre)['position']
            config['local_map_lanes'] = local_lane_geometry(lane_samples, nominal_t0)
            config['local_map_geometry'] = {'center_xyz': nominal_t0, 'radius_m': 120., 'sample_distance_m': 2.,
                'point_order': 'increasing OpenDRIVE s within road/lane/section'}
            config['map_geometry_path'] = 'maps/Town05.xodr'
            validate_lane_change_routes(route, config['action'])
            scenes.append(config)
    for c in scenes:
        c['provenance'].update(requested_benchmark_scene_count=20,
            implemented_families=['rear_following', 'adjacent_parallel', 'unprotected_intersection', 'merging'],
            pending_families=[], scope_status='planned_geometry_not_collected_or_accepted',
            geography_note='10 road anchors plus two junction locations; each junction family uses five predeclared parameter variants across those two locations.')
    return scenes



# Diversity profile: old builders and their frozen outputs remain unchanged.
def diverse_adjacent_routes(points):
    """Return only whole-route, station-matched, permitted actual neighbours.

    Missing/restricted neighbours are recorded, never extrapolated. Checking
    the whole frozen route is conservative: a side may physically exist at
    spawn yet be unavailable for this intervention horizon.
    """
    source = np.asarray([row(p) for p in points], float)
    if source.ndim != 2 or source.shape[1] != 4 or len(source) < 2 or not np.isfinite(source).all():
        raise ValueError('Source requires finite Nx4 original map waypoints')
    lengths = np.linalg.norm(np.diff(source[:, :3], axis=0), axis=1)
    if np.any(lengths <= 1e-5) or np.any(lengths > 2.1):
        raise ValueError('Source has degenerate or discontinuous map segments')
    if any(p.lane_type != carla.LaneType.Driving or p.lane_width <= 0 for p in points):
        raise ValueError('Source includes a non-Driving or zero-width lane')
    widths = np.array([p.lane_width for p in points], float)
    normal = np.c_[-np.sin(np.radians(source[:, 3])), np.cos(np.radians(source[:, 3]))]
    forward = np.c_[np.cos(np.radians(source[:, 3])), np.sin(np.radians(source[:, 3]))]
    changes, unavailable = {}, {}
    for side, sign, permission in [('left', -1, carla.LaneChange.Left), ('right', 1, carla.LaneChange.Right)]:
        targets = []
        try:
            for i, p in enumerate(points):
                q = getattr(p, 'get_'+side+'_lane')()
                if q is None or q.lane_type != carla.LaneType.Driving or q.lane_width <= 0:
                    raise ValueError(f'no actual positive-width Driving neighbour at station {i}')
                if (q.road_id, q.section_id) != (p.road_id, p.section_id) or p.lane_id == q.lane_id or abs(p.s-q.s) > 1e-3:
                    raise ValueError(f'source/target station mismatch at station {i}')
                if angle(q.transform.rotation.yaw, p.transform.rotation.yaw) > 5:
                    raise ValueError(f'opposite or divergent direction at station {i}')
                if not (int(p.lane_change) & int(permission)):
                    raise ValueError(f'map lane_change metadata disallows {side} at station {i}')
                targets.append(q)
            target = np.asarray([row(q) for q in targets], float)
            target_widths = np.asarray([q.lane_width for q in targets], float)
            delta = target[:, :2]-source[:, :2]
            signed = np.einsum('ij,ij->i', delta, normal)*sign
            longitudinal = np.einsum('ij,ij->i', delta, forward)
            if np.max(np.abs(signed-(widths+target_widths)/2)) > .15 or np.max(np.abs(longitudinal)) > .15:
                raise ValueError('neighbour centres do not form matched adjacent stations')
            segments = np.linalg.norm(np.diff(target[:, :3], axis=0), axis=1)
            if not np.isfinite(target).all() or np.any(segments <= 1e-5) or np.any(segments > 2.1):
                raise ValueError('actual target route has missing or discontinuous segments')
        except ValueError as exc:
            unavailable[side] = str(exc)
            continue
        changes[side] = {'available': True, 'target_route': target.tolist(),
            'width_m': float(target_widths.min()), 'evidence': {
                'method': f'original CARLA waypoint.get_{side}_lane at every source station',
                'source_lane_keys': [[int(p.road_id), int(p.section_id), int(p.lane_id)] for p in points],
                'target_lane_keys': [[int(q.road_id), int(q.section_id), int(q.lane_id)] for q in targets],
                'source_opendrive_s_m': [float(p.s) for p in points],
                'target_opendrive_s_m': [float(q.s) for q in targets],
                'source_lane_widths_m': widths.tolist(), 'target_lane_widths_m': target_widths.tolist(),
                'same_direction': True, 'target_lane_type': 'Driving', 'station_count': len(points),
                'source_is_junction': [bool(p.is_junction) for p in points],
                'source_lane_change_permissions': [str(p.lane_change) for p in points],
                'source_boundary_marking_type': [str(getattr(p, side+'_lane_marking').type) for p in points],
                'map_lane_change_metadata_permits_all_stations': True,
                'public_road_legal_permission_verified': False,
                'protocol': 'controlled experiment; map permission is not a public-road legality determination',
                'min_signed_lateral_centre_separation_m': float(signed.min())}}
    return source.tolist(), changes, unavailable


def diverse_lane_ribbons(points):
    """Real contiguous lane ribbons, including locally available neighbours.

    A neighbour disappearing anywhere disables that intervention, but its
    existing physical lane must still be considered in a junction footprint
    corridor. Each ribbon is split at any missing/invalid neighbour station.
    """
    ribbons = [{'name': 'main', 'route': [row(p) for p in points],
                'widths_m': [float(p.lane_width) for p in points]}]
    for side in ('left', 'right'):
        chunk, serial = [], 0
        def flush():
            nonlocal chunk, serial
            if len(chunk) >= 2:
                ribbons.append({'name': f'{side}_{serial}', 'route': [row(p) for p in chunk],
                                'widths_m': [float(p.lane_width) for p in chunk]})
                serial += 1
            chunk = []
        for p in points:
            q = getattr(p, 'get_'+side+'_lane')()
            valid = (q is not None and q.lane_type == carla.LaneType.Driving and q.lane_width > 0
                     and angle(q.transform.rotation.yaw, p.transform.rotation.yaw) <= 5
                     and (q.road_id, q.section_id) == (p.road_id, p.section_id)
                     and abs(q.s-p.s) <= 1e-3)
            if not valid:
                flush()
                continue
            if chunk and not 1e-5 < np.linalg.norm(np.array(row(chunk[-1]))[:3]-np.array(row(q))[:3]) <= 2.1:
                flush()
            chunk.append(q)
        flush()
    return ribbons


def diverse_priority_zone(main_route, lane_corridors, crossing_route, priority_roles,
                          yield_role='cross_traffic', sample_step_m=.1, mode='crossing'):
    """Actual one-/two-/three-lane footprint corridor for the diverse profile.

    lane_corridors contains real map waypoint ribbons; no missing adjacent lane
    is synthesized. For merging, pass only the actual lane receiving the entry.
    The legacy three-lane helper above is deliberately unchanged.
    """
    main, crossing = np.asarray(main_route, float), np.asarray(crossing_route, float)
    if not 0 < sample_step_m <= .25:
        raise ValueError('Priority-zone sampling step must be in (0, .25] metres')
    if mode not in ('crossing', 'merging'):
        raise ValueError('Priority zone must be a crossing or right-lane merge')
    if main.ndim != 2 or crossing.ndim != 2 or main.shape[1] != 4 or crossing.shape[1] != 4:
        raise ValueError('Priority-zone routes must be Nx4 map centreline poses')
    if not np.isfinite(main).all() or not np.isfinite(crossing).all():
        raise ValueError('Priority-zone routes must be finite')
    roles = list(priority_roles)
    if not roles or len(set(roles)) != len(roles) or yield_role in roles:
        raise ValueError('Priority-zone roles must be unique and exclude yielding role')
    distances = cumulative(crossing)
    lengths = np.diff(distances)
    if np.any(lengths <= 1e-6):
        raise ValueError('Priority-zone route has degenerate segments')
    # CARLA 0.9.16 native_v2 measurements include local bbox offsets. The largest
    # longitudinal bound is Charger 2.503913+0.005352 m; lateral is Tesla
    # 1.081725+0.000010 m. This fixed envelope covers all four supported models.
    half_length, half_width = 2.8, 1.2
    yaw = np.unwrap(np.radians(crossing[:, 3]))
    radius = math.hypot(half_length, half_width)
    max_yaw_per_m = float(np.max(abs(np.diff(yaw))/lengths))
    # Every continuous pose is within half a sample interval in translation and
    # rotation of a sampled pose. Inflate both rectangle axes by that bound.
    sampling_guard = sample_step_m/2*(1.+radius*max_yaw_per_m)
    extent = np.array([half_length+sampling_guard, half_width+sampling_guard])
    lanes = [(v['name'], np.asarray(v['route'], float), np.asarray(v['widths_m'], float))
             for v in lane_corridors]
    if not lanes or len({v[0] for v in lanes}) != len(lanes):
        raise ValueError('Priority corridor requires uniquely named actual map lane ribbons')
    if mode == 'merging' and len(lanes) != 1:
        raise ValueError('Merge corridor must be the actual receiving lane only')
    outer = []
    for name, route, width in lanes:
        if route.ndim != 2 or route.shape[1] != 4 or len(route) < 2 or not np.isfinite(route).all():
            raise ValueError('Priority ribbon requires a finite continuous map route')
        if np.any(np.linalg.norm(np.diff(route[:, :2], axis=0), axis=1) > 2.1):
            raise ValueError('Priority ribbon cannot bridge a missing map segment')
        if width.shape != (len(route),) or not np.isfinite(width).all() or np.any(width <= 0):
            raise ValueError('Priority ribbon requires actual positive map widths')
        for position, w in zip(route, width):
            progress = city_project_s(main, position[:2])
            center = city_route_point(main, progress)
            yaw_here = np.interp(progress, cumulative(main), np.unwrap(np.radians(main[:, 3])))
            offset = float((position[:2]-center[:2]) @ np.array([-np.sin(yaw_here), np.cos(yaw_here)]))
            outer.extend([offset-w/2, offset+w/2])
    lateral_bounds = [float(min(outer)), float(max(outer))]
    polygons, heights, lane_names = [], [], []
    for name, route, width in lanes:
        if width.shape != (len(route),) or not np.isfinite(width).all() or np.any(width <= 0):
            raise ValueError('Priority zone requires positive measured lane widths')
        normal = np.c_[-np.sin(np.radians(route[:, 3])), np.cos(np.radians(route[:, 3]))]
        lo, hi = route[:, :2]-normal*width[:, None]/2, route[:, :2]+normal*width[:, None]/2
        for i in range(len(route)-1):
            polygons.append(np.array([lo[i], lo[i+1], hi[i+1], hi[i]]))
            heights.append((min(route[i, 2], route[i+1, 2]), max(route[i, 2], route[i+1, 2])))
            lane_names.append(name)
    polygons, heights = np.asarray(polygons), np.asarray(heights)
    lower, upper = polygons.min(axis=1), polygons.max(axis=1)
    stations = np.linspace(0., distances[-1], int(math.ceil(distances[-1]/sample_step_m))+1)
    centers = np.column_stack([np.interp(stations, distances, crossing[:, k]) for k in range(3)])
    angles = np.interp(stations, distances, yaw)
    signs = np.array([[-1., -1.], [1., -1.], [1., 1.], [-1., 1.]])
    hits, swept_boxes, hit_lanes, contained_stations = [], [], set(), []
    for station, center, angle_r in zip(stations, centers, angles):
        forward = np.array([math.cos(angle_r), math.sin(angle_r)])
        right = np.array([-forward[1], forward[0]])
        box = center[:2]+signs[:, :1]*extent[0]*forward+signs[:, 1:]*extent[1]*right
        candidates = np.flatnonzero(((upper >= box.min(axis=0)-1e-8) & (lower <= box.max(axis=0)+1e-8)).all(axis=1)
                                   & (heights[:, 0] <= center[2]+.5) & (heights[:, 1] >= center[2]-.5))
        touched = [j for j in candidates if _convex_polygons_overlap(box, polygons[j])]
        if touched:
            hits.append(float(station))
            swept_boxes.append(box)
            hit_lanes.update(lane_names[j] for j in touched)
            if mode == 'merging':
                # A merging path stays on the main road. Its conflict interval
                # ends only after a full footprint is aligned inside that lane
                # for at least one conservative vehicle length of shared path.
                inside = []
                for corner in box:
                    progress = city_project_s(main, corner)
                    sample = city_route_point(main, progress)
                    inside.append(np.linalg.norm(corner-sample[:2]) <= min(lanes[0][2])/2)
                main_s = city_project_s(main, center[:2])
                main_yaw = float(np.interp(main_s, cumulative(main), np.unwrap(np.radians(main[:, 3]))))
                aligned = abs((angle_r-main_yaw+math.pi) % (2*math.pi)-math.pi) <= math.radians(5.)
                contained_stations.append(bool(all(inside) and aligned))
    expected_lanes = {v[0] for v in lanes}
    if not hits or (mode == 'merging' and hit_lanes != expected_lanes):
        raise ValueError('Yielding footprint does not intersect the declared actual receiving corridor')
    entry = max(0., min(hits)-sample_step_m/2)
    if mode == 'merging':
        run_start, exit_s = None, None
        for station, contained in zip(hits, contained_stations):
            run_start = (station if run_start is None else run_start) if contained else None
            if run_start is not None and station-run_start >= 2*half_length:
                exit_s = station+sample_step_m/2
                break
        if exit_s is None:
            raise ValueError('Merge needs sustained full-body alignment inside its actual receiving lane')
        swept_boxes = [box for station, box in zip(hits, swept_boxes) if station <= exit_s]
    else:
        exit_s = min(float(distances[-1]), max(hits)+sample_step_m/2)
    if entry <= sample_step_m or exit_s >= distances[-1]-sample_step_m:
        raise ValueError('Crossing route must approach and leave the complete priority corridor')
    # Enclose the swept boxes in a world-axis rectangle before projection. Its
    # full edges, not just corners, are projected so a gently bending map route
    # cannot hide an interior longitudinal extreme. This is conservative.
    corners = np.asarray(swept_boxes).reshape(-1, 2)
    low, high = corners.min(axis=0), corners.max(axis=0)
    rectangle = np.array([[low[0], low[1]], [high[0], low[1]], high, [low[0], high[1]]])
    projected = []
    for a, b in zip(rectangle, np.roll(rectangle, -1, axis=0)):
        for fraction in np.linspace(0., 1., int(math.ceil(np.linalg.norm(b-a)/sample_step_m))+1):
            projected.append(city_project_s(main, a+fraction*(b-a)))
    main_guard = sample_step_m
    main_interval = [max(0., min(projected)-main_guard),
                     min(float(cumulative(main)[-1]), max(projected)+main_guard)]
    return {'yield_role': yield_role, 'priority_roles': roles, 'main_route': main.tolist(),
            'main_zone_s': main_interval, 'yield_route_s': [entry, exit_s],
            'main_lateral_bounds_m': lateral_bounds,
            'stop_margin_m': 3., 'clearance_time_s': 4., 'lateral_lookahead_s': 1.,
            'evidence': {
                'method': 'oriented maximum-native-vehicle footprint sweep against full map-derived lane ribbons',
                'mode': mode, 'lane_corridors': sorted(expected_lanes), 'intersected_lane_corridors': sorted(hit_lanes),
                'mainroad_priority': 'all declared main-road actors, subject to live footprint/corridor overlap',
                'body_half_length_m': half_length, 'body_half_width_m': half_width,
                'supported_blueprints': ['vehicle.tesla.model3', 'vehicle.dodge.charger_2020',
                                         'vehicle.lincoln.mkz_2020', 'vehicle.mercedes.coupe_2020'],
                'measured_bbox_with_offset_upper_bound_m': [2.509265, 1.081736],
                'footprint_source': 'CARLA 0.9.16 static bbox extent plus absolute local offset, native_v2 measurements; independent of future outcomes',
                'sample_step_m': sample_step_m, 'continuous_pose_inflation_m': sampling_guard,
                'yield_interval_semantics': 'actor-centre entry/exit with yielding body already included; for merge exit requires full-body alignment over one vehicle length of shared lane; stop_margin is additional, do not subtract half-length again',
                'main_interval_semantics': 'longitudinal envelope of crossing body sweep; do not add priority-body half-length here because controller projects live priority boxes',
                'main_projection_padding_m': main_guard, 'height_tolerance_m': .5,
                'admission_window': '4 seconds is a minimum heuristic constant-speed clearance window; live control may enlarge it using remaining route distance and observed speed, not a safety guarantee',
                'scope': 'fixed map routes, lane widths and body envelope; actual off-route tracking, contacts and dynamics still require capture audit',
                'uses_ego_branch_or_future': False}}


DIVERSE_NPC_MODELS = ('vehicle.tesla.model3', 'vehicle.dodge.charger_2020',
                      'vehicle.lincoln.mkz_2020', 'vehicle.mercedes.coupe_2020')
DIVERSE_COLORS = ('215,215,210', '35,55,75', '145,45,40', '80,90,85',
                  '170,150,105', '30,90,105', '100,100,110', '220,220,220')


def diverse_road_candidates(cmap, map_name):
    """Map-only pool, long enough for shared history and accelerated future."""
    pool, rejected = [], []
    for w in sorted(cmap.generate_waypoints(30.), key=wp_key):
        if w.is_junction or w.lane_type != carla.LaneType.Driving:
            continue
        identity = {'road_id': int(w.road_id), 'section_id': int(w.section_id),
                    'lane_id': int(w.lane_id), 'opendrive_s_m': float(w.s)}
        try:
            before, after = extend(w, 65., True), extend(w, 140.)
            if before is None or after is None:
                raise ValueError('incomplete 65m history / 140m forward map route')
            points = list(reversed(before))+[w]+after
            if max(angle(p.transform.rotation.yaw, w.transform.rotation.yaw) for p in points[65:]) > 25:
                raise ValueError('road family forward route bends by more than 25 degrees')
            route, changes, unavailable = diverse_adjacent_routes(points)
            pool.append({'map': map_name, 'points': points, 'route': route, 'changes': changes,
                         'unavailable': unavailable, 'spawn_s': float(cumulative(route)[65]),
                         'anchor': identity, 'origin': row(w)[:3]})
        except ValueError as exc:
            rejected.append({'map': map_name, 'anchor': identity, 'reason': str(exc)})
    return pool, rejected


def _diverse_junction_candidates(maps):
    """Geographically varied explicit connections, plus geometric fallbacks.

    Town10HD provides five opposing-left-turn crossings and three distinct
    same-lane entry merges. Two further city merges and two Town05 entries
    joining the right lane provide additional real geometric alternatives;
    their crossing connectors are backups when a city connection cannot keep
    a real adjacent lane through the required route horizon. Rejections are
    recorded; actual unique-site counts are computed from the selected plan.
    """
    candidates = {name: [] for name in ('unprotected_intersection', 'merging')}
    declarations = [('Town10HD', 'unprotected_intersection', item) for item in CITY_CROSSING_ANCHORS]
    declarations += [('Town10HD', 'merging', item) for item in CITY_MERGE_ANCHORS]
    for jid, connection in sorted(LANE_CHANGE_CONNECTIONS.items()):
        declarations.append(('Town05', 'merging', (jid, *connection['ego'], *connection['entry'])))
        declarations.append(('Town05', 'unprotected_intersection', (jid, *connection['ego'], *connection['cross'])))
    junction_maps = {name: {w.junction_id: w.get_junction() for w in cmap.generate_waypoints(4) if w.is_junction}
                     for name, cmap in maps.items()}
    for name, family, declared in declarations:
        jid, ego_in, ego_out, npc_in, npc_out = declared
        junction = junction_maps[name][jid]
        points = lane_change_connector_points(junction, ego_in, ego_out)
        route, changes, unavailable = diverse_adjacent_routes(points)
        ego, npc = city_connector(junction, ego_in, ego_out), city_connector(junction, npc_in, npc_out)
        merging = family == 'merging'
        if merging and name == 'Town05':
            receiving_connection = LANE_CHANGE_CONNECTIONS[jid]['right']
            receiving = city_connector(junction, *receiving_connection)
            receiving_points = lane_change_connector_points(junction, *receiving_connection)
            point, _, npc_s, evidence = city_merge_geometry(receiving, npc)
            ego_s = city_project_s(route, point)
            evidence.update(type='right_adjacent_lane_entry_merge', merges_into_ego_reference_lane=False,
                            merges_into_ego_right_lane=True, merged_outgoing_lane=list(receiving_connection[1]))
        else:
            point, ego_s, npc_s, evidence = (city_merge_geometry(ego, npc) if merging else city_crossing_geometry(ego, npc))
            receiving_points = points
            if merging:
                evidence.update(merges_into_ego_reference_lane=True, merges_into_ego_right_lane=False,
                                merged_outgoing_lane=list(ego_out))
        main_route = [row(p) for p in receiving_points] if merging else route
        # A 200m route can meet the same urban street again at ANOTHER junction.
        # The fixed right-of-way zone belongs to this declared junction only.
        # Use real nearby map stations, still leaving 25m before / 35m after
        # its exact geometric crossing/join, not an artificial lane extension.
        zone_center = city_project_s(main_route, point)
        zone_distances = cumulative(main_route)
        lo = max(0, int(np.searchsorted(zone_distances, zone_center-25.))-1)
        hi = min(len(main_route), int(np.searchsorted(zone_distances, zone_center+35.))+1)
        local_points = receiving_points[lo:hi]
        ribbons = ([{'name': 'receiving', 'route': [row(p) for p in local_points],
                     'widths_m': [float(p.lane_width) for p in local_points]}] if merging
                   else diverse_lane_ribbons(local_points))
        role = 'merging_traffic' if merging else 'cross_traffic'
        zone = diverse_priority_zone(main_route, ribbons, npc['route'], ['ego'], role,
                                     mode='merging' if merging else 'crossing')
        zone['evidence']['map_corridor_scope'] = {
            'declared_junction_id': int(jid), 'main_conflict_station_m': float(zone_center),
            'actual_ribbon_source_station_interval_m': [float(zone_distances[lo]), float(zone_distances[hi-1])],
            'reason': 'local right-of-way area for this real junction; distant subsequent junctions are separate interactions'}
        candidates[family].append({'map': name, 'points': points, 'route': route, 'changes': changes,
            'unavailable': unavailable, 'npc': npc, 'npc_role': role, 'priority_zone': zone,
            'ego_conflict_s': float(ego_s), 'npc_conflict_s': float(npc_s), 'origin': point,
            'evidence': evidence, 'anchor': {'junction_id': int(jid), 'ego_incoming_lane': list(ego_in),
                'ego_outgoing_lane': list(ego_out), 'npc_incoming_lane': list(npc_in), 'npc_outgoing_lane': list(npc_out)},
            'main_road_topology': ego['topology'], 'entry_road_topology': npc['topology']})
    return candidates


def _diverse_scoped_routes(candidate, spawn_s, speed, pre_t0):
    """Crop only ego's irrelevant padding while keeping actual map samples.

    NPC routes remain complete independent map routes. No absent neighbour is
    filled to match the ego route. The intervention-window cross-check rejects
    any remaining side falsely lost only because of conservative padding.
    """
    points = candidate['points']
    original = np.asarray(candidate['route'], float)
    along = cumulative(original)
    # Max acceleration is the explicit strong longitudinal action, even though
    # lateral actions themselves maintain speed. Reserve the existing 20m
    # route budget and several complete original map samples at both ends.
    required_end = spawn_s+speed*(pre_t0+4.)+.5*1.5*4.**2+20.
    if required_end+2. > along[-1]:
        raise ValueError('full original map route cannot cover required ego horizon')
    first = max(0, int(np.searchsorted(along, spawn_s-10.))-1)
    last = min(len(points), int(np.searchsorted(along, required_end+2.))+1)
    ego_points = points[first:last]
    ego_route, changes, unavailable = diverse_adjacent_routes(ego_points)
    intervention_start = max(0., spawn_s+speed*pre_t0-3.)
    intervention_end = spawn_s+speed*(pre_t0+4.)+12.
    active_first = max(0, int(np.searchsorted(along, intervention_start))-1)
    active_last = min(len(points), int(np.searchsorted(along, intervention_end))+1)
    _, active_changes, active_unavailable = diverse_adjacent_routes(points[active_first:active_last])
    if set(changes) != set(active_changes):
        raise ValueError('conservative route padding changes actual intervention-window lane availability; reject geometry, do not downgrade actions')
    required_directions = candidate.get('required_directions')
    if required_directions is not None and set(changes) != set(required_directions):
        raise ValueError('actual intervention window does not match the predeclared availability stratum')
    local_spawn = float(spawn_s-along[first])
    lanes = {'main': (candidate['route'], spawn_s)}
    for side, spec in changes.items():
        # Reconstruct supporting traffic routes from original neighbour
        # Waypoints, not from extrapolating the cropped target tensor.
        matched = [getattr(p, 'get_'+side+'_lane')() for p in ego_points]
        before, after = extend(matched[0], 65., True), extend(matched[-1], 70.)
        if before is None or after is None:
            raise ValueError(f'{side}: incomplete independent NPC map route')
        traffic_points = list(reversed(before))+matched+after
        traffic_route = np.asarray([row(p) for p in traffic_points], float)
        lengths = np.linalg.norm(np.diff(traffic_route[:, :3], axis=0), axis=1)
        if np.any(lengths <= 1e-5) or np.any(lengths > 2.1):
            raise ValueError(f'{side}: independent NPC route has degenerate/discontinuous map segments')
        prefix = float(cumulative(traffic_route)[len(before)])
        station = prefix+float(np.interp(local_spawn, cumulative(ego_route), cumulative(spec['target_route'])))
        lanes[side] = (traffic_route.tolist(), station)
    scope = {'original_source_index_interval': [first, last],
             'original_source_s_interval_m': [float(along[first]), float(along[last-1])],
             'source_station_origin_shift_m': float(along[first]),
             'spawn_s_original_m': float(spawn_s), 'spawn_s_cropped_m': local_spawn,
             'intervention_window_original_source_s_m': [intervention_start, intervention_end],
             'intervention_window_definition': 'nominal shared t0 minus 3m through 4s maintained-speed travel plus 12m tracking/lookahead allowance; actual rollout coverage requires audit',
             'cropped_route_requirement': 'spawn minus 10m through warmup/history/4s strong-acceleration upper bound plus 20m route margin',
             'available_directions_match_intervention_window': True,
             'intervention_window_unavailable_reasons': active_unavailable,
             'npc_routes': 'independent complete real map routes, not cropped ego target routes',
             'uses_observed_future_or_model': False}
    return ego_route, local_spawn, changes, unavailable, lanes, scope


def _diverse_traffic(candidate, family, rng, pre_t0, attempt, randomization=None):
    """Sample a bounded mapped layout using initial geometry, never a rollout."""
    route = candidate['route']
    speed = float(rng.uniform(4.8, 6.5) if 'npc' not in candidate else rng.uniform(4.5, 5.7))
    if randomization:
        speed = randomization['parameters']['ego_initial_speed_m_s']
    target_count = randomization['participants']['vehicle_count_including_ego'] if randomization else 7
    ego_arrival = float(rng.uniform(2.0, 3.1))
    own_s = (candidate['ego_conflict_s']-speed*(pre_t0+ego_arrival) if 'npc' in candidate else candidate['spawn_s'])
    if randomization and 'npc' not in candidate:
        own_s += randomization['parameters']['road_anchor_shift_m']
    ego_route, local_spawn, changes, unavailable, lanes, scope = _diverse_scoped_routes(candidate, own_s, speed, pre_t0)
    cars = [vehicle('ego', ego_route, local_spawn, speed, priority=0, color='195,35,30')]
    slots = [{'role': 'ego', 'route_lane': 'main', 'gap_from_ego_station_m': 0., 'relative_initial_speed_mps': 0.}]

    def add(role, lane, gap, relative_speed=None, desired_delta=None):
        selected_route, station = lanes[lane]
        relative_speed = float(rng.uniform(-.65, .8) if relative_speed is None else relative_speed)
        initial = max(2., speed+relative_speed)
        desired = max(2., speed+float(rng.uniform(-.3, 1.25) if desired_delta is None else desired_delta))
        car = vehicle(role, selected_route, station+gap, initial, desired, priority=1)
        if any(spawn_footprints_overlap(car, other) for other in cars):
            raise ValueError(f'initial bounding envelopes overlap for {role} in {lane}')
        # A low minimum follows from spawn geometry only; it does not promise
        # collision-free futures. Natural following control still runs online.
        for other, slot in zip(cars, slots):
            other_station = (own_s if other['role'] == 'ego' else other['spawn_route_s'])
            if slot['route_lane'] == lane and abs((station+gap)-other_station) < 10.:
                raise ValueError(f'initial same-route centre gap below 10m for {role}')
        cars.append(car)
        slots.append({'role': role, 'route_lane': lane, 'gap_from_ego_station_m': float(gap),
                      'relative_initial_speed_mps': initial-speed, 'desired_speed_mps': desired})

    if family == 'rear_following':
        rear_gap=randomization['parameters']['rear_gap_m'] if randomization else float(rng.uniform(14.,20.))
        relative=randomization['parameters']['rear_relative_speed_m_s'] if randomization else float(rng.uniform(-.2,.6))
        add('rear', 'main', -rear_gap, relative_speed=relative,
            desired_delta=float(rng.uniform(.4, 1.3)))
    for side in changes:
        if randomization:
            from scene_randomization import stream
            reserved = int(family == 'adjacent_parallel')
            required_rear = (randomization['template']=='ego_lane_change_straight'
                and randomization['phase'] in ('tight_gap','closing_rear')
                and not any(c['role'] in ('left_rear','right_rear') for c in cars))
            if not required_rear and (len(cars)+reserved >= target_count or not stream(randomization['seed'],'participant/'+side+'_rear').integers(2)):
                continue
        add(side+'_rear', side, -float(rng.uniform(11., 19.)), relative_speed=float(rng.uniform(-.4, .8)),
            desired_delta=float(rng.uniform(.3, 1.2)))
    if family == 'adjacent_parallel':
        if not changes:
            raise ValueError('adjacent_parallel needs an actual feasible adjacent Driving lane')
        side = list(changes)[int(rng.integers(len(changes)))]
        relative = float(rng.uniform(.4, 2.)) if randomization else float(rng.uniform(.8, 1.4))
        nominal_offset = float(rng.uniform(1., 9.)) if randomization else float(rng.uniform(4., 6.))
        add('parallel_vehicle', side, nominal_offset-relative*pre_t0,
            relative_speed=relative, desired_delta=relative)
        slots[-1].update(nominal_t0_longitudinal_offset_m=nominal_offset,
                        relation='nearby moving car on the adjacent parallel lane, slightly ahead; not a claim of exact side-by-side alignment')
    if 'npc' in candidate:
        npc_speed = float(rng.uniform(3.3, 4.4))
        npc_entry_eta = float(rng.uniform(.6, 1.6))
        npc_s = candidate['priority_zone']['yield_route_s'][0]-npc_speed*(pre_t0+npc_entry_eta)
        npc_arrival = (candidate['npc_conflict_s']-npc_s)/npc_speed-pre_t0
        car = vehicle(candidate['npc_role'], candidate['npc']['route'], npc_s, npc_speed, priority=2)
        if any(spawn_footprints_overlap(car, other) for other in cars):
            raise ValueError('interaction NPC overlaps another initial footprint')
        cars.append(car)
        slots.append({'role': car['role'], 'route_lane': 'independent_approach',
                      'gap_from_ego_station_m': None, 'relative_initial_speed_mps': npc_speed-speed,
                      'nominal_conflict_after_t0_s': npc_arrival,
                      'nominal_priority_body_entry_after_t0_s': npc_entry_eta,
                      'spawn_anchor': 'actual priority-zone body-entry station, not its later centreline conflict'})
    # Fill different lane/longitudinal slots instead of one front+rear per lane.
    # Each continuous jitter, relative speed, model and colour is scene-seeded.
    rejected_slots = []
    for slot_attempt in range(240):
        if len(cars) == target_count:
            break
        lane = list(lanes)[int(rng.integers(len(lanes)))]
        gap = float(rng.choice([-51., -35., -19., -8., 8., 19., 35., 51.])+rng.uniform(-3., 3.))
        if randomization:
            gap=float(rng.uniform(10.,55.))*(-1. if rng.integers(2) else 1.)
            if family=='rear_following' and randomization['phase']=='platoon':
                lane='main';gap=-float(rng.uniform(12.,55.))
        role = f'traffic_{len(cars):02d}'
        try:
            add(role, lane, gap)
        except ValueError as exc:
            rejected_slots.append({'slot_attempt': slot_attempt, 'lane': lane, 'gap_m': gap, 'reason': str(exc)})
    if len(cars) != target_count:
        raise ValueError('could not place requested disjoint vehicles in declared map routes')
    for car in cars[1:]:
        car['blueprint_id'] = DIVERSE_NPC_MODELS[int(rng.integers(len(DIVERSE_NPC_MODELS)))]
        car['blueprint_attributes']['color'] = DIVERSE_COLORS[int(rng.integers(len(DIVERSE_COLORS)))]
    opportunities = [{'role': slot['role'], 'relation': ('ego follower' if slot['role'] == 'rear' else 'potential target-lane follower'),
                      'spawn_gap_m': slot['gap_from_ego_station_m'],
                      'relative_initial_speed_mps': slot['relative_initial_speed_mps']}
                     for slot in slots if slot['role'] == 'rear' or slot['role'] in ('left_rear', 'right_rear')]
    if 'npc' in candidate:
        opportunities.append({'role': candidate['npc_role'], 'relation': candidate['evidence']['type'],
                              'ego_nominal_conflict_after_t0_s': ego_arrival,
                              'npc_nominal_conflict_after_t0_s': npc_arrival,
                              'npc_nominal_priority_body_entry_after_t0_s': npc_entry_eta,
                              't0_phase_requires_actual_capture_audit': True})
    if family == 'adjacent_parallel':
        opportunities.append({'role': 'parallel_vehicle', 'relation': 'nearby car on parallel adjacent lane',
                              'nominal_t0_longitudinal_offset_m': nominal_offset,
                              'actual_t0_offset_requires_capture_audit': True})
    traffic = {'vehicle_count_including_ego': len(cars), 'layout_attempt': int(attempt),
                  'roles_and_slots': slots, 'rejected_initial_slots': rejected_slots,
                  'approach_opportunities': opportunities, 'ego_initial_speed_mps': speed,
                  'gaps_defined_at': 'spawn, before shared warmup and history',
                  'reaction_guaranteed': False, 'future_outcome_filtering': False}
    return cars, traffic, changes, unavailable, scope


def build_diverse_scenes(opendrive_root, seed=20261022, count_per_family=5,
                         history_s=2., driving_warmup_s=1.):
    """Twenty frozen, seeded layouts with five geometrically feasible actions.

    Selection reads OpenDRIVE geometry only. Scene variation is between paired
    initial conditions; all action branches within a scene use the same config.
    Whole-route neighbour availability determines which lateral interventions
    are allowed. Remaining actions are explicit longitudinal profiles.
    """
    import copy
    from collections import Counter
    if __package__:
        from .controllers_v2 import branch_names, select_action_profiles, ScenarioController
    else:
        from controllers_v2 import branch_names, select_action_profiles, ScenarioController

    if not 1 <= count_per_family <= 5 or history_s < 0 or driving_warmup_s < 0:
        raise ValueError('Require one to five cases per family and nonnegative history/warmup')
    root = Path(opendrive_root)
    maps = {name: carla.Map(name, (root/(name+'.xodr')).read_text()) for name in ('Town05', 'Town10HD')}
    hashes = {name: hashlib.sha256((root/(name+'.xodr')).read_bytes()).hexdigest() for name in maps}
    map_lanes = {name: cmap.generate_waypoints(2.) for name, cmap in maps.items()}
    pools, exclusions = {}, {}
    for name, cmap in maps.items():
        pools[name], exclusions[name] = diverse_road_candidates(cmap, name)
    junctions = _diverse_junction_candidates(maps)
    # These strata are declared before any collection. They ensure that the
    # pilot exercises absent, one-sided and bilateral neighbour availability.
    strata = {
        'rear_following': [('Town05', ()), ('Town05', ('left',)), ('Town05', ('right',)),
                           ('Town05', ('left', 'right')), ('Town10HD', ('right',))],
        'adjacent_parallel': [('Town10HD', ('left',)), ('Town05', ('right',)), ('Town05', ('left',)),
                              ('Town05', ('left', 'right')), ('Town05', ('left', 'right'))]}
    selected_origins, scenes, used_junctions = [], [], set()
    for family in FAMILIES:
        family_candidates = junctions.get(family)
        if family_candidates is not None:
            primary = [c for c in family_candidates if family != 'unprotected_intersection' or c['map'] == 'Town10HD']
            fallback = [c for c in family_candidates if family == 'unprotected_intersection' and c['map'] == 'Town05']
            order_rng = np.random.default_rng(np.random.SeedSequence([int(seed), FAMILIES.index(family), 913]))
            family_candidates = [primary[int(i)] for i in order_rng.permutation(len(primary))]
            family_candidates += [fallback[int(i)] for i in order_rng.permutation(len(fallback))]
        for variant in range(count_per_family):
            index = len(scenes)
            scene_seed = int(np.random.SeedSequence([int(seed), FAMILIES.index(family), variant]).generate_state(1)[0])
            rng = np.random.default_rng(scene_seed)
            if family in strata:
                name, directions = strata[family][variant]
                candidates = [dict(v, required_directions=directions) for v in pools[name] if tuple(v['changes']) == directions]
                candidates = [candidates[int(i)] for i in rng.permutation(len(candidates))]
            else:
                candidates = [v for v in family_candidates if (family, v['map'], v['anchor']['junction_id']) not in used_junctions]
            failures, config = [], None
            for candidate in candidates:
                if any(name == candidate['map'] and np.linalg.norm(np.asarray(origin[:2])-candidate['origin'][:2]) < 30.
                       for name, origin in selected_origins) and family in strata:
                    failures.append({'anchor': candidate['anchor'], 'reason': 'road anchor within 30m of earlier selected road anchor'})
                    continue
                for attempt in range(80):
                    try:
                        cars, traffic, changes, unavailable, scope = _diverse_traffic(candidate, family, rng, history_s+driving_warmup_s, attempt)
                        geometry = {'map': candidate['map'], 'origin': candidate['origin'], 'conflicts': [],
                            'type': candidate['evidence']['type'] if 'evidence' in candidate else 'seeded same-direction road traffic',
                            'selection': {'anchor': candidate['anchor'], 'scene_seed': scene_seed,
                                          'outcome_based_selection': False, 'public_road_legal_permission_verified': False},
                            'traffic_design': traffic, 'unavailable_lane_change_directions': unavailable,
                            'ego_route_scope': scope}
                        if 'npc' in candidate:
                            geometry.update(junction_id=candidate['anchor']['junction_id'],
                                main_road_topology=candidate['main_road_topology'], entry_road_topology=candidate['entry_road_topology'],
                                ego_conflict_s=candidate['ego_conflict_s']-scope['source_station_origin_shift_m'],
                                ego_conflict_s_original_map_route=candidate['ego_conflict_s'], npc_conflict_s=candidate['npc_conflict_s'],
                                conflict_xyz=candidate['origin'], interaction_evidence=candidate['evidence'])
                            geometry['priority_zone'] = copy.deepcopy(candidate['priority_zone'])
                            geometry['priority_zone']['priority_roles'] = [v['role'] for v in cars if v['role'] != candidate['npc_role']]
                        config = finish_config(family, index, int(seed), cars, geometry, history_s, driving_warmup_s)
                        config['seed'] = scene_seed
                        config['version'] = 'paired-multicam-diverse-v1'
                        config['scene_profile'] = 'diverse_v1'
                        action = config['action']
                        action.pop('lateral_offset_m')
                        action.update(lateral_mode='adjacent_lane_route', lane_changes=copy.deepcopy(changes),
                                      available_lane_change_directions=list(changes),
                                      reference='shared t0 speed; actual adjacent Driving targets cover the required cropped ego route and are cross-checked against the intervention window')
                        action['profiles'] = select_action_profiles(action['available_lane_change_directions'],
                            braking_deceleration_m_s2=action['braking_deceleration_m_s2'], acceleration_m_s2=action['acceleration_m_s2'],
                            acceleration_speed_increase_cap_m_s=action['acceleration_speed_increase_cap_m_s'])
                        config.pop('branches')
                        config['branches'] = list(branch_names(config))
                        config['controller']['leader_geometry'] = 'route_projection'
                        check_lane_change_future_budget(config)
                        ScenarioController(config)
                        break
                    except ValueError as exc:
                        failures.append({'anchor': candidate['anchor'], 'layout_attempt': attempt, 'reason': str(exc)})
                        config = None
                if config is not None:
                    break
            if config is None:
                raise ValueError(f'Cannot construct declared diverse scene {family}/{variant}; rejected={failures[-5:]}')
            config['geometry']['selection']['rejected_geometry_attempts'] = failures
            config['geometry']['selection']['road_pool_exclusion_counts'] = {
                name: dict(Counter(r['reason'] for r in values)) for name, values in exclusions.items()}
            config['geometry']['selection']['road_pool_exclusions'] = exclusions if index == 0 else {'recorded_in': 'scene_0000/spec.json'}
            nominal_t0 = point_at_s(cars[0]['route'], cars[0]['spawn_route_s']+cars[0]['initial_speed_mps']*(history_s+driving_warmup_s))['position']
            config['local_map_lanes'] = local_lane_geometry(map_lanes[candidate['map']], nominal_t0)
            config['local_map_geometry'] = {'center_xyz': nominal_t0, 'radius_m': 120., 'sample_distance_m': 2.,
                                          'point_order': 'increasing OpenDRIVE s within road/lane/section'}
            config['map_geometry_path'] = 'maps/'+candidate['map']+'.xodr'
            config['provenance'].update(opendrive_sha256=hashes[candidate['map']],
                description='Seeded varied CARLA road/junction layouts; not official Bench2Drive replay.',
                selection='Static map geometry, finite horizon and disjoint initial footprints only; no observed future or model metric.',
                sampling={'generator': 'numpy.default_rng PCG64', 'seed_sequence': [int(seed), FAMILIES.index(family), variant],
                          'scene_seed': scene_seed, 'scene_parameters_frozen_across_all_branches': True,
                          'npc_model_choices': list(DIVERSE_NPC_MODELS), 'npc_color_choices': list(DIVERSE_COLORS)},
                requested_benchmark_scene_count=4*count_per_family, implemented_families=list(FAMILIES),
                scope_status='planned_geometry_not_collected_or_accepted',
                spawn_footprint_check={'shape': 'oriented conservative rectangle', 'length_m': 5.6, 'width_m': 2.3,
                                      'all_pairs_disjoint': True, 'actual_blueprint_bbox_requires_post_collection_audit': True},
                vehicle_assignment='Scene-seeded native-v2 model/color assignments must be preserved by the collector; ego remains Tesla.')
            selected_origins.append((candidate['map'], candidate['origin']))
            if 'junction_id' in candidate['anchor']:
                used_junctions.add((family, candidate['map'], candidate['anchor']['junction_id']))
            scenes.append(config)
    unique_junctions = {(c['map'], c['geometry']['junction_id']) for c in scenes if 'junction_id' in c['geometry']}
    for c in scenes:
        c['provenance']['geography'] = {'road_anchor_count': 2*count_per_family,
            'distinct_junction_identity_count': len(unique_junctions),
            'junction_configuration_count': 2*count_per_family,
            'note': 'Road anchors differ; junction families may reuse a physical junction with distinct real connector interactions. Counts are not independent statistical samples.'}
    return scenes

