"""Versioned ego interventions with current-observation decision state.

No CARLA or NPC mutation occurs here. ``context`` is computed from current
observations and fixed map geometry by controllers_v2; it contains no recorded
future or branch-dependent NPC policy. Waiting and release are latched per
controller instance and are reported for audit.
"""
from __future__ import annotations

import json
import math
import re


KINDS = frozenset(('maintain', 'nominal', 'longitudinal', 'lane_change',
                   'stop_hold', 'yield_wait', 'commit_go'))
DEFAULT_STOP_RESPONSE_TIME_S = .75
DEFAULT_STOP_MARGIN_M = .75
STRICT_YIELD = 'specified_actors_cleared_and_entry_clear_v1'
STRICT_HOLD = 'hold_until_window_end_v1'


def number(value, name, minimum=None, positive=False):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f'{name} requires a finite number')
    value = float(value)
    if minimum is not None and value < minimum or positive and value <= 0:
        raise ValueError(f'{name} is outside the permitted physical range')
    return value


def validate_profiles(settings):
    """Validate v2 policy meanings independently of branch display names."""
    if settings.get('profile_version') != 2:
        raise ValueError('Decision profiles require profile_version=2')
    if settings.get('schema_version', 2) != 2:
        raise ValueError('Decision profiles require schema_version=2')
    profiles = settings.get('profiles')
    if not isinstance(profiles, dict) or len(profiles) != 5 or 'maintain' not in profiles:
        raise ValueError('Decision profiles require five named conditions including maintain')
    names = tuple(settings.get('profile_order', profiles))
    if len(names) != 5 or len(set(names)) != 5 or set(names) != set(profiles):
        raise ValueError('profile_order must enumerate each of five profiles once')
    directions = settings.get('available_lane_change_directions', [])
    if (not isinstance(directions, (list, tuple)) or len(set(directions)) != len(directions)
            or not set(directions).issubset(('left', 'right'))):
        raise ValueError('Lane directions must be distinct left/right values')
    conflicts = settings.get('conflicts', [])
    if not isinstance(conflicts, list):
        raise ValueError('action.conflicts must be a list')
    conflict_ids = set()
    for conflict in conflicts:
        identity = conflict.get('id')
        if not isinstance(identity, str) or not identity or identity in conflict_ids:
            raise ValueError('Action conflict IDs must be unique strings')
        conflict_ids.add(identity)
        enter = number(conflict.get('ego_entry_s'), f'{identity}.ego_entry_s', 0)
        leave = number(conflict.get('ego_exit_s'), f'{identity}.ego_exit_s', 0)
        if leave <= enter:
            raise ValueError('Conflict ego interval must have positive length')
        if 'stop_route_s' in conflict and number(conflict['stop_route_s'], 'conflict.stop_route_s', 0) >= enter:
            raise ValueError('Conflict stop station must precede its entry')
        if not isinstance(conflict.get('ego_must_yield', True), bool):
            raise ValueError('ego_must_yield must be boolean')
        participants = conflict.get('participants')
        if not isinstance(participants, list) or not participants:
            raise ValueError('A decision conflict requires current observed participants')
        seen = set()
        for actor in participants:
            role = actor.get('role')
            if not isinstance(role, str) or not role or role == 'ego' or role in seen:
                raise ValueError('Conflict participants must have unique non-ego roles')
            seen.add(role)
            first = number(actor.get('entry_s'), f'{role}.entry_s', 0)
            last = number(actor.get('exit_s'), f'{role}.exit_s', 0)
            if last <= first:
                raise ValueError('Participant conflict interval must have positive length')
    baseline = settings.get('nominal_policy')
    if not isinstance(baseline, dict):
        raise ValueError('v2 requires an explicit common-history nominal_policy')
    number(baseline.get('desired_speed_m_s'), 'nominal.desired_speed_m_s', positive=True)
    for key in ('max_acceleration', 'comfortable_brake', 'time_headway', 'minimum_gap'):
        number(baseline.get('idm', {}).get(key), f'nominal.idm.{key}', positive=True)
    for key in ('release_gap_s', 'comfortable_deceleration_m_s2'):
        number(baseline.get(key), f'nominal.{key}', positive=True)
    number(baseline.get('response_time_s', DEFAULT_STOP_RESPONSE_TIME_S), 'nominal.response_time_s', 0)
    if baseline.get('respect_signals') is not True:
        raise ValueError('The common normal driving policy must respect signals')
    signatures = set()
    for name, profile in profiles.items():
        if not isinstance(name, str) or not re.fullmatch(r'[a-z][a-z0-9_]*', name) or name == 'maintain_repeat':
            raise ValueError('Model branch names must be safe unique non-repeat identifiers')
        if not isinstance(profile, dict) or profile.get('kind') not in KINDS:
            raise ValueError(f'{name}: unsupported decision kind')
        kind = profile['kind']
        if 'terminal_brake_min' in profile:
            brake = number(profile['terminal_brake_min'], f'{name}.terminal_brake_min', .4)
            if kind not in ('stop_hold','yield_wait') or brake > 1.:
                raise ValueError('terminal_brake_min requires stop_hold or yield_wait and a normalized command in [0.4, 1]')
        if 'response_time_s' in profile:
            number(profile['response_time_s'], f'{name}.response_time_s', 0)
        if (name == 'maintain') != (kind == 'maintain'):
            raise ValueError('maintain is the single speed-reference profile')
        if kind in ('maintain', 'lane_change') and any(profile.get(key, 0) != 0 for key in ('acceleration_m_s2', 'speed_increase_cap_m_s')):
            raise ValueError(f'{name}: speed-reference policies require zero acceleration and speed cap')
        onset = number(profile.get('onset_s', 0), f'{name}.onset_s', 0)
        if kind in ('maintain', 'nominal') and onset != 0:
            raise ValueError('Reference and nominal continuation start at t0')
        if kind in ('longitudinal', 'commit_go'):
            acceleration = number(profile.get('acceleration_m_s2'), f'{name}.acceleration_m_s2')
            cap = number(profile.get('speed_increase_cap_m_s'), f'{name}.speed_increase_cap_m_s', 0)
            number(profile.get('min_target_speed_m_s', 0), f'{name}.min_target_speed_m_s', 0)
            if kind == 'longitudinal' and (acceleration == 0 or acceleration > 0 and cap <= 0 or acceleration < 0 and cap != 0):
                raise ValueError('Longitudinal acceleration/cap must define a nonzero intervention')
            if kind == 'commit_go' and (acceleration < 0 or not isinstance(profile.get('respect_signals'), bool)):
                raise ValueError('commit_go requires nonnegative acceleration and explicit signal policy')
        if kind == 'lane_change':
            if profile.get('direction') not in directions:
                raise ValueError('Lane-change direction must have a mapped adjacent lane')
            number(profile.get('duration_s'), f'{name}.duration_s', positive=True)
            number(profile.get('settling_time_s', 0.), f'{name}.settling_time_s', 0)
            number(profile.get('required_final_hold_s', .5), f'{name}.required_final_hold_s', .5)
        if kind in ('stop_hold', 'yield_wait') or kind == 'commit_go' and onset > 0:
            number(profile.get('stop_route_s'), f'{name}.stop_route_s', 0)
            if 'comfortable_deceleration_m_s2' in profile:
                number(profile['comfortable_deceleration_m_s2'], f'{name}.comfortable_deceleration_m_s2', positive=True)
            margin = number(profile.get('stop_margin_m', 0.), f'{name}.stop_margin_m', 0)
            if margin > profile['stop_route_s']:
                raise ValueError(f'{name}: stop margin puts the control station before the route')
        if 'stop_deadline_s' in profile:
            if kind not in ('stop_hold', 'yield_wait'):
                raise ValueError('Only stopping/waiting profiles declare a stop deadline')
            deadline = number(profile['stop_deadline_s'], f'{name}.stop_deadline_s', positive=True)
            tau = float(profile.get('response_time_s', baseline.get('response_time_s', DEFAULT_STOP_RESPONSE_TIME_S)))
            if deadline <= onset + tau:
                raise ValueError(f'{name}: stop deadline leaves no braking time after response reserve')
            end = profile.get('hold_until_s') if kind == 'stop_hold' else profile.get('timeout_s')
            if deadline > number(end, f'{name}.stop/hold end', positive=True):
                raise ValueError(f'{name}: stop deadline follows the hold or waiting window')
        if kind in ('stop_hold', 'yield_wait') and 'resume_speed_m_s' in profile:
            number(profile['resume_speed_m_s'], f'{name}.resume_speed_m_s', positive=True)
        if kind == 'stop_hold':
            if 'hold_policy' in profile and profile['hold_policy'] != STRICT_HOLD:
                raise ValueError('Unknown stop_hold policy')
            if number(profile.get('hold_until_s'), f'{name}.hold_until_s', 0) < onset:
                raise ValueError('stop_hold cannot end before intervention onset')
        if kind == 'yield_wait':
            if profile.get('conflict_id') not in conflict_ids:
                raise ValueError('yield_wait must identify a mapped conflict')
            number(profile.get('min_wait_s', .25), f'{name}.min_wait_s', 0)
            number(profile.get('release_gap_s'), f'{name}.release_gap_s', positive=True)
            if number(profile.get('timeout_s'), f'{name}.timeout_s', positive=True) <= onset:
                raise ValueError('yield_wait timeout must follow intervention onset')
            if 'release_condition' in profile:
                if profile['release_condition'] != STRICT_YIELD:
                    raise ValueError('Unknown yield release condition')
                conflict = next(x for x in conflicts if x['id'] == profile['conflict_id'])
                expected = {x['role'] for x in conflict['participants']}
                wait = profile.get('wait_for_roles')
                if (not isinstance(wait, list) or len(wait) != len(set(wait))
                        or set(wait) != expected):
                    raise ValueError('Strict yield wait_for_roles must name all declared conflict participants')
                if profile.get('min_wait_s', 0) != 0:
                    raise ValueError('Strict yield permits rolling release, not a minimum stationary dwell')
            elif 'wait_for_roles' in profile:
                raise ValueError('wait_for_roles requires a versioned strict release condition')
        # Notes/risk labels cannot turn identical executable profiles into two actions.
        executable = {key: value for key, value in profile.items()
                      if key not in ('description', 'label', 'risk_label', 'risk_condition', 'risk_conditions')}
        signature = json.dumps(executable, sort_keys=True)
        if signature in signatures:
            raise ValueError('Duplicate executable ego profiles are not independent action conditions')
        signatures.add(signature)
    return names


def longitudinal_target(profile, time_s, initial_speed):
    """Shared ideal speed target for execution, offline estimates and audits.

    The deceleration floor cannot accelerate an ego already below that floor.
    Positive acceleration is limited by its increase cap, not by the floor.
    Delayed commit_go uses this only after onset; its approach is stateful.
    """
    elapsed = max(0., float(time_s)-float(profile.get('onset_s', 0)))
    acceleration = float(profile.get('acceleration_m_s2', 0))
    floor = min(float(initial_speed), float(profile.get('min_target_speed_m_s', 0)))
    if acceleration < 0:
        return max(floor, float(initial_speed)+acceleration*elapsed)
    return float(initial_speed)+min(acceleration*elapsed, float(profile.get('speed_increase_cap_m_s', 0)))


def stop_target(station, stop_station, deceleration, response_time_s=DEFAULT_STOP_RESPONSE_TIME_S):
    """Desired speed reserving distance for actuator/controller response.

    Solve d = v*tau + v**2/(2*a), instead of assuming instantaneous tracking
    of an ideal deceleration. Tau is a frozen controller setting, not an
    acceptance tolerance; actual stop position still needs native validation.
    """
    distance = float(stop_station)-float(station)
    a = number(deceleration, 'stop.deceleration', positive=True)
    tau = number(response_time_s, 'stop.response_time_s', 0)
    offset = a*tau
    return max(0., math.sqrt(offset*offset + 2*a*max(0., distance))-offset), distance


def deadline_stop_target(time_s, onset_s, deadline_s, initial_speed, deceleration,
                         response_time_s=DEFAULT_STOP_RESPONSE_TIME_S):
    """A speed cap anchored to measured t0 speed, independent of line distance.

    The zero target precedes the actual-stop deadline by the declared response
    reserve. A distant maximum stopping line is not a demand to reach that line.
    Reject a time budget requiring more than the declared braking capability;
    the resulting physical stop still requires execution auditing.
    """
    speed = number(initial_speed, 'stop.measured_t0_speed', 0)
    a = number(deceleration, 'stop.deceleration', positive=True)
    tau = number(response_time_s, 'stop.response_time_s', 0)
    duration = (number(deadline_s, 'stop.deadline_s', positive=True)-tau
                -number(onset_s, 'stop.onset_s', 0))
    if duration <= 0 or speed > a*duration+1e-9:
        raise ValueError('Stop deadline is physically infeasible at measured t0 speed with declared deceleration/response reserve')
    elapsed = max(0., float(time_s)-float(onset_s))
    return speed*max(0., 1.-elapsed/duration)


class EgoDecisionState:
    def __init__(self):
        self.active_branch = None
        self.stopped_at = None
        self.released_at = None
        self.timed_out = False
        self.committed = False

    def _strict_yield(self, profile, t, speed, resume_speed, context, stopping_target,
                      stopped_hold, diagnostic):
        """Re-evaluate permission every tick; a prior release is not a waiver."""
        conflict = context['conflicts'][profile['conflict_id']]
        current = {x['role']: x for x in conflict['participants']}
        roles = list(profile['wait_for_roles'])
        if any(role not in current or 'body_cleared' not in current[role] for role in roles):
            raise ValueError('Strict yield requires measured body-clearance evidence for every waiting role')
        guard = conflict.get('entry_guard')
        if not isinstance(guard, dict) or not isinstance(guard.get('clear'), bool):
            raise ValueError('Strict yield requires a current observed entry guard')
        pending = [role for role in roles if not current[role]['body_cleared']]
        signal = context.get('signal')
        signal_clear = signal is None or not signal['must_stop']
        permitted = not pending and guard['clear'] and signal_clear
        if self.released_at is None and t >= float(profile['timeout_s']):
            self.timed_out = True
        first = self.released_at is None and not self.timed_out and permitted
        if first:
            self.released_at = t
        released = self.released_at is not None
        diagnostic.update(yield_semantics=STRICT_YIELD, conflict=conflict,
            wait_for_roles=roles, pending_wait_roles=pending,
            all_wait_targets_cleared=not pending, entry_clear=bool(guard['clear'] and signal_clear),
            entry_blocking_roles=guard['blocking_roles'], entry_permitted_now=bool(permitted),
            release_reason='specified_actors_cleared_and_entry_clear' if first else None,
            rolling_release=bool(first and speed > .15), reblocked=bool(released and not permitted))
        if released and permitted:
            # Still follow the normal observed leader/signal/pedestrian safety
            # cap. Named targets clearing cannot force another actor to yield.
            return min(resume_speed, float(context['nominal_target'])), False, 'released'
        if released:
            # Never aim back at the old stopping line after entering the zone.
            # The current obstruction defines the braking cap instead.
            cap = float(guard['target_speed_limit_m_s'])
            if pending:
                cap = 0.  # a target re-entered; stop rather than retain permission
            cap = min(cap, float(context['nominal_target']), resume_speed, speed)
            if speed <= .15:
                cap = 0.  # an occupied entry never authorizes a stopped restart
            hold = bool(cap <= .01)
            return cap, hold, 'released_reblocked'
        phase = ('timeout_hold' if stopped_hold else 'timeout_stopping') if self.timed_out else (
            'holding' if stopped_hold else 'stopping')
        target = min(stopping_target, float(guard['target_speed_limit_m_s']))
        return (0. if stopped_hold else target), bool(stopped_hold), phase

    def evaluate(self, branch, profile, time_s, initial_speed, context):
        """Return target, stopped-brake clamp, and an inspectable policy state."""
        kind = profile['kind']
        if self.active_branch not in (None, branch):
            raise ValueError('A stateful controller cannot switch ego intervention mid-rollout')
        self.active_branch = branch
        t = float(time_s)
        station, speed = float(context['route_s']), float(context['speed'])
        nominal = float(context['nominal_target'])
        resume_speed = float(profile.get('resume_speed_m_s',
            max(float(initial_speed), float(context.get('nominal_desired_speed_m_s', initial_speed)))))
        onset = float(profile.get('onset_s', 0))
        response_time = float(profile.get('response_time_s',
            context.get('stop_response_time_s', DEFAULT_STOP_RESPONSE_TIME_S)))
        target, force_hold = float(initial_speed), False
        diagnostic = {'kind': kind, 'branch': branch, 'phase': 'active',
                      'onset_s': onset, 'ego_route_s': station,
                      'risk_condition': profile.get('risk_condition', profile.get('risk_label')),
                      'signal': context.get('signal'), 'npc_policy_modified': False}
        if kind == 'nominal':
            target = nominal
            diagnostic.update(phase='normal_policy_continuation', baseline=context['nominal_diagnostic'])
            force_hold = context.get('nominal_hold', False)
        elif kind == 'longitudinal':
            target = longitudinal_target(profile, t, initial_speed)
            diagnostic['phase'] = 'before_onset' if t < onset else 'active'
        elif kind in ('stop_hold', 'yield_wait') and t < onset:
            diagnostic['phase'] = 'before_onset'
        elif kind in ('stop_hold', 'yield_wait') or kind == 'commit_go' and t < onset:
            deceleration = profile.get('comfortable_deceleration_m_s2', context['stop_deceleration'])
            margin = float(profile.get('stop_margin_m', 0.))
            stopping, control_remaining = stop_target(station, float(profile['stop_route_s'])-margin,
                deceleration, response_time)
            remaining = float(profile['stop_route_s'])-station
            target = min(float(initial_speed), stopping)
            diagnostic.update(stop_route_s=float(profile['stop_route_s']), remaining_to_stop_m=remaining,
                              stop_station_overshoot_m=max(0., -remaining), response_time_s=response_time)
            if margin:
                diagnostic.update(stop_margin_m=margin, control_stop_route_s=float(profile['stop_route_s'])-margin,
                                  remaining_to_control_stop_m=control_remaining)
            deadline_target = None
            if 'stop_deadline_s' in profile:
                deadline_target = deadline_stop_target(t, onset, profile['stop_deadline_s'],
                                                       initial_speed, deceleration, response_time)
                target = min(target, deadline_target)
                diagnostic.update(stop_deadline_s=float(profile['stop_deadline_s']),
                    deadline_target_speed_m_s=deadline_target,
                    target_zero_deadline_s=float(profile['stop_deadline_s'])-response_time)
            # Once genuinely stationary, do not creep forward under the PI bias.
            # A t0 queue may already be stationary upstream of the latest safe
            # stopping station. Do not drive toward the conflict just to latch
            # a stop, and do not make it wait forever because v0 was zero.
            if speed <= .15 and (control_remaining <= .5 or initial_speed <= .15
                    or deadline_target is not None and deadline_target <= .01) and self.stopped_at is None:
                self.stopped_at = t
            if self.stopped_at is not None:
                diagnostic['stationary_before_requested_stop'] = bool(remaining > .5)
            phase = 'stopping' if self.stopped_at is None else 'holding'
            if kind == 'yield_wait' and profile.get('release_condition') == STRICT_YIELD:
                target, force_hold, phase = self._strict_yield(profile, t, speed, resume_speed,
                    context, target, self.stopped_at is not None or control_remaining <= 0 or deadline_target == 0.,
                    diagnostic)
            elif kind == 'yield_wait':
                diagnostic['yield_semantics'] = 'legacy_gap_after_stop_v0'
                conflict = context['conflicts'][profile['conflict_id']]
                diagnostic['conflict'] = conflict
                signal = context.get('signal')
                # A red lamp no longer bars clearing after the ego has already
                # passed its stop line. Before that line, Red/Yellow blocks.
                signal_clear = signal is None or not signal['must_stop']
                waited = self.stopped_at is not None and t-self.stopped_at >= float(profile.get('min_wait_s', .25))
                arrival = conflict['next_arrival_s']
                clear = (not conflict['occupied'] and (arrival is None or arrival >= float(profile['release_gap_s'])))
                # A release is a decision to enter; after entry it is not revoked
                # by another approaching actor, avoiding a stop inside the zone.
                if self.released_at is None and t >= float(profile['timeout_s']):
                    self.timed_out = True
                if self.released_at is None and not self.timed_out and waited and clear and signal_clear:
                    self.released_at = t
                if self.released_at is not None:
                    target, phase = resume_speed, 'released'
                    # A release is not permission to run a subsequently red
                    # signal while still upstream of its physical stop line.
                    if signal is not None and signal['must_stop']:
                        distance = max(0., signal['distance_front_to_stop_line_m']-1.)
                        target = min(target, stop_target(0., distance, context['stop_deceleration'], response_time)[0])
                        force_hold = distance <= .1
                        phase = 'released_signal_stop'
                elif self.timed_out:
                    phase = 'timeout_hold' if self.stopped_at is not None else 'timeout_stopping'
            elif kind == 'stop_hold' and profile.get('hold_policy') == STRICT_HOLD:
                diagnostic['hold_policy'] = STRICT_HOLD
            elif kind == 'stop_hold' and t > float(profile['hold_until_s']):
                self.released_at = self.released_at if self.released_at is not None else t
                target, phase = resume_speed, 'released'
            elif kind == 'commit_go':
                phase = 'await_entry_time'
            if self.released_at is None:
                force_hold = self.stopped_at is not None or control_remaining <= 0 or deadline_target == 0.
                if force_hold:
                    target = 0.
            diagnostic['phase'] = phase
        elif kind == 'commit_go':
            self.committed = True
            target = longitudinal_target(profile, t, initial_speed)
            diagnostic['phase'] = 'committed_entry'
            diagnostic['yield_override'] = True
            signal = context.get('signal')
            if profile['respect_signals'] and signal is not None and signal['must_stop']:
                distance = max(0., signal['distance_front_to_stop_line_m']-1.)
                target = min(target, stop_target(0., distance, context['stop_deceleration'], response_time)[0])
                force_hold = distance <= .1
                diagnostic['phase'] = 'signal_stop_before_entry'
        elif kind == 'lane_change':
            diagnostic['phase'] = 'before_lane_change' if t < onset else 'changing_lane'
            diagnostic['direction'] = profile['direction']
        diagnostic.update(target_speed_m_s=float(target), stopped_at_s=self.stopped_at,
                          released_at_s=self.released_at, timed_out=self.timed_out,
                          force_brake_hold=bool(force_hold), resume_speed_m_s=resume_speed)
        return max(0., target), bool(force_hold), diagnostic
