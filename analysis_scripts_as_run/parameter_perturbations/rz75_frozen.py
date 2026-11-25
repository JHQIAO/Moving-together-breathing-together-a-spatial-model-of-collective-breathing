"""RZ75: periodic Couzin shoaling with reserve-dependent surface access.

Standalone NumPy implementation, consolidated from the frozen Sim25 path.
Units are body lengths and seconds. States: 0 stay, 1 leave, 2 accumulation,
3 turn, 4 back. Numerical operation order and independent RNG streams are
preserved for byte-for-byte replay on the tested NumPy/platform environment.
See README.md for equation mapping, provenance, and reproducibility limits.
"""
from dataclasses import dataclass, field, replace
import numpy as np

STATE_NAMES = ('stay', 'leave', 'accumulation', 'turn', 'back')

@dataclass(frozen=True)
class Params:
    """Active RZ75 parameters; defaults are the saved A_RZ75_s7301 values."""
    N: int = 200
    box: tuple = (10.0, 3.3, 3.3)
    dt: float = 0.025
    speed: float = 0.6
    rr: float = 0.15
    ro: float = 0.45
    ra: float = 2.95
    vision_k: int = 24
    fov_deg: float = 335.0
    turning_rate: float = 3.141592653589793
    eta: float = 0.3
    surface_strength: float = 0.3
    glide_layer: float = 0.6
    guard_margin: float = 0.05
    upward_strength: float = 0.35
    consumption_rates: tuple = (0.00840887578389495, 0.005158809198668612, 0.003100327856481332)
    type_ratio: tuple = (1.0, 1.0, 1.0)
    trigger_depth: float = 0.6
    breath_time: float = 0.15
    refill_level: float = 0.95
    return_depth: float = 0.7
    fermi_midpoint: float = 0.8
    fermi_temperature: float = 0.1
    decision_timescale: float = 0.3
    ascent_speed: float = 2.1
    return_turning_rate: float = 7.853981633974483
    release_depth: float = 0.5
    reaction_interval: float = 0.05
    need_setpoint: float = 0.75
    surface_layer: float = 1.5
    dilution: bool = True
    dilution_radius: float = 1.0
    dilution_min: float = 0.2
    dilution_scale: float = 6.0
    vertical_noise_scale: float = 0.2
    need_down: float = 0.1
    descent_speed: float = 0.6
    kappa_bottom: float = 0.3
    rest_height: float = 1.0
    return_clearance: float = field(init=False, default=0.1)

    def __post_init__(self):
        object.__setattr__(self, 'return_clearance', self.return_depth-self.trigger_depth)
        if isinstance(self.N, bool) or int(self.N) != self.N or self.N < 1:
            raise ValueError('N must be a positive integer')
        if self.vision_k < 1 or int(self.vision_k) != self.vision_k:
            raise ValueError('vision_k must be a positive integer')
        if len(self.box) != 3 or not np.isfinite(self.box).all() or min(self.box) <= 0:
            raise ValueError('box must contain three positive finite lengths')
        vals = [getattr(self, key) for key in self.__dataclass_fields__
                if key not in ('box', 'type_ratio', 'consumption_rates')]
        if not np.isfinite(vals).all():
            raise ValueError('all scalar parameters must be finite')
        if min(self.dt,self.speed,self.descent_speed,self.ascent_speed,self.turning_rate,
               self.return_turning_rate,self.breath_time,self.fermi_temperature,
               self.decision_timescale,self.surface_layer,self.dilution_radius,
               self.dilution_scale,self.rest_height) <= 0:
            raise ValueError('time, speed, scale and layer parameters must be positive')
        if min(self.eta,self.surface_strength,self.upward_strength,self.need_down,self.kappa_bottom) < 0:
            raise ValueError('noise and steering strengths must be nonnegative')
        if not (0 < self.rr < self.ro < self.ra and 0 < self.fov_deg <= 360):
            raise ValueError('invalid interaction geometry')
        if not 0 < self.guard_margin < self.glide_layer < min(self.box)/2:
            raise ValueError('invalid boundary layer geometry')
        if not 0 < self.trigger_depth < self.return_depth < self.box[2]:
            raise ValueError('invalid decision/return depths')
        if not 0 < self.release_depth < self.box[2]:
            raise ValueError('invalid release depth')
        if not (0 <= self.need_setpoint <= 1 and 0 <= self.fermi_midpoint <= 1
                and 0 < self.refill_level <= 1 and 0 < self.dilution_min <= 1
                and 0 <= self.vertical_noise_scale <= 1):
            raise ValueError('invalid fraction')
        ratio=self.reaction_interval/self.dt
        if ratio < 1 or not np.isclose(ratio,round(ratio),rtol=0,atol=1e-10):
            raise ValueError('reaction interval must be an integer multiple of dt')
        if self.dt > self.breath_time+1e-12 or not np.isclose(round(self.breath_time/self.dt)*self.dt,self.breath_time):
            raise ValueError('dt must resolve dwell time exactly')
        for key in ('consumption_rates','type_ratio'):
            value=np.asarray(getattr(self,key),dtype=float)
            if value.shape != (3,) or not np.isfinite(value).all() or (value < 0).any():
                raise ValueError('three finite nonnegative rates/type weights required')
        if np.sum(self.type_ratio) <= 0:
            raise ValueError('type weights must have positive sum')

    @property
    def max_turn(self):
        return min(np.pi,self.turning_rate*self.dt)


def parameters(**changes):
    """Construct frozen defaults; cruise-speed changes also set descent by default."""
    if 'speed' in changes and 'descent_speed' not in changes:
        changes['descent_speed']=changes['speed']
    return Params(**changes)

def _row_normalize(v):
    """L2-normalize rows; zero rows stay zero (as sklearn normalize did)."""
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return np.divide(v, n, out=np.zeros_like(v), where=n > 0)

def minimum_image(displacement, box):
    """Shortest displacement on x/y torus; leave vertical separation intact."""
    result = np.array(displacement, dtype=float, copy=True)
    lengths = np.asarray(box)[:2]
    result[..., :2] -= lengths * np.rint(result[..., :2] / lengths)
    return result

def initial_reserves(n,seed):
    """Independent RNG keeps the previously tested movement initial conditions."""
    return np.random.default_rng(np.random.SeedSequence([seed,417])).uniform(.5,1.,n)

def type_counts(n,ratio):
    """Largest-remainder allocation; fixed fast/medium/slow tie order."""
    target=n*np.asarray(ratio,dtype=float)/np.sum(ratio)
    counts=np.floor(target).astype(int)
    order=np.argsort(-(target-counts),kind='stable')
    counts[order[:n-int(counts.sum())]]+=1
    return counts

def assign_types(n,ratio,seed):
    counts=type_counts(n,ratio)
    order=np.random.default_rng(np.random.SeedSequence([seed,618])).permutation(n)
    labels=np.empty(n,dtype=np.int8)
    labels[order]=np.repeat(np.arange(3),counts)
    return labels

def scaled_need(tank, midpoint, temperature):
    q=np.asarray(tank,dtype=float)
    if not np.isfinite([midpoint,temperature]).all() or not 0<=midpoint<=1 or temperature<=0:
        raise ValueError('midpoint must be in [0,1], temperature positive, both finite')
    if not np.isfinite(q).all() or ((q<0)|(q>1)).any():
        raise ValueError('tank must be a finite fraction in [0,1]')
    with np.errstate(over='ignore'):
        return (midpoint-q)/temperature

def decision_probability(tank, dt, midpoint=.5, temperature=.1, timescale=1.):
    """P_dt = 1-(1-P_F)^(dt/timescale), evaluated without log(0) or overflow.

    At fixed q, the chance of at least one acceptance over timescale seconds
    spent continuously eligible equals the Fermi probability exactly. q and
    eligibility are updated at each integration step. timescale is physical
    time, not an integration step or a compulsory waiting/refractory period.
    """
    if not np.isfinite([dt,timescale]).all() or dt<=0 or timescale<=0:
        raise ValueError('dt and decision timescale must be positive and finite')
    x=scaled_need(tank,midpoint,temperature)
    with np.errstate(over='ignore',under='ignore'):
        return -np.expm1(-(dt/timescale)*np.logaddexp(0.,x))

def school_band_counts(pos,heading,status,p,ids):
    """Nearest-visible selection precedes band/state filtering; no self count."""
    ids=np.asarray(ids,dtype=int);counts=np.zeros((len(ids),5),dtype=np.int16)
    if p.N<2 or not len(ids):return counts,np.zeros(len(ids),dtype=np.int16)
    delta=minimum_image(pos[None,:,:]-pos[ids,None,:],p.box)
    distance=np.linalg.norm(delta,axis=2)
    unit=np.divide(delta,distance[:,:,None],out=np.zeros_like(delta),where=distance[:,:,None]>0)
    visible=(np.einsum('ijk,ik->ij',unit,heading[ids])>=np.cos(np.deg2rad(p.fov_deg/2))-1e-12)&(distance>0)
    k=min(p.vision_k,p.N-1)
    near=np.argpartition(np.where(visible,distance,np.inf),k-1,axis=1)[:,:k]
    valid=np.take_along_axis(visible,near,axis=1)
    band=pos[:,2]>=p.box[2]-p.trigger_depth-1e-12
    for state in range(5):counts[:,state]=(valid&band[near]&(status[near]==state)).sum(axis=1)
    return counts,valid.sum(axis=1).astype(np.int16)

TRIAL_DTYPE = [('time', 'f8'), ('agent', 'i4'), ('z', 'f8'), ('tank', 'f8'), ('probability', 'f8'), ('draw', 'f8'), ('accepted', '?')]
BAND_DTYPE = np.dtype([('time', '<f8'), ('agent', '<i4'), ('n0', '<i2'), ('n1', '<i2'), ('n2', '<i2'), ('n3', '<i2'), ('n4', '<i2'), ('visible', '<i2'), ('p_tank', '<f8'), ('p_social', '<f8')])

class RZ75Sim:
    """One simulation; independent movement, reserve, type and decision streams."""
    def __init__(self,params=None,seed=7,reserves=None,audit_burn=300.):
        self.p=params or parameters(); p=self.p
        self.seed=seed; self.t=0.; self.total_steps=0
        self.rng=np.random.default_rng(seed)
        self.pos=self.rng.uniform(size=(p.N,3))*np.asarray(p.box)
        self.dir=_row_normalize(self.rng.normal(size=(p.N,3)))
        self.initialization=dict(kind='random')
        self.reflections=np.zeros(3,dtype=int)
        self.floor_hits=0; self.surface_hits=0
        self.corrected_axes=np.zeros(3,dtype=int)
        self.predictive_turns=0; self.smooth_turns=0
        self.turn_cap_excess=0; self.pitch_cap_excess=0
        self.max_step_turn_deg=0.; self.max_ordinary_turn_deg=0.; self.max_pitch_step_deg=0.
        self.wraps=np.zeros(2,dtype=int)
        q=initial_reserves(self.p.N,seed) if reserves is None else np.array(reserves,dtype=float,copy=True)
        if q.shape!=(self.p.N,) or not np.isfinite(q).all() or ((q<0)|(q>1)).any():
            raise ValueError('reserves must be N finite fractions in [0,1]')
        self.type_id=assign_types(self.p.N,self.p.type_ratio,seed)
        self.agent_consumption_rates=np.asarray(self.p.consumption_rates)[self.type_id]
        self.reserves=q
        self.status=np.zeros(self.p.N,dtype=np.int8)
        self.arrival_time=np.full(self.p.N,np.nan)
        self.events=[]
        self.breath_counts=np.zeros(self.p.N,dtype=np.int64)
        self.consumed=np.zeros(self.p.N)
        self.refilled=np.zeros(self.p.N)
        self.max_budget_error=0.
        self.empty_agent_seconds=0.
        self.state_agent_seconds=np.zeros(5)
        self.max_free_displacement_error=0.
        if not np.isfinite(audit_burn) or audit_burn < 0:
            raise ValueError('audit_burn must be finite and nonnegative')
        self.audit_burn = float(audit_burn)
        # One draw for every agent at every eligible clock step, independent of
        # eligibility. Matched runs retain identical draws by agent and step.
        self.decision_rng = np.random.default_rng(np.random.SeedSequence([seed,829]))
        self.trial_batches = []
        self.empty_postburn_agent_seconds = 0.
        self.episodes=[]
        self.cycle=[None for _ in range(self.p.N)]
        self.departure_log=[]
        self.max_state_turn_excess=0.
        self.postburn_state_turn_violations=0
        self.postburn_corrections=0
        self.floor_limited_agent_seconds=0.
        self.floor_limited_postburn_seconds=0.
        self.reaction_stride=round(self.p.reaction_interval/self.p.dt)
        self.reaction_count=0
        self.cached_desired=None;self.cached_social=None
        self.entry_armed=np.ones(self.p.N,dtype=bool)
        self.entry_id=np.zeros(self.p.N,dtype=int)
        self.entry_log=[]
        self.band_previous=np.zeros(self.p.N,dtype=bool)
        self.neighbour_sum=0.;self.neighbour_observations=0
        self.band_batches=[]
        self.noise_sum=np.zeros(3);self.noise_squares=np.zeros(3);self.noise_n=0

    @property
    def protected_axes(self):
        return (2,)

    def _cap_turn(self, desired):
        """Rotate self.dir towards `desired`, at most max_turn per step.

        Rows within the cap return `desired` unchanged (noise included, as in
        the legacy turning()); rows over it rotate the CURRENT unit heading by
        exactly max_turn about the mutual normal.
        """
        p = self.p
        cur = self.dir
        d_norm = np.linalg.norm(desired, axis=1)
        c_norm = np.linalg.norm(cur, axis=1)
        out = desired.copy()
        # a fully cancelled social sum gives a zero desired direction (only
        # reachable at eta=0): keep the heading rather than freeze the agent
        zero_d = (d_norm == 0) & (c_norm > 0)
        out[zero_d] = cur[zero_d]
        safe = (d_norm > 0) & (c_norm > 0)
        cos = np.zeros(p.N)
        np.divide(np.einsum('ij,ij->i', desired, cur), d_norm * c_norm,
                  out=cos, where=safe)
        ang = np.arccos(np.clip(cos, -1.0, 1.0))
        need = safe & (ang >= p.max_turn)
        if not need.any():
            return out

        c = cur[need]
        axis = np.cross(c, desired[need])
        a_norm = np.linalg.norm(axis, axis=1, keepdims=True)
        # antiparallel targets give a zero cross product: rotate about any
        # axis perpendicular to the current heading instead of dividing by 0
        degen = (a_norm[:, 0] < 1e-12)
        if degen.any():
            alt = np.cross(c[degen], np.array([1.0, 0.0, 0.0]))
            alt_bad = np.linalg.norm(alt, axis=1) < 1e-12
            if alt_bad.any():
                alt[alt_bad] = np.cross(c[degen][alt_bad],
                                        np.array([0.0, 1.0, 0.0]))
            axis[degen] = alt
            a_norm = np.linalg.norm(axis, axis=1, keepdims=True)
        axis /= a_norm
        # axis is perpendicular to c, so Rodrigues reduces to two terms
        out[need] = (c * np.cos(p.max_turn)
                     + np.cross(axis, c) * np.sin(p.max_turn))
        return out

    def cap_at(self,target,rate):
        # Same exact-turn fallback as inherited _cap_turn, at a distinct motor
        # rate. The frozen dataclass is replaced only for this pure operation.
        from dataclasses import replace
        old=self.p
        self.p=replace(old,turning_rate=rate)
        try:return self._cap_turn(target)
        finally:self.p=old

    def social_direction(self):
        p = self.p
        dif = minimum_image(self.pos[None, :, :] - self.pos[:, None, :], p.box)
        dist = np.linalg.norm(dif, axis=2)
        unit = np.divide(dif, dist[:, :, None], out=np.zeros_like(dif),
                         where=dist[:, :, None] > 0)
        cosine = np.einsum('ijk,ik->ij', unit, self.dir)
        visible = cosine >= np.cos(np.deg2rad(p.fov_deg/2)) - 1e-12
        rep = (dist > 0) & (dist < p.rr)
        ori = (dist >= p.rr) & (dist < p.ro) & visible
        att = (dist >= p.ro) & (dist < p.ra) & visible
        if p.vision_k is not None and p.N > 1:
            k = min(p.vision_k, p.N-1)
            ranked = np.where(visible & (dist > 0), dist, np.inf)
            nearest = np.argpartition(ranked, k-1, axis=1)[:, :k]
            topo = np.zeros_like(visible)
            np.put_along_axis(topo, nearest, True, axis=1)
            ori &= topo
            att &= topo
        in_rep = rep.any(axis=1)
        dr = -_row_normalize(np.einsum('ijk,ij->ik', unit, rep))
        do = _row_normalize(ori.astype(float) @ self.dir)
        da = _row_normalize(np.einsum('ijk,ij->ik', unit, att))
        direction = _row_normalize(do + da)
        cancelled = np.linalg.norm(direction, axis=1) < 1e-12
        direction[cancelled] = self.dir[cancelled]
        direction[in_rep] = dr[in_rep]
        cancelled = np.linalg.norm(direction, axis=1) < 1e-12
        direction[cancelled] = self.dir[cancelled]
        return direction, in_rep

    def nearby_count(self):
        p=self.p
        if p.N==1:return np.zeros(1,dtype=int)
        delta=minimum_image(self.pos[None,:,:]-self.pos[:,None,:],p.box)
        dist=np.linalg.norm(delta,axis=2)
        dots=np.einsum('ijk,ik->ij',delta,self.dir)
        visible=dots>=dist*np.cos(np.deg2rad(p.fov_deg/2))-1e-12
        return np.minimum(((dist>0)&(dist<=p.dilution_radius)&visible).sum(axis=1),p.vision_k)

    def tangent(self, direction, blocked):
        tangent = direction.copy()
        tangent[blocked] = 0
        zero = np.linalg.norm(tangent, axis=1) < 1e-12
        fallback = self.dir.copy()
        fallback[blocked] = 0
        tangent[zero] = fallback[zero]
        zero = np.linalg.norm(tangent, axis=1) < 1e-12
        # Exactly vertical degenerate headings have no horizontal preference.
        # A fixed +x fallback avoids introducing imaginary side-wall clearance.
        tangent[zero] = [1., 0., 0.]
        return _row_normalize(tangent)

    def steer_boundary(self, desired):
        """Smooth tangent blend plus a turn-radius look-ahead, both cap-limited."""
        p = self.p
        box = np.asarray(p.box)
        target = _row_normalize(desired)
        touched = np.zeros(p.N, dtype=bool)
        for axis in self.protected_axes:
            toward_low = target[:, axis] < 0
            distance = np.where(toward_low, self.pos[:, axis], box[axis]-self.pos[:, axis])
            x = np.clip((p.glide_layer-distance)/(p.glide_layer-p.guard_margin), 0, 1)
            weight = x*x*(3-2*x)
            active = (weight > 0) & (np.abs(target[:, axis]) > 1e-12)
            blocked = np.zeros_like(target, dtype=bool)
            blocked[:, axis] = True
            tangent = self.tangent(target, blocked)
            target[active] = ((1-weight[active, None])*target[active]
                              + weight[active, None]*tangent[active])
            target = _row_normalize(target)
            touched |= active
        self.smooth_turns += int(touched.sum())

        heading = _row_normalize(self._cap_turn(target))
        critical = np.zeros_like(heading, dtype=bool)
        radius = p.speed/p.turning_rate
        # R*|u_normal| conservatively bounds the normal travel of a 90-degree
        # turn into the tangent plane; the candidate step is included explicitly.
        # Revisit corners because protecting one face can affect another face.
        for _ in range(3):
            next_pos = self.pos + p.speed*p.dt*heading
            distance = np.where(heading < 0, next_pos, box-next_pos)
            risk = distance < p.guard_margin + radius*np.abs(heading)
            risk &= np.abs(heading) > 1e-12
            if True:  # only z is bounded
                risk[:, :2] = False
            updated = critical | risk
            if np.array_equal(updated, critical):
                break
            critical = updated
            active = critical.any(axis=1)
            tangent = self.tangent(self.dir, critical)
            new_target = target.copy()
            new_target[active] = tangent[active]
            heading = _row_normalize(self._cap_turn(new_target))
        self.predictive_turns += int(critical.any(axis=1).sum())
        return heading

    def constrain(self, proposed, heading):
        """Exceptional outside starts/steps: clip and remove outward components.

        There is no sign reversal on protected axes. All corrections are counted.
        The vertical-only diagnostic keeps legacy side reflections explicitly.
        """
        wrapped = proposed.copy()
        lengths = np.asarray(self.p.box)[:2]
        self.wraps += np.abs(np.floor(wrapped[:, :2]/lengths)).astype(int).sum(axis=0)
        wrapped[:, :2] %= lengths
        proposed = wrapped
        box = np.asarray(self.p.box)
        low, high = proposed < 0, proposed > box
        hit = low | high
        self.floor_hits += int(low[:, 2].sum())
        self.surface_hits += int(high[:, 2].sum())
        self.corrected_axes += hit.sum(axis=0)
        result = proposed.copy()
        result_dir = heading.copy()
        protected = np.zeros_like(hit)
        protected[:, self.protected_axes] = True
        protected &= hit
        result = np.where(protected, np.clip(result, 0, box), result)
        result_dir[protected] = 0
        remaining = hit & ~protected
        result = np.where(remaining & low, -result, result)
        result = np.where(remaining & high, 2*box-result, result)
        result_dir[remaining] *= -1
        self.reflections += remaining.sum(axis=0)
        zero = np.linalg.norm(result_dir, axis=1) < 1e-12
        if zero.any():
            fallback = self.tangent(heading, protected)
            result_dir[zero] = fallback[zero]
        return result, _row_normalize(result_dir), hit.any(axis=1)

    def reaction_target(self):
        """Eqs. need--noise-scale; refresh targets on the reaction clock."""
        p=self.p
        if self.total_steps%self.reaction_stride==0:
            desired,in_rep=self.social_direction()
            self.cached_social=desired.copy()
            n=self.nearby_count() if p.dilution else np.zeros(p.N)
            need=np.where(self.reserves<p.need_setpoint,p.upward_strength,p.need_down)*(p.need_setpoint-self.reserves)
            push=-p.surface_strength*np.clip(1-(p.box[2]-self.pos[:,2])/p.surface_layer,0,1)
            factor=p.dilution_min+(1-p.dilution_min)/(1+n/p.dilution_scale) if p.dilution else 1.
            push=push*factor
            if p.kappa_bottom == 0:
                vertical=need+push
            else:
                kappa=p.kappa_bottom*np.clip((p.rest_height-self.pos[:,2])/p.rest_height,0,1)
                vertical=need+push-kappa*self.dir[:,2]
            desired[~in_rep,2]+=vertical[~in_rep]
            desired=_row_normalize(desired)
            noise=self.rng.normal(0,p.eta*np.sqrt(p.reaction_interval/.1),(p.N,3))
            noise[:,2]*=p.vertical_noise_scale
            desired+=noise
            self.cached_desired=desired
            self.reaction_count+=1
            if self.t>=self.audit_burn:
                self.neighbour_sum+=float(n.sum());self.neighbour_observations+=p.N
                self.noise_sum+=noise.sum(axis=0);self.noise_squares+=(noise*noise).sum(axis=0);self.noise_n+=p.N
        return self.cached_desired.copy(),self.cached_social.copy()

    def transition(self,mask,to_state,when,reason,before=None):
        """Fermi hazard, passive band/entry logs, state events and breath starts."""
        original_reason=reason
        start=len(self.events)
        if reason=='proximity':
            p=self.p
            assert to_state==1 and np.all(self.status[mask]==0)
            ids=np.flatnonzero(mask)
            count,visible=school_band_counts(self.pos,self.dir,self.status,p,ids)
            prob=decision_probability(self.reserves[ids],p.dt,p.fermi_midpoint,p.fermi_temperature,p.decision_timescale)
            rows=np.empty(len(ids),dtype=BAND_DTYPE)
            for key,val in [('time',when),('agent',ids),('visible',visible),('p_tank',prob),('p_social',prob)]:rows[key]=val
            for j in range(5):rows[f'n{j}']=count[:,j]
            if len(ids):self.band_batches.append(rows)
            band=self.pos[:,2]>=p.box[2]-p.trigger_depth-1e-12
            self.entry_armed[~band]=True
            new=np.flatnonzero(mask&self.entry_armed)
            self.entry_armed[new]=False;self.entry_id[new]+=1
            for i in new:self.entry_log.append((when,i,self.entry_id[i],self.pos[i,2],self.reserves[i],np.nan,np.nan,False))
            # N draws EVERY step, including when there are no eligible fish.
            draws=self.decision_rng.random(p.N)
            if len(ids):
                accepted=draws[ids]<prob
                rows=np.empty(len(ids),dtype=TRIAL_DTYPE)
                for key,val in [('time',when),('agent',ids),('z',self.pos[ids,2]),('tank',self.reserves[ids]),('probability',prob),('draw',draws[ids]),('accepted',accepted)]:rows[key]=val
                self.trial_batches.append(rows)
                mask=mask.copy();mask[ids]=accepted
            reason='fermi_decision'
        for i in np.flatnonzero(mask):
            self.events.append((when,int(i),int(self.status[i]),to_state,
                float(self.pos[i,2]),float(self.reserves[i] if before is None else before[i]),
                float(self.reserves[i]),reason))
        self.status[mask]=to_state
        if to_state==1:
            for event in self.events[start:]:
                i=event[1]
                self.cycle[i]=dict(depart=when,release=np.nan,rx=np.nan,ry=np.nan,
                    contact=np.nan,cx=np.nan,cy=np.nan,q0=float(self.reserves[i]),reason=original_reason)
                self.departure_log.append((when,i,self.pos[i,2],self.reserves[i],original_reason))
                if self.pos[i,2]>=self.p.box[2]-self.p.release_depth:
                    self.cycle[i].update(release=when,rx=self.pos[i,0],ry=self.pos[i,1])

    def _changed_step(self):
        p=self.p;t0=self.total_steps*p.dt;t1=(self.total_steps+1)*p.dt
        if t0>=self.audit_burn-1e-12:
            self.empty_postburn_agent_seconds+=float(((self.reserves<=1e-12)&(self.status!=2)).sum())*p.dt
        # Only cues existing before this step are seen. Recruits can relay next
        # step; there is no same-step recursive cascade or global event manager.
        if True:  # breathing is part of RZ75
            enter=(self.status==0)&(self.pos[:,2]>=p.box[2]-p.trigger_depth-1e-12)
            self.transition(enter,1,t0,'proximity')
        state=self.status.copy();before_q=self.reserves.copy();before_pos=self.pos.copy();before_heading=self.dir.copy()
        self.state_agent_seconds+=np.bincount(state,minlength=5)*p.dt
        self.empty_agent_seconds+=float(((before_q<=1e-12)&(state!=2)).sum())*p.dt
        # Hold the voluntary target between reaction ticks. Boundary safety,
        # turn caps, motor states, reserve depletion and Fermi hazards use dt.
        desired,social=self.reaction_target()
        heading=self.steer_boundary(desired)
        motor_target=self.dir.copy();motor_target[state==1]=[0.,0.,1.]
        motor_target[(state==3)|(state==4)]=[0.,0.,-1.]
        motor=(state==1)|(state==3)|(state==4)
        capped=self._cap_turn(motor_target)
        if p.return_turning_rate!=p.turning_rate and np.any(state==3):
            capped[state==3]=self.cap_at(motor_target,p.return_turning_rate)[state==3]
        heading[motor]=capped[motor];heading[state==2]=self.dir[state==2]
        moving=(state==0)|(state==1)|(state==4)
        speed=np.where(state==1,p.ascent_speed,p.speed)
        speed[state==4]=p.descent_speed
        # A newly recruited fish may still point down while turning up. Limit
        # its forward speed at the floor rather than clipping position/heading.
        downward_motor=motor & (heading[:,2]<0)
        safe_speed=np.divide(np.maximum(self.pos[:,2]-p.guard_margin,0.),
            -p.dt*heading[:,2],out=np.full(p.N,np.inf),where=heading[:,2]<0)
        limited=downward_motor & (speed>safe_speed)
        speed[limited]=safe_speed[limited]
        self.floor_limited_agent_seconds+=float(limited.sum())*p.dt
        if t0>=self.audit_burn-1e-12:self.floor_limited_postburn_seconds+=float(limited.sum())*p.dt
        proposed=self.pos+speed[:,None]*p.dt*heading*moving[:,None]
        arrival=(state==1)&(proposed[:,2]>=p.box[2]-1e-12)
        proposed[arrival,2]=p.box[2]
        self.pos,self.dir,corrected=self.constrain(proposed,heading)
        drain=np.minimum(self.reserves,self.agent_consumption_rates*p.dt)*(state!=2)
        self.reserves-=drain;self.consumed+=drain;refill=np.zeros(p.N)
        if arrival.any():
            pre=self.reserves.copy()
            refill[arrival]=np.maximum(p.refill_level-self.reserves[arrival],0.)
            self.reserves[arrival]+=refill[arrival];self.refilled+=refill
            self.arrival_time[arrival]=t1;self.breath_counts[arrival]+=1
            self.transition(arrival,2,t1,'surface_contact',before=pre)
        self.transition((state==2)&(t1-self.arrival_time>=p.breath_time-1e-12),3,t1,'dwell_complete')
        self.transition((state==3)&(self.dir[:,2]<-.05),4,t1,'downward_heading')
        self.transition((state==4)&(self.pos[:,2]<=p.box[2]-p.trigger_depth-p.return_clearance+1e-12),0,t1,'cleared_surface_band')
        self.max_budget_error=max(self.max_budget_error,float(np.max(np.abs(before_q-drain+refill-self.reserves))))
        angle=np.arccos(np.clip(np.einsum('ij,ij->i',before_heading,self.dir),-1,1))
        allowed=np.where(state==3,p.return_turning_rate,p.turning_rate)*p.dt
        self.turn_cap_excess+=int((angle>allowed+1e-8).sum())
        pitch=np.abs(np.arcsin(np.clip(self.dir[:,2],-1,1))-np.arcsin(np.clip(before_heading[:,2],-1,1)))
        self.pitch_cap_excess+=int((pitch>allowed+1e-8).sum())
        self.max_pitch_step_deg=max(self.max_pitch_step_deg,float(np.degrees(pitch.max())))
        self.max_step_turn_deg=max(self.max_step_turn_deg,float(np.degrees(angle.max())))
        if (~corrected).any():self.max_ordinary_turn_deg=max(self.max_ordinary_turn_deg,float(np.degrees(angle[~corrected].max())))
        free=moving&~arrival&~corrected
        if free.any():
            displacement=minimum_image(self.pos-before_pos,p.box)
            error=np.linalg.norm(displacement[free]-speed[free,None]*p.dt*self.dir[free],axis=1)
            self.max_free_displacement_error=max(self.max_free_displacement_error,float(error.max()))
        self.total_steps+=1;self.t=t1

    def step(self):
        before_pos=self.pos.copy();before_dir=self.dir.copy();before_state=self.status.copy()
        previous_hits=int(self.corrected_axes.sum());start=len(self.events)
        self._changed_step()
        p=self.p;t0=self.t-p.dt
        # Passive logs: no RNG calls, no writes to the inherited state.
        for e in self.events[start:]:
            if e[3]==2:
                i=e[1];c=self.cycle[i]
                if c is not None:c.update(contact=e[0],cx=self.pos[i,0],cy=self.pos[i,1])
        for i in range(p.N):
            c=self.cycle[i]
            if c is None:continue
            if not np.isfinite(c['release']) and self.status[i] in (1,2):
                level=p.box[2]-p.release_depth
                if before_pos[i,2]<level<=self.pos[i,2]:
                    f=(level-before_pos[i,2])/(self.pos[i,2]-before_pos[i,2])
                    xy=(before_pos[i]+f*minimum_image(self.pos[i]-before_pos[i],p.box))%np.array(p.box)
                    c.update(release=t0+f*p.dt,rx=xy[0],ry=xy[1])
            if np.isfinite(c['contact']) and self.pos[i,2]<before_pos[i,2]-1e-12 and before_state[i]==4:
                assert np.isfinite(c['release'])
                self.episodes.append((c['release'],self.t,i,c['rx'],c['ry'],c['contact'],c['cx'],c['cy'],c['depart'],c['q0'],c['reason']))
                self.cycle[i]=None
        angle=np.arccos(np.clip(np.einsum('ij,ij->i',before_dir,self.dir),-1,1))
        allowed=np.where(before_state==3,p.return_turning_rate,p.turning_rate)*p.dt
        excess=np.maximum(angle-allowed,0.)
        self.max_state_turn_excess=max(self.max_state_turn_excess,float(excess.max()))
        if t0>=self.audit_burn-1e-10:
            self.postburn_state_turn_violations+=int((excess>1e-8).sum())
            self.postburn_corrections+=int(self.corrected_axes.sum())-previous_hits

    def run(self,seconds=600.,sample_dt=.1):
        p=self.p
        steps,stride=round(seconds/p.dt),round(sample_dt/p.dt)
        if steps<=0 or stride<1 or not np.isclose(steps*p.dt,seconds) or not np.isclose(stride*p.dt,sample_dt):
            raise ValueError('duration and sample cadence must be positive multiples of dt')
        fields=('time','pos','heading','tank','status','breath_counts','consumed','refilled',
                'cumulative_hits','cumulative_diagnostics','cumulative_wraps')
        d={k:[] for k in fields}
        for step in range(steps+1):
            if step%stride==0:
                d['time'].append(self.t)
                for key,arr in [('pos',self.pos),('heading',self.dir),('tank',self.reserves),
                                ('status',self.status),('breath_counts',self.breath_counts),
                                ('consumed',self.consumed),('refilled',self.refilled)]:d[key].append(arr.copy())
                d['cumulative_hits'].append([self.floor_hits,self.surface_hits,*self.corrected_axes])
                d['cumulative_diagnostics'].append([self.smooth_turns,self.predictive_turns,self.turn_cap_excess,
                                                  *self.reflections,self.pitch_cap_excess])
                d['cumulative_wraps'].append(self.wraps.copy())
            if step<steps:self.step()
        d={k:np.asarray(v) for k,v in d.items()}
        dtype=[('time','f8'),('agent','i4'),('from_state','i1'),('to_state','i1'),
               ('z','f8'),('tank_before','f8'),('tank_after','f8'),('reason','U24')]
        d['events']=np.array(self.events,dtype=dtype)
        d['state_agent_seconds']=self.state_agent_seconds.copy()
        d['empty_agent_seconds']=np.array(self.empty_agent_seconds)
        d['type_id']=self.type_id.copy()
        d['agent_consumption_rates']=self.agent_consumption_rates.copy()
        d['decision_trials'] = (np.concatenate(self.trial_batches) if self.trial_batches
                                  else np.empty(0, dtype=TRIAL_DTYPE))
        d['empty_postburn_agent_seconds'] = np.array(self.empty_postburn_agent_seconds)
        d['audit_burn'] = np.array(self.audit_burn)
        d['episodes13']=np.array(self.episodes,dtype=[('release','f8'),('exit','f8'),('agent','i4'),('release_x','f8'),('release_y','f8'),('contact','f8'),('contact_x','f8'),('contact_y','f8'),('depart','f8'),('tank_depart','f8'),('reason','U24')])
        d['social_trials']=np.array([],dtype=[('time','f8'),('agent','i4'),('z','f8'),('tank','f8'),('probability','f8'),('draw','f8'),('accepted','?'),('reach','f8')])
        d['entries21']=np.array(self.entry_log,dtype=[('time','f8'),('agent','i4'),('entry','i4'),
            ('z','f8'),('tank','f8'),('probability','f8'),('draw','f8'),('accepted','?')])
        d['band22']=np.concatenate(self.band_batches) if self.band_batches else np.empty(0,dtype=BAND_DTYPE)
        return d

Sim25 = RZ75Sim  # convenient migration alias; no inherited implementation
