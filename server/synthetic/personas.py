from __future__ import annotations
import time, math, random
from typing import List, Dict

def _now(): return time.time()

def reader(uid="u_reader", sid_prefix="s_reader", minutes=3, step=1.0) -> List[Dict]:
    """Steady scroll, low jitter, few clicks."""
    t0 = _now(); ev=[]; sid=f"{sid_prefix}_{int(t0)}"
    y=0
    for i in range(int(minutes*60/step)):
        ts = t0 + i*step
        y = (y + random.randint(40,80)) % 8000
        ev += [
            {"sid":sid,"uid":uid,"ts":ts,"ev":"scroll","view":{"y":y}},
            {"sid":sid,"uid":uid,"ts":ts+0.02,"ev":"mousemove","x":600+random.randint(-5,5),"y":400+random.randint(-5,5)}
        ]
        if i%45==10:
            ev.append({"sid":sid,"uid":uid,"ts":ts+0.05,"ev":"click","x":600,"y":400,"el":"a#next"})
    return ev

def skimmer(uid="u_skimmer", sid_prefix="s_skimmer", minutes=3, step=0.7) -> List[Dict]:
    """Fast scroll bursts, shallow depth."""
    t0=_now(); ev=[]; sid=f"{sid_prefix}_{int(t0)}"; y=0
    for i in range(int(minutes*60/step)):
        ts=t0+i*step
        dy=random.randint(120,240)
        y=(y+dy)%4000
        ev += [
            {"sid":sid,"uid":uid,"ts":ts,"ev":"scroll","view":{"y":y}},
            {"sid":sid,"uid":uid,"ts":ts+0.02,"ev":"mousemove","x":500+random.randint(-30,30),"y":350+random.randint(-30,30)}
        ]
        if i%50==5:
            ev.append({"sid":sid,"uid":uid,"ts":ts+0.05,"ev":"click","x":520,"y":360,"el":"button#cta"})
    return ev

def rager(uid="u_rager", sid_prefix="s_rager", minutes=2, step=0.8) -> List[Dict]:
    """Rage-click clusters."""
    t0=_now(); ev=[]; sid=f"{sid_prefix}_{int(t0)}"; y=0
    for i in range(int(minutes*60/step)):
        ts=t0+i*step
        y=(y+random.randint(30,60))%5000
        ev += [{"sid":sid,"uid":uid,"ts":ts,"ev":"scroll","view":{"y":y}}]
        if i%12 in (3,4,5):
            # 3 fast clicks (rage)
            base=ts+0.05
            for j in range(3):
                ev.append({"sid":sid,"uid":uid,"ts":base+0.01*j,"ev":"click","x":300,"y":600,"el":"div#dead"})
        ev.append({"sid":sid,"uid":uid,"ts":ts+0.02,"ev":"mousemove","x":300+random.randint(-80,80),"y":600+random.randint(-80,80)})
    return ev

def form_lost(uid="u_form", sid_prefix="s_form", minutes=3, step=1.2) -> List[Dict]:
    """Hover-stall near CTA, dead clicks on labels."""
    t0=_now(); ev=[]; sid=f"{sid_prefix}_{int(t0)}"; y=0
    for i in range(int(minutes*60/step)):
        ts=t0+i*step
        y=(y+random.randint(40,80))%6000
        ev += [{"sid":sid,"uid":uid,"ts":ts,"ev":"scroll","view":{"y":y}}]
        # slow jitter (hover-stall)
        for k in range(10):
            ev.append({"sid":sid,"uid":uid,"ts":ts+0.03+0.02*k,"ev":"mousemove","x":700+random.randint(-3,3),"y":420+random.randint(-3,3)})
        if i%20==8:
            ev.append({"sid":sid,"uid":uid,"ts":ts+0.4,"ev":"click","x":705,"y":425,"el":"label#name"})
    return ev
