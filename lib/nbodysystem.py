import sys
sys.path.insert(0, '..')  # allow import from parent dir
import lib.aochelper as aoc

import logging
logging.basicConfig(stream=sys.stdout, level=logging.WARN)
log = logging.getLogger(__name__)
log.setLevel(logging.WARN)

import itertools
import re

class Body:
  def __init__(self, pos=None, vel=None):
    self.pos = pos
    self.vel = vel
    if pos is None:
      self.pos = [0, 0, 0]
    if vel is None:
      self.vel = [0, 0, 0]

  def __repr__(self):
    return "Body"

  def __str__(self):
    return f"Body[pos={self.pos}, vel={self.vel}]"

  def get_id(self):
    """Get a unique representation of body(posisition,velocity) as str."""
    return ','.join(map(str, self.pos)) + ':' + '.'.join(map(str, self.vel))

  
class NBodySystem:
  def __init__(self, tm=0, bodies=None):
    self.tm = tm
    self.bodies = bodies
    if bodies is None:
      self.bodies = []
    log.info(f"initialised {self}")

  def __repr__(self):
    return "NBodySystem"

  def __str__(self):
    return f"NBodySystem[tm={self.tm}, #bodies={len(self.bodies)}]"  
  
  def get_id(self):
    return '|'.join(map(lambda b: b.get_id(), self.bodies))

  def get_bodies_str(self):
    return "\n".join(map(str, self.bodies))

  def add_body(self, b) -> None:
    self.bodies.append(b)
    log.info(f"  added {b}")

  def iterate_step(self) -> None:
    for b in self.bodies:
      x, y, z = b.pos
      dx, dy, dz = b.vel
      b.pos = [x+dx , y+dy, z+dz]

  def iterate(self, tm_steps=1) -> None:
    for i in range(tm_steps):
      self.iterate_step()

def get_sys_from_str(s: str):
  #log.debug(f"get_sys_from_str called with {s}")
  str_2_sys_re = re.compile(r'<x=(-?\d+), y=(-?\d+), z=(-?\d+)>')
  nbodysystem = NBodySystem()
  for line in s.split("\n"):
    m = str_2_sys_re.match(line)
    if not m:
      raise(RuntimeError(f"unparseable line {line}"))
    body_pos = list(map(int, [m.group(1), m.group(2), m.group(3)]))
    body = Body(pos=body_pos)
    log.debug(f"body={body}")
    nbodysystem.add_body(body)
  log.debug(f"nbodysystem={nbodysystem}")
  return nbodysystem