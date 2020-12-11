#!/usr/bin/env python
# coding: utf-8

# # Advent of Code 2018
# 
# This solution (Jupyter notebook; python3.7) by kannix68, @ 2020-12 (2 years late).  \
# Using anaconda distro, conda v4.9.2. installation on MacOS v10.14.6 "Mojave".

# ## Generic AoC code

# In[ ]:


import sys
import logging

import lib.aochelper as aoc
from lib.aochelper import map_list as mapl

print("Python version:", sys.version)
print("Version info:", sys.version_info)

log = aoc.getLogger(__name__)
#log.setLevel(logging.DEBUG) # std: logging.INFO
print(f"initial log-level={log.getEffectiveLevel()}")


# ## Problem domain code

# ### Day 1: Chronal Calibration

# In[ ]:


print("Day 1 a")


# In[ ]:


test = """
+1, -2, +3, +1
""".strip().split(', ')
tests = aoc.map_list(int, test)
aoc.assert_msg("test 1", 3 == sum(tests))


# In[ ]:


ins = aoc.read_file_to_list('./day01/day01.in')
print("Day 1 a solution:", sum(aoc.map_list(int, ins)))


# In[ ]:


print("Day 1 b", "TODO here, but already solved 2018")


# In[ ]:


import itertools


# In[ ]:


# A list and list.append(frequency are resource hog tools to keep track of seen entries),
#  using dict instead.
def solve01b(l):
  iter = 0
  freq = 0
  freqs = {freq:True}
  for freq_inc in itertools.cycle(l):
    iter += 1
    freq += freq_inc
    if (len(freqs)%100_000 == 0):
      log.info(f"iter={iter}, freq={freq} num-frequencies={len(freqs)}")
    if freq in freqs:
      log.info(f"frequency {freq} used 2nd time, iteration={iter}, num-frequencies={len(freqs)}")
      return freq
    elif iter > 10_000_000:
      raise Exception("fail")
    else:
      freqs[freq] = True


# In[ ]:


solve01b(tests)
iins = aoc.map_list(int, ins)
log.debug(f"len={len(iins)},elems={iins}")
result = solve01b(iins)
print("Day 1 b result:", result)


# ### Day 2: Inventory Management System

# In[ ]:


print("Day 2", "TODO here, but already solved 2018")


# ### Day 3: No Matter How You Slice It

# In[ ]:


from collections import defaultdict

class C2d_space_inc:
  
  def __init__(self):
    self.spc = defaultdict(int)
    self.clms = {}
  
  def set_xy(self, x, y):
    self.spc[tuple([x,y])] += 1
  
  def set_range(self, x, y, rgx, rgy, id=None):
    for px in range(x, x+rgx):
      for py in range(y, y+rgy):
        self.spc[tuple([px,py])] += 1
    if id is not None:
      self.clms[id] = [x, y, rgx, rgy]
      #log_info(f"create claim {id} => {self.claims[id] }")
  
  def get_range(self, x, y, rgx, rgy):
    outspc = {}
    for px in range(x, x+rgx):
      for py in range(y, y+rgy):
        outspc[tuple([px,py])] = self.spc[tuple([px,py])]
    return outspc

  def cols(self):
    return sorted(set(map(lambda it: it[0], self.spc.keys())))

  def rows(self):
    return sorted(set(map(lambda it: it[1], self.spc.keys())))
  
  def values(self):
    return self.spc.values()

  def claims(self):
    return self.clms

  def get_pp_repr(self):
    return "[c2d_space_inc]: " + str(self.spc)

  def pp(self):
    print(self.get_pp_repr())

  def get_show_repr(self):
    rows = self.rows()
    cols = self.cols()
    rowstr = ''
    for y in range(0, max(rows)+1): #range(min(rows), max(rows)+1):
      colstr = ''
      for x in range(0, max(cols)+1):
        colstr += str(self.spc[tuple([x,y])])
      rowstr += colstr + "\n"
    return rowstr

  def show(self):
    print(self.get_show_repr())


# In[ ]:


# just some testing:
c2d = C2d_space_inc()
c2d.set_xy(1,1)
c2d.set_xy(2,2)
log.debug(c2d.get_pp_repr())
c2d.set_xy(1,1)
c2d.set_xy(1,3)
log.debug(c2d.get_pp_repr())
log.debug(f"cols: {c2d.cols()}")
log.debug(f"rows:, {c2d.rows()}")
log.debug("\n"+c2d.get_show_repr()) #c2d.show()


# In[ ]:


def create_space(l):
  import re
  c2d = C2d_space_inc()
  for line in l:
    rx = re.search('^#(\w+)\s+@\s+(\d+),(\d+):\s*(\d+)x(\d+)$', line) #r'^#([^\s]+) @ (d+),(\d+): ((\d+)),((\d+))$', line)
    #log.debug(rx.groups())
    id, sx, sy, srgx, srgy = rx.groups()
    x, y, rgs, rgy = aoc.map_list(int, [sx, sy, srgx, srgy])
    #c2d.set_range(x, y, rgs, rgy)
    c2d.set_range(x, y, rgs, rgy, id=id)
  #c2d.show()
  return c2d


# In[ ]:


tests = """
#123 @ 3,2: 5x4
""".strip().split("\n")
log.info(f"tests {tests}")


# In[ ]:


print("1st test representation:")
create_space(tests).show()


# In[ ]:


tests = """
#1 @ 1,3: 4x4
#2 @ 3,1: 4x4
#3 @ 5,5: 2x2
""".strip().split("\n")
log.info(f"tests={tests}")


# In[ ]:


print("2nd test representation:")
create_space(tests).show()


# In[ ]:


def get_space_overlaps(l):
  c2d = create_space(l)
  return len(aoc.filter_list(lambda it: it > 1, c2d.values()))


# In[ ]:


aoc.assert_msg( "4 test cells that overlap", 4 == get_space_overlaps(tests) )


# In[ ]:


ins = aoc.read_file_to_list('./day03/day03.in')
result = get_space_overlaps(ins)
print("Day 3 a solution:", result)


# In[ ]:


def get_singular_space(l):
  c2d = create_space(l)
  for k, v in c2d.claims().items():
    #log.debug([k, v])
    rg = c2d.get_range(v[0], v[1], v[2], v[3])
    #log.debug(rg)
    #log.debug(rg.values())
    if max(rg.values()) == 1:
      log.info(f"found id={k}, num of 1-cells:{len(rg.values())}")
      result = k
  return result


# In[ ]:


print("Day 3 b tests result: ", get_singular_space(tests))
print("Day 3 b solution:", get_singular_space(ins))


# ### Day 4

# In[ ]:


# TODO


# ### Day 5

# In[ ]:


# TODO


# ### Day 6:  Chronal Coordinates

# In[ ]:


#TODO


# ### Day 7: The Sum of Its Parts

# In[ ]:


DEBUG_FLAG = 0


# In[ ]:


import networkx as nx


# In[ ]:


tests = """
Step C must be finished before step A can begin.
Step C must be finished before step F can begin.
Step A must be finished before step B can begin.
Step A must be finished before step D can begin.
Step B must be finished before step E can begin.
Step D must be finished before step E can begin.
Step F must be finished before step E can begin.
""".strip().split("\n")


# In[ ]:


def create_graph(l):
  graph = nx.DiGraph()
  for line in l:
    linesplit = line.split(' ')
    srcnd, trgnd = [linesplit[1], linesplit[-3]]
    if not srcnd in graph.nodes:
      log.debug(f"g add node {srcnd}")
    if not trgnd in graph.nodes:
      log.debug(f"g add node {trgnd}")
    graph.add_edge(srcnd, trgnd)
  log.info(f"graph-edges={sorted(list(graph.edges))}")
  return graph


# In[ ]:


# still using networkx graph li, inspiration from user VikeStep:
#  [- 2018 Day 7 Solutions - : adventofcode](https://www.reddit.com/r/adventofcode/comments/a3wmnl/2018_day_7_solutions/)
def solve07a(l):
  graph = create_graph(l)
  
  nodes_lst = list(graph.nodes)
  log.debug(f"nodes: entry-order {str.join('', nodes_lst)}")
  nodes_lst = list(nx.topological_sort(graph))
  log.debug(f"nodes: topo {str.join('', nodes_lst)}")
  nodes_lst = list(nx.algorithms.dag.lexicographical_topological_sort(graph))
  log.info("nodes: lexico-topo {str.join('', nodes_lst)}")
  return str.join('', nodes_lst)


# In[ ]:


solve07a(tests)


# In[ ]:


ins = aoc.read_file_to_list('./in/day07.in')
solve07a(ins)


# In[ ]:


print("Day 07 b, # TODO")


# ### Day 8

# In[ ]:


# TODO


# ### Day 9

# In[ ]:


# TODO


# ### Day 10

# In[ ]:


# TODO


# ### Day 11

# In[ ]:


# TODO


# ### Day 12

# In[ ]:


# TODO


# ### Day 13

# In[ ]:


# TODO


# In[ ]:


### Day 14: Chocolate Charts


# In[ ]:


def find_improved_score(target_num):
  elf1 = 0
  elf2 = 1
  recipies = [3, 7]
  #num_new_recipies = 0
  log.info(f"#0: num-recipes={len(recipies)}, recipies={recipies}")
  for i in range(2*target_num):
    new_recipy = recipies[elf1] +  recipies[elf2]
    #found_recipies.add(new_recipy)
    digits = aoc.map_list(int, str(new_recipy))
    #num_new_recipies += len(digits)
    for d in digits:
      recipies.append(d)
    elf1 = (elf1+1+recipies[elf1]) % len(recipies)
    elf2 = (elf2+1+recipies[elf2]) % len(recipies)
    #log.debug(f"#{i+1}: num-recipes={len(recipies)}")
    #log.debug(f"{len(digits)}, {digits}, {recipies}, elf1:{elf1} elf2:{elf2}, {len(recipies)}") #found_recipies)
    if len(recipies) >= target_num + 10:
      res = str.join('', aoc.map_list(str, recipies))[target_num:target_num+10] # 0124515891
      log.info(f"found: {res}")
      return res


# In[ ]:


print( find_improved_score(9) )
print( find_improved_score(5) )
print( find_improved_score(18) )
print( find_improved_score(2018) )


# In[ ]:


ins = 846021 # << insert personal input here
print( "Day 14 a solution:", find_improved_score(ins) )


# In[ ]:


print("Day 14 b")

def find_num_recips_before(target_num):
  target_str = str(target_num)
  elf1 = 0
  elf2 = 1
  recipies = [3, 7]
  #num_new_recipies = 0
  log.info(f"#0: num-recipes={len(recipies)}, recipies={recipies}")
  for i in range(1000*int(target_num)):
    new_recipy = recipies[elf1] +  recipies[elf2]
    #found_recipies.add(new_recipy)
    digits = aoc.map_list(int, str(new_recipy))
    #num_new_recipies += len(digits)
    for d in digits:
      recipies.append(d)
    elf1 = (elf1+1+recipies[elf1]) % len(recipies)
    elf2 = (elf2+1+recipies[elf2]) % len(recipies)
    #log_debug(f"#{i+1}: num-recipes={len(recipies)}")
    #log_debug(len(digits), digits, recipies, " elf1:", elf1, " elf2:", elf2, found_recipies)
    if i % 1_000_000 == 0:
      log.info(f"calculating, iter: {i}")
    recips_end_str = str.join('', aoc.map_list(str, recipies[-12:]))
    if target_str in recips_end_str:
      offset = 0
      if not recips_end_str.endswith(target_str):
        recips_end_str = recips_end_str[0:-1]
        offset = 1
      assert( recips_end_str.endswith(target_str) )
      #recips_str = str.join('', lmap(str, recipies))
      #res = recips_str.index(target_str)
      #log.info(f"length={len(recipies)}-{len(target_str)}") #" from {recips_str}")
      res = len(recipies) - len(target_str) - offset
      log.info(f"target-num={target_str}, found: idx={res} @iter={i}") #" from {recips_str}")
      return res
  raise Exception("not terminated with find-criterium")


# In[ ]:


find_num_recips_before(51589)


# In[ ]:


find_num_recips_before('01245')
find_num_recips_before('92510')
find_num_recips_before('59414')


# In[ ]:


PERFORM_RESOURCE_HOGS = False
if PERFORM_RESOURCE_HOGS:
  find_num_recips_before(ins)


# ### Day 15

# In[ ]:


# TODO


# ### Day 16

# In[ ]:


test = """
Before: [3, 2, 1, 1]
9 2 1 2
After:  [3, 2, 2, 1]
""".strip().split("\n")
mem_before = eval(test[0].replace('Before: ', ''))
cpu_instruct = aoc.map_list(int, test[1].split(' '))
mem_after = eval(test[2].replace('After:  ', ''))
log.debug([mem_before, cpu_instruct, mem_after])


# In[ ]:


OPC, A, B, C = [0, 1, 2, 3] # positions

def op_addr(instr, regs):
  """addr (add register) stores into register C the result of adding register A and register B."""
  oregs = regs.copy()
  oregs[instr[C]] = oregs[instr[A]] + oregs[instr[B]]
  return oregs
def op_addi(instr, regs):
  """addi (add immediate) stores into register C the result of adding register A and value B."""
  oregs = regs.copy()
  oregs[instr[C]] = oregs[instr[A]] + instr[B]
  return oregs
def op_mulr(instr, regs):
  """mulr (multiply register) stores into register C the result of multiplying register A and register B."""
  oregs = regs.copy()
  oregs[instr[C]] = oregs[instr[A]] * oregs[instr[B]]
  return oregs
def op_muli(instr, regs):
  """muli (multiply immediate) stores into register C the result of multiplying register A and value B."""
  oregs = regs.copy()
  oregs[instr[C]] = oregs[instr[A]] * instr[B]
  return oregs
def op_banr(instr, regs):
  """banr (bitwise AND register) stores into register C the result of the bitwise AND of register A and register B."""
  oregs = regs.copy()
  oregs[instr[C]] = oregs[instr[A]] & oregs[instr[B]]
  return oregs
def op_bani(instr, regs):
  """bani (bitwise AND immediate) stores into register C the result of the bitwise AND of register A and value B."""
  oregs = regs.copy()
  oregs[instr[C]] = oregs[instr[A]] & instr[B]
  return oregs
def op_borr(instr, regs):
  """borr (bitwise OR register) stores into register C the result of the bitwise OR of register A and register B."""
  oregs = regs.copy()
  oregs[instr[C]] = oregs[instr[A]] | oregs[instr[B]]
  return oregs
def op_bori(instr, regs):
  """bori (bitwise OR immediate) stores into register C the result of the bitwise OR of register A and value B."""
  oregs = regs.copy()
  oregs[instr[C]] = oregs[instr[A]] | instr[B]
  return oregs
def op_setr(instr, regs):
  """setr (set register) copies the contents of register A into register C. (Input B is ignored.)"""
  oregs = regs.copy()
  oregs[instr[C]] = oregs[instr[A]]
  return oregs
def op_seti(instr, regs):
  """seti (set immediate) stores value A into register C. (Input B is ignored.)"""
  oregs = regs.copy()
  oregs[instr[C]] = instr[A]
  return oregs
def op_gtir(instr, regs):
  """gtir (greater-than immediate/register) sets register C to 1 if value A is greater than register B.
  Otherwise, register C is set to 0."""
  oregs = regs.copy()
  if instr[A] > oregs[instr[B]]:
    oregs[instr[C]] = 1
  else:
    oregs[instr[C]] = 0
  return oregs
def op_gtri(instr, regs):
  """gtri (greater-than register/immediate) sets register C to 1 if register A is greater than value B.
  Otherwise, register C is set to 0."""
  oregs = regs.copy()
  if oregs[instr[A]] > instr[B]:
    oregs[instr[C]] = 1
  else:
    oregs[instr[C]] = 0
  return oregs
def op_gtrr(instr, regs):
  """gtrr (greater-than register/register) sets register C to 1 if register A is greater than register B.
  Otherwise, register C is set to 0."""
  oregs = regs.copy()
  if oregs[instr[A]] > oregs[instr[B]]:
    oregs[instr[C]] = 1
  else:
    oregs[instr[C]] = 0
  return oregs
def op_eqir(instr, regs):
  """eqir (equal immediate/register) sets register C to 1 if value A is equal to register B.
  Otherwise, register C is set to 0."""
  oregs = regs.copy()
  if instr[A] == oregs[instr[B]]:
    oregs[instr[C]] = 1
  else:
    oregs[instr[C]] = 0
  return oregs
def op_eqri(instr, regs):
  """eqri (equal register/immediate) sets register C to 1 if register A is equal to value B.
  Otherwise, register C is set to 0."""
  oregs = regs.copy()
  if oregs[instr[A]] == instr[B]:
    oregs[instr[C]] = 1
  else:
    oregs[instr[C]] = 0
  return oregs
def op_eqrr(instr, regs):
  """eqrr (equal register/register) sets register C to 1 if register A is equal to register B.
  Otherwise, register C is set to 0."""
  oregs = regs.copy()
  if oregs[instr[A]] == oregs[instr[B]]:
    oregs[instr[C]] = 1
  else:
    oregs[instr[C]] = 0
  return oregs

opcodes = [op_addr, op_addi, op_mulr, op_muli, op_banr, op_bani, op_borr, op_bori
           , op_setr, op_seti, op_gtir, op_gtri, op_gtrr, op_eqir, op_eqri, op_eqrr]  


# In[ ]:


def pp_opfun(opfun):
  return str(opfun).replace('<function ', '').split(' ')[0]

def get_ops_match_count(mem_before, cpu_instruct, mem_after):
  ct = 0
  for opfun in opcodes:
    tst_after = opfun(cpu_instruct, mem_before)
    #print(opfun, tst_after)
    if tuple(tst_after) == tuple(mem_after):
      log.trace(f"opcode {cpu_instruct[OPC]} matches {pp_opfun(opfun)}")
      ct += 1 #print("  matches!")
  return ct
  


# In[ ]:


#mem_before = eval(test[0].replace('Before: ', ''))
#cpu_instruct = lmap(int, test[1].split(' '))
#mem_after = eval(test[2].replace('After:  ', ''))
log.info(f"mem-before={mem_before}")
log.info(f"mem-after-expected={mem_after}")
res = get_ops_match_count(mem_before, cpu_instruct, mem_after)
print("test-result:", res)


# In[ ]:


ins = aoc.read_file_to_str('./in/day16.in')
ins1 = ins.split("\n\n\n")[0].strip().split("\n\n")
log.trace(ins1)


# In[ ]:


ct = 0
for in1 in ins1:
  test = in1.split("\n")
  mem_before = eval(test[0].replace('Before: ', ''))
  cpu_instruct = aoc.map_list(int, test[1].split(' '))
  mem_after = eval(test[2].replace('After:  ', ''))
  opscount = get_ops_match_count(mem_before, cpu_instruct, mem_after)
  if opscount >= 3:
    log.debug(["found-test", opscount, mem_before, cpu_instruct, mem_after])
    ct += 1
log.info(f"Day 16 solution: found fulfilling samples: {ct}")


# In[ ]:


def pp_opfun(opfun):
  return str(opfun).replace('<function ', '').split(' ')[0]

def get_ops_matching(mem_before, cpu_instruct, mem_after):
  maybe_ops = set()
  ct = 0
  for opfun in opcodes:
    tst_after = opfun(cpu_instruct, mem_before)
    #print(opfun, tst_after)
    if tuple(tst_after) == tuple(mem_after):
      #log_debug(f"opcode {cpu_instruct[OPC]} matches {pp_opfun(opfun)}")
      ct += 1 #print("  matches!")
      maybe_ops.add(opfun)
  return maybe_ops
  


# In[ ]:


test = """
Before: [3, 2, 1, 1]
9 2 1 2
After:  [3, 2, 2, 1]
""".strip().split("\n")
mem_before = eval(test[0].replace('Before: ', ''))
cpu_instruct = aoc.map_list(int, test[1].split(' '))
mem_after = eval(test[2].replace('After:  ', ''))
op_num = cpu_instruct[OPC]
log.debug([mem_before, cpu_instruct, mem_after])
res = get_ops_matching(mem_before, cpu_instruct, mem_after)
ops_dict = dict()
if op_num in ops_dict:
  True
else:
  ops_dict[op_num] = res
print("test-result:", res)


# In[ ]:


def find_op_relations(samples):
  log.info("[find_op_relations]")
  #log.debug(samples)
  
  ops_dict = dict()
  for idx, sample in enumerate(samples):
    if False and idx > 10:
      log.warn("BREAK HACK")
      break
    test = sample.split("\n")
    mem_before = eval(test[0].replace('Before: ', ''))
    cpu_instruct = aoc.map_list(int, test[1].split(' '))
    mem_after = eval(test[2].replace('After:  ', ''))
    op_num = cpu_instruct[OPC]
    
    res = get_ops_matching(mem_before, cpu_instruct, mem_after)
    if op_num in ops_dict:
      ops_dict[op_num] = ops_dict[op_num].intersection(res)
    else:
      ops_dict[op_num] = res

  opcodes = {}
  ict = 0
  found_codes = True
  while(found_codes and ict < 20):
    ict += 1
    log.debug(f"resolve-iter {ict}")
    log.debug("ops-dict {ops_dict}")
    found_codes = False
    for k, v in ops_dict.items():
      if v is not None and len(v) == 1:
        opcodes[k] = list(v)[0]
        log.debug(f"found code {k}: {opcodes[k]}")
        found_codes = True
    for k in opcodes.keys():
      if k in ops_dict:
        log.debug(f"removing {k} from maybes")
        del ops_dict[k]
    for k in ops_dict.keys():
      for v in opcodes.values():
        if v in ops_dict[k]:
          log.debug(f"  removing elem {v} from maybe:{k}")
          ops_dict[k].remove(v)
    log.debug("opcodes!:: {opcodes}")
    if len(ops_dict.keys()) == 0:
      log.info(f"resolved all {len(opcodes.keys())} opcodes")
      break
    
  #return ops_dict
  return opcodes


# In[ ]:


test = """
Before: [3, 2, 1, 1]
9 2 1 2
After:  [3, 2, 2, 1]
""".strip()
find_op_relations([test])


# In[ ]:


ins1 = ins.split("\n\n\n")[0].strip().split("\n\n")
ins2 = ins.split("\n\n\n")[1].strip().split("\n")
#log_debug(ins1)


# In[ ]:


op_funs = find_op_relations(ins1)


# In[ ]:


#log_info(ins2)
mem_before = [0, 0, 0, 0]
memnext = mem_before.copy()
for idx, line in enumerate(ins2):
  instruct = aoc.map_list(int, line.split(' '))
  opfun = op_funs[instruct[OPC]]
  memnext = opfun(instruct, memnext)
  log.debug(f"#{idx} memnext={memnext} after {pp_opfun(opfun)} instruct={instruct}")
res = [idx, memnext]
print(f"Day 16 part b solution:, rrun-idx={idx}, mem-regs={memnext}, mem:0={memnext[0]}")


# ### Day 17

# In[ ]:


# TODO


# ### Day 18: Settlers of The North Pole

# In[ ]:


tests = """
.#.#...|#.
.....#|##|
.|..|...#.
..|#.....#
#.#|||#|#|
...#.||...
.|....|...
||...#|.#|
|.||||..|.
...#.|..|.
""".strip().split("\n")
tests2 = []
for t in tests:
  tests2.append(list(t))


# In[ ]:


import copy # for deepcopy

class CellularWorld:
  def __init__(self, world):
    """World object constructor, world has to be given as a list-of-lists of chars."""
    self.world = world
    self.dim = [len(world[0]), len(world)]
    log.info(f'[CellularWorld] new dim={self.dim}')
    self.world = world
  
  def repr(self):
    """Return representation str (can be used for printing)."""
    l = []
    for line in self.world:
      l.append( str.join('', line) )
    return str.join("\n", l)
  
  def set_world(self, world):
    self.world = world
    self.dim = [len(world[0]), len(world)]

  def get_neighbors8(self, x, y):
    """Get cell's surrounding 8 neighbors, omitting boundaries."""
    log.trace(f"[CellularWorld]:get_neighbors8({x},{y})")
    dim_x = self.dim[0]
    dim_y = self.dim[1]
    neighbors = ''
    for nx in range(x-1, x+2):
      for ny in range(y-1, y+2):
        if (nx >= 0 and nx < dim_x) and (ny >= 0 and ny < dim_y) and not (nx == x and ny == y):
          #log.info(f"  neighb={[nx, ny]}")
          neighbors += self.world[ny][nx]
    return neighbors
  
  def iterate(self, n=1):
    for i in range(n):
      world2 = copy.deepcopy(self.world)
      for y in range(self.dim[1]):
        for x in range(self.dim[0]):
          val = self.world[y][x]
          neighbors = self.get_neighbors8(x, y)
          #log.trace(f"[{x},{y}]='{val}' nbs='{neighbors}'")
          if val == '.' and neighbors.count('|') >= 3:
            world2[y][x] = '|'
          elif val == '|' and neighbors.count('#') >= 3:
            world2[y][x] = '#'
          elif val == '#':
            if neighbors.count('#') >= 1 and neighbors.count('|') >= 1:
              world2[y][x] = '#'
            else:
              world2[y][x] = '.'
      self.set_world(world2)

  def find_cycle(self, max_iter=1_000):
    """This may only be called at initial state, before any iterations."""
    seen = [world.repr]
    for i in range(max_iter):
      if i % 1_000 == 0:
        log.debug(f"iter# {i}, still running")
      world.iterate()
      world_repr = world.repr()
      if world_repr in seen:
        start_idx = seen.index(world_repr)
        log.info(f"found cycle @ iter={i+1}, seen-idx={start_idx}")
        return([start_idx, i+1])
      else:
        seen.append(world_repr)
    raise Exception("no world iter cycle found")


# In[ ]:


#t = CellularWorld([[]])
world = CellularWorld(tests2)
log.info(f"world created: dim={world.dim}")
log.info(f"\n{world.repr()}\n")
for i in range(1,11):
  world.iterate()
log.info(f"world iter {i}:")
log.info(f"\n{world.repr()}\n")
world_repr = world.repr()
num_trees = world_repr.count('|')
num_lumberyds = world_repr.count('#')
log.info(f"test num-trees={num_trees}, num-lumberyds={num_lumberyds}, result={num_trees*num_lumberyds}")


# In[ ]:


ins = aoc.map_list(list, aoc.read_file_to_list('./in/day18.in'))
world = CellularWorld(ins)
log.info(f"world created: dim={world.dim}")
log.debug(f"\n{world.repr()}\n")
for i in range(1,11):
  world.iterate()
log.info(f"world iter {i}:")
log.debug(f"\n{world.repr()}\n")
world_repr = world.repr()
num_trees = world_repr.count('|')
num_lumberyds = world_repr.count('#')
day18a_solution = num_trees*num_lumberyds
log.info(f"Day 18 a solution: result={day18a_solution} from num-trees={num_trees}, num-lumberyds={num_lumberyds}")


# In[ ]:


ins = aoc.map_list(list, aoc.read_file_to_list('./in/day18.in'))
world = CellularWorld(ins)
world_cycle = world.find_cycle()
cycle_len = world_cycle[1] - world_cycle[0]
log.info(f"cycle-len={cycle_len} from cycle={world_cycle}")

world.set_world(ins)
scores = {}
for i in range(0, world_cycle[1]+3+28):
  world_repr = world.repr()
  num_trees = world_repr.count('|')
  num_lumberyds = world_repr.count('#')
  score = num_trees * num_lumberyds
  scores[i] = score
  world.iterate()
log.info("finished scoring")
log.debug(scores)


# In[ ]:


def get_cycled_index(idx, cycle_lst):
  cycle_start, cycle_end_bound = cycle_lst
  cycle_len = cycle_end_bound - cycle_start 
  cycle_end = cycle_end_bound-1
  #log.debug(f"[get_cycled_index] cyle is {cycle_lst}, cycle ends-including @{cycle_end}, cycle-len={cycle_len}")
  if idx <= cycle_end:
    return idx
  else:
    #log.trace(idx-cycle_end_bound)
    return cycle_start + ((idx-cycle_end_bound) % cycle_len )

aoc.assert_msg( "day 18 b, solution of 1st part ok", day18a_solution == scores[10])

#557 => 557, 558 => 530, 559 => 531
for v in [529, 530, 531, 556, 557, 558, 559, 560
          , 556+cycle_len, 557+cycle_len, 558+cycle_len, 559+cycle_len]:
  log.debug(f"in idx={v} out idx={get_cycled_index(v, world_cycle)}")

tgt_iter = 1_000_000_000 # 1000000000
idx = get_cycled_index(tgt_iter, world_cycle)
log.info(f"Day 18 b solution: score={scores[idx]} from tgt_iter={tgt_iter:9,.0f} target-index={idx}")


# ### Day 19: Go With The Flow

# In[ ]:


import time
from datetime import datetime

class CPU:
  OPC, A, B, C = [0, 1, 2, 3] # positions
  
  def __init__(self):
    self.flow_mode = -1
    self.prog = []
    self.mem = [0, 0, 0, 0, 0, 0]
    self.iptr = 0
    self.ict = 0
    self.state = 'INITED'
    self.opfuns = {
      'addr':self.opc_addr, 'addi':self.opc_addi,
      'mulr':self.opc_mulr, 'muli':self.opc_muli,
      'banr':self.opc_banr, 'bani':self.opc_bani,
      'borr':self.opc_borr, 'bori':self.opc_bori,
      'setr':self.opc_setr, 'seti':self.opc_seti,
      'gtir':self.opc_gtir, 'gtri':self.opc_gtri, 'gtrr':self.opc_gtrr,
      'eqir':self.opc_eqir, 'eqri':self.opc_eqri, 'eqrr':self.opc_eqrr,
    }
  
  def prepare(self, los):
    idx = 0
    if los[idx].startswith("#ip"):
      log.info(f"program-mode {los[0]}")
      self.flow_mode = int(los[0].split(" ")[1])
      idx += 1
    #cpu_instruct = aoc.map_list(int, test[1].split(' ')
    for i in range(idx, len(los)):
      cells = los[i].split(" ")
      self.prog.append([cells[0]] + mapl(int, cells[1:]))
    log.info(f"CPU prod={self.prog}")
  
  def interpret(self, steps=10_000_000_001):
    start_tm = int(time.time())
    log.info(self.mem)
    self.state = 'RUNNING'
    tm = f"{int(time.time()-start_tm)}s @" + datetime.now().strftime("%H:%M:%S")
    log.info(f"cpu prog interpret started ict#{self.ict:,}, in-mem={self.mem}")
    for i in range(1, steps+1):
      self.ict += 1
      instruct = self.prog[self.iptr]
      #log.debug(f"ict#{self.ict} iptr={self.iptr} instr={instruct} mem=[self.mem]")
      if self.flow_mode >= 0:
        self.mem[self.flow_mode] = self.iptr
      #op = f"self.op_{instruct[CPU.OPC]}({instruct}, {self.mem})"
      self.opfuns[instruct[CPU.OPC]](instruct)
      #log.debug(f"op={op} returns {self.mem}")
      if self.flow_mode >= 0:
        self.iptr = self.mem[self.flow_mode]
      self.iptr += 1
      if self.iptr >= len(self.prog):
        tm = f"{int(time.time()-start_tm)}s @" + datetime.now().strftime("%H:%M:%S")
        self.state = 'TERMINATED'
        log.info(f"cpu prog terminated gracefully! {tm}" )
        log.info(f"  ict#{self.ict:,} curstep#{i:,} iptr={self.iptr} mem={self.mem}")
        return
      if self.ict % 10_000_000 == 0:
        tm = f"{int(time.time()-start_tm)}s @" + datetime.now().strftime("%H:%M:%S")
        log.info(f"cpu-prog running {int(time.time()-start_tm)}s, ict={self.ict:,} iptr={self.iptr} mem={self.mem}")
        if self.ict > 100_000_000_000:
          raise Exception("FAILSAFE")
    tm = f"{int(time.time()-start_tm)}s @" + datetime.now().strftime("%H:%M:%S")
    self.state = 'PAUSED'
    log.info(f"cpu prog interpret PAUSED ict#{self.ict:,}, end of interpret(), curstep#{i:,}! {tm}")

  def opc_addr(self, instr):
    """addr (add register) stores into register C the result of adding register A and register B."""
    self.mem[instr[3]] = self.mem[instr[1]] + self.mem[instr[2]]
  def opc_addi(self, instr):
    """addi (add immediate) stores into register C the result of adding register A and value B."""
    self.mem[instr[3]] = self.mem[instr[1]] + instr[2]
  def opc_mulr(self, instr):
    """mulr (multiply register) stores into register C the result of multiplying register A and register B."""
    self.mem[instr[3]] = self.mem[instr[1]] * self.mem[instr[2]]
  def opc_muli(self, instr):
    """muli (multiply immediate) stores into register C the result of multiplying register A and value B."""
    self.mem[instr[3]] = self.mem[instr[1]] * instr[2]
  def opc_banr(self, instr):
    """banr (bitwise AND register) stores into register C the result of the bitwise AND of register A and register B."""
    self.mem[instr[3]] = self.mem[instr[1]] & self.mem[instr[2]]
  def opc_bani(self, instr):
    """bani (bitwise AND immediate) stores into register C the result of the bitwise AND of register A and value B."""
    self.mem[instr[3]] = self.mem[instr[1]] & instr[2]
  def opc_borr(self, instr):
    """borr (bitwise OR register) stores into register C the result of the bitwise OR of register A and register B."""
    self.mem[instr[3]] = self.mem[instr[1]] | self.mem[instr[2]]
  def opc_bori(self, instr):
    """bori (bitwise OR immediate) stores into register C the result of the bitwise OR of register A and value B."""
    self.mem[instr[3]] = self.mem[instr[1]] | instr[2]
  def opc_setr(self, instr):
    """setr (set register) copies the contents of register A into register C. (Input B is ignored.)"""
    self.mem[instr[3]] = self.mem[instr[1]]
  def opc_seti(self, instr):
    """seti (set immediate) stores value A into register C. (Input B is ignored.)"""
    self.mem[instr[3]] = instr[1]
  def opc_gtir(self, instr):
    """gtir (greater-than immediate/register) sets register C to 1 if value A is greater than register B.
    Otherwise, register C is set to 0."""
    if instr[1] > self.mem[instr[2]]:
      self.mem[instr[3]] = 1
    else:
      self.mem[instr[3]] = 0
  def opc_gtri(self, instr):
    """gtri (greater-than register/immediate) sets register C to 1 if register A is greater than value B.
    Otherwise, register C is set to 0."""
    if self.mem[instr[1]] > instr[2]:
      self.mem[instr[3]] = 1
    else:
      self.mem[instr[3]] = 0
  def opc_gtrr(self, instr):
    """gtrr (greater-than register/register) sets register C to 1 if register A is greater than register B.
    Otherwise, register C is set to 0."""
    if self.mem[instr[1]] > self.mem[instr[2]]:
      self.mem[instr[3]] = 1
    else:
      self.mem[instr[3]] = 0
  def opc_eqir(self, instr):
    """eqir (equal immediate/register) sets register C to 1 if value A is equal to register B.
    Otherwise, register C is set to 0."""
    if instr[1] == self.mem[instr[2]]:
      self.mem[instr[3]] = 1
    else:
      self.mem[instr[3]] = 0
  def opc_eqri(self, instr):
    """eqri (equal register/immediate) sets register C to 1 if register A is equal to value B.
    Otherwise, register C is set to 0."""
    if self.mem[instr[1]] == instr[2]:
      self.mem[instr[3]] = 1
    else:
      self.mem[instr[3]] = 0
  def opc_eqrr(self, instr):
    """eqrr (equal register/register) sets register C to 1 if register A is equal to register B.
    Otherwise, register C is set to 0."""
    if self.mem[instr[1]] == self.mem[instr[2]]:
      self.mem[instr[3]] = 1
    else:
      self.mem[instr[3]] = 0


# In[ ]:


tests = """
#ip 0
seti 5 0 1
seti 6 0 2
addi 0 1 0
addr 1 2 3
setr 1 0 0
seti 8 0 4
seti 9 0 5
""".strip().split("\n")


# In[ ]:


log.setLevel( aoc.LOGLEVEL_TRACE )
log.debug(f"effective-log-level={log.getEffectiveLevel()}")

log.debug(tests)
cpu = CPU()
cpu.prepare(tests)
cpu.interpret()
assert( [6, 5, 6, 0, 0, 9] == cpu.mem )


# In[ ]:


log.setLevel( logging.INFO )
log.info(f"effective-log-level={log.getEffectiveLevel()}")

ins = aoc.read_file_to_list('./in/day19.in')
log.debug(ins)

cpu = CPU()
cpu.prepare(ins)
cpu.interpret()
log.info(f"Day 19 a result-mem={cpu.mem}; cpu-state={cpu.state}")


# In[ ]:


RUN_DOUBTFUL = False

if RUN_DOUBTFUL: # this currently fails after _hours_ of runtime
  cpu = CPU()
  cpu.prepare(ins)
  cpu.mem[0] = 1
  log.info(f"  set-init-mem={cpu.mem}")
  cpu.interpret(steps=1_000_000)
  log.info(f"Day 19 b result-mem={cpu.mem}, cpu-state={cpu.state}")


# In[ ]:




