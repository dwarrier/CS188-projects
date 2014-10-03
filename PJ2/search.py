# search.py
# ---------
# Licensing Information:  You are free to use or extend these projects for 
# educational purposes provided that (1) you do not distribute or publish 
# solutions, (2) you retain this notice, and (3) you provide clear 
# attribution to UC Berkeley, including a link to 
# http://inst.eecs.berkeley.edu/~cs188/pacman/pacman.html
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero 
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and 
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


# search.py
# ---------
# Licensing Information:  You are free to use or extend these projects for 
# educational purposes provided that (1) you do not distribute or publish 
# solutions, (2) you retain this notice, and (3) you provide clear 
# attribution to UC Berkeley, including a link to 
# http://inst.eecs.berkeley.edu/~cs188/pacman/pacman.html
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero 
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and 
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""

import util
import sys
import logic

class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def getGhostStartStates(self):
        """
        Returns a list containing the start state for each ghost.
        Only used in problems that use ghosts (FoodGhostSearchProblem)
        """
        util.raiseNotDefined()

    def terminalTest(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()
        
    def getGoalState(self):
        """
        Returns goal state for problem. Note only defined for problems that have
        a unique goal state such as PositionSearchProblem
        """
        util.raiseNotDefined()

    def result(self, state, action):
        """
        Given a state and an action, returns resulting state and step cost, which is
        the incremental cost of moving to that successor.
        Returns (next_state, cost)
        """
        util.raiseNotDefined()

    def actions(self, state):
        """
        Given a state, returns available actions.
        Returns a list of actions
        """        
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()

    def getWidth(self):
        """
        Returns the width of the playable grid (does not include the external wall)
        Possible x positions for agents will be in range [1,width]
        """
        util.raiseNotDefined()

    def getHeight(self):
        """
        Returns the height of the playable grid (does not include the external wall)
        Possible y positions for agents will be in range [1,height]
        """
        util.raiseNotDefined()

    def isWall(self, position):
        """
        Return true if position (x,y) is a wall. Returns false otherwise.
        """
        util.raiseNotDefined()


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return  [s, s, w, s, w, w, s, w]


def atLeastOne(expressions) :
    """
    Given a list of logic.Expr instances, return a single logic.Expr instance in CNF (conjunctive normal form)
    that represents the logic that at least one of the expressions in the list is true.
    >>> A = logic.PropSymbolExpr('A');
    >>> B = logic.PropSymbolExpr('B');
    >>> symbols = [A, B]
    >>> atleast1 = atLeastOne(symbols)
    >>> model1 = {A:False, B:False}
    >>> print logic.pl_true(atleast1,model1)
    False
    >>> model2 = {A:False, B:True}
    >>> print logic.pl_true(atleast1,model2)
    True
    >>> model3 = {A:True, B:True}
    >>> print logic.pl_true(atleast1,model2)
    True
    """
    "*** YOUR CODE HERE ***"
    if not expressions:
      return logic.FALSE 
    return expressions[0] | atLeastOne(expressions[1::])

def atMostOne(expressions) :
    """
    Given a list of logic.Expr instances, return a single logic.Expr instance in CNF (conjunctive normal form)
    that represents the logic that at most one of the expressions in the list is true.
    """
    "*** YOUR CODE HERE ***"
    if not expressions:
      return logic.TRUE 
    # Actual logic:
    '''
    return (expressions[0] & ~atLeastOne(expressions[1::])) \
	| (~expressions[0] & atMostOne(expressions[1::]))
    '''
    # CNF:
    return (expressions[0] | ~expressions[0]) \
	& (expressions[0] | atMostOne(expressions[1::])) \
	& (~expressions[0] | ~atLeastOne(expressions[1::])) \
	& (~atLeastOne(expressions[1::]) | atMostOne(expressions[1::]))

def exactlyOne(expressions) :
    """
    Given a list of logic.Expr instances, return a single logic.Expr instance in CNF (conjunctive normal form)
    that represents the logic that exactly one of the expressions in the list is true.
    """
    "*** YOUR CODE HERE ***"
    if not expressions:
      return logic.FALSE 
    # Actual logic:
    '''
    return (expressions[0] & ~atLeastOne(expressions[1::])) \
	| (~expressions[0] & exactlyOne(expressions[1::]))
    '''
    # CNF:
    return (expressions[0] | ~expressions[0]) \
	& (expressions[0] | exactlyOne(expressions[1::])) \
	& (~expressions[0] | ~atLeastOne(expressions[1::])) \
	& (~atLeastOne(expressions[1::]) | exactlyOne(expressions[1::]))


def extractActionSequence(model, actions):
    """
    Convert a model in to an ordered list of actions.
    model: Propositional logic model stored as a dictionary with keys being
    the symbol strings and values being Boolean: True or False
    Example:
    >>> model = {"North[3]":True, "P[3,4,1]":True, "P[3,3,1]":False, "West[1]":True, "GhostScary":True, "West[3]":False, "South[2]":True, "East[1]":False}
    >>> actions = ['North', 'South', 'East', 'West']
    >>> plan = extractActionSequence(model, actions)
    >>> print plan
    ['West', 'South', 'North']
    """
    "*** YOUR CODE HERE ***"
    action_list = []
    time_expired = False
    t = 0
    while not time_expired:
      time_expired = True
      for a in actions:
        key = logic.PropSymbolExpr(a,t) 
	if key in model and model[key]:
	  action_list.append(a)
	  time_expired = False
      t += 1;
    return action_list

def positionLogicPlan(problem):
    """
    Given an instance of a PositionSearchProblem, return a list of actions that lead to the goal.
    Available actions are game.Directions.{NORTH,SOUTH,EAST,WEST}
    Note that STOP is not an available action.
    """
    "*** YOUR CODE HERE ***"
    # No layouts will require above 50 timesteps
    # as specified in the PJ2 spec.
    T_MAX = 51
    # @return value is list of logic.PropSymbolExpr
    all_exprs = []
    # all possible actions
    actions = [game.Directions.NORTH, \
	game.Directions.SOUTH, \
	game.Directions.EAST, \
	game.Directions.WEST }

    # ENCODE all walls
    for i in range(1, problem.getWidth()):
      for j in range(1, problem.getHeight()):
	wallExpr = logic.PropSymbolExpr("W",i,j)
	if problem.isWall((i,j)):
	  all_exprs.append(wallExpr)
	else:
	  all_exprs.append(~wallExpr)

    # ENCODE restriction for single action per timestep
    for t in range(T_MAX):
      # actions_for_t are
      # possible actions taken at a single timestep
      actions_for_t = []
      for a in actions:
	actions_for_t.append(
	    logic.PropSymbolExpr(a,t))
      all_exprs.append(
	  exactlyOne(actions_for_t))

    # ENCODE that goal must be true only once
    # among all timesteps
    gx, gy = problem.getGoalState()
    goals_for_t = []
    for t in range(T_MAX)
      goals_for_t.append(
	  logic.PropSymbolExpr("P", gx, gy, t))
    all_exprs.append(
	exactlyOne(goals_for_t))

    # ENCODE start state as given
    start_state = problem.getStartState()
    start_pos = logic.PropSymbolExpr("P",start_state[0], start_state[1], 0) 
    all_exprs.append(
	logic.PropSymbolExpr(
	  "P",start_state[0], start_state[1], 0))

    # ENCODE position requirements for each timestep

    # wallExpr encodes the proper logic.PropSymbolExpr
    # for a wall at position pos
    wallExpr = lambda pos : \
      logic.PropSymbolExpr("W", pos[0], pos[1])

    for t in range(T_MAX):
      pos_list = possiblePositionsFromStartForTimestep(
	  problem, t)
      for pos in pos_list:
	# don't want to enforce requirements
	# for the start state at t = 0
	if pos == start_state:
	  continue
	req_expr = makeRequirementsForPosition(pos, action_list)
	all_exprs.append( 
	  ~wallExpr(pos)
	  & (~pos | req_expr) 
	  & (~req_expr | pos))
	# TODO: make these seperate entries for speed?
    
    # SOLVE all_exprs and return actions
    model = logic.pycoSAT(all_exprs)
    return extractActionSequence(model)

# ASSUMES pacman can only move one square per timestep.
# returns a list of all possible positions pacman could
# have reached from start pos in timestep t.
# Returns a list of logic.PropSymbolExprs.
# @param t is the timestep
def possiblePositionsFromStartForTimestep(problem, t):
  pos_list = []
  x, y = problem.getStartState()
  # account for problem bounds when iterating
  for i in range(max(1,x-t), min(x+t+1, problem.getWidth())):
    for j in range(max(1, y-t), min(y+t+1, problem.getHeight())):
      pos_list.append(logic.PropSymbolExpr("P",i,j,t))
  return pos_list 

def makeRequirementsForPosition(problem, pos):
  # wallExpr encodes the proper logic.PropSymbolExpr
  # for a wall at position pos
  wallExpr = lambda pos : \
    logic.PropSymbolExpr("W", pos[0], pos[1])
  # flipTup is used for looking up actions from current
  # position that hit a wall in the previous timestep
  # (this is just backwards from the regular action_map
  # keys)
  flipTup = lambda tup : \
      (-1*tup[0],-1*tup[1])
  # action_map gives the action that must lead
  # to this position starting from (x+i, y+j) for
  # the key (i,j)
  action_map = {(-1,0) : game.directions.EAST, \
      (0, 1) : game.directions.SOUTH, \
      (1, 0) : game.directions.WEST, \
      (0, -1) : game.directions.NORTH }
  # actionExpr makes a logic.PropSymbolExpr
  # given an action and timestep
  actionExpr = lambda action, t : \
      logic.PropSymbolExpr(action,t)

  # current x pos, y pos, and timestep
  x,y,t = logic.PropSymbolExpr.parse(pos)

  arglist = []
  for i in range(x-1, x+1):
    for j in range(y-1, y+1):
      # rule out positions outside the proper range
      if math.abs(i) != math.abs(j):
	# position modifier
	pos = (x,y)
	pos_mod = (i-x,j-y)
        # EITHER you were in square (x,y) at t-1
	# and there was a wall in position p
	# that corresponds to action a
        arglist.append(
	  wallExpr(pos, pos_mod)
	  & actionExpr(action_map[flipTup(pos_mod)],t-1)
	  & logic.PropSymbolExpr("P",x,y,t-1))
	# OR you were in square (i,j) at t-1
	# and took the appropriate action a
	# to get here
	# (we don't need the wall condition here
	# because that is accounted for in the
	# requirements for p')
	arglist.append(
	  actionExpr(action_map[pos_mod],t-1)
	  & logic.PropSymbolExpr("P",i,j,t-1))
  # return <=> statement, and make sure there is
  # no wall at this position!
  action_reqs = exactlyOne(arglist)
  return action_reqs

# Returns a logic.PropSymbolExpr to check whether
# there is a wall in the given position.
def wallExistsAt(position):


def foodLogicPlan(problem):
    """
    Given an instance of a FoodSearchProblem, return a list of actions that help Pacman
    eat all of the food.
    Available actions are game.Directions.{NORTH,SOUTH,EAST,WEST}
    Note that STOP is not an available action.
    """
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()

def foodGhostLogicPlan(problem):
    """
    Given an instance of a FoodGhostSearchProblem, return a list of actions that help Pacman
    eat all of the food and avoid patrolling ghosts.
    Ghosts only move east and west. They always start by moving East, unless they start next to
    and eastern wall. 
    Available actions are game.Directions.{NORTH,SOUTH,EAST,WEST}
    Note that STOP is not an available action.
    """
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()


# Abbreviations
plp = positionLogicPlan
flp = foodLogicPlan
fglp = foodGhostLogicPlan

# Some for the logic module uses pretty deep recursion on long expressions
sys.setrecursionlimit(100000)



