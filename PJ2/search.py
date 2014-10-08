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
import game

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
    expr = expressions[0]
    for e in expressions:
      expr = expr | e
    return expr

def atMostOne(expressions) :
    """
    Given a list of logic.Expr instances, return a single logic.Expr instance in CNF (conjunctive normal form)
    that represents the logic that at most one of the expressions in the list is true.
    """
    "*** YOUR CODE HERE ***"
    expr = expressions[0] | ~expressions[0]
    for i in range(len(expressions)):
      for j in range(i+1, len(expressions)):
	expr = expr & (~expressions[i] | ~expressions[j])
    return expr 

def exactlyOne(expressions) :
    """
    Given a list of logic.Expr instances, return a single logic.Expr instance in CNF (conjunctive normal form)
    that represents the logic that exactly one of the expressions in the list is true.
    """
    "*** YOUR CODE HERE ***"
    return atMostOne(expressions) & atLeastOne(expressions)

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
        key = PSE(a,t) 
	if key in model and model[key]:
	  action_list.append(a)
	  time_expired = False
      t += 1;
    return action_list

ALL_ACTIONS = dN, dE, dS, dW = \
    [ game.Directions.NORTH, game.Directions.EAST,
      game.Directions.SOUTH, game.Directions.WEST ]

# Use a global dict to cache successor states
pos_succ_state_dict = {}

def positionLogicPlan(problem):
    """
    Given an instance of a PositionSearchProblem, return a list of actions that lead to the goal.
    Available actions are game.Directions.{NORTH,SOUTH,EAST,WEST}
    Note that STOP is not an available action.
    """
    "*** YOUR CODE HERE ***"
    T_MAX = 51
    pycoSAT_args = []
    pos_succ_state_dict = {}

    # ENCODE start state axioms
    # there's only one start state 
    (sx,sy) = problem.getStartState()
    pycoSAT_args.append(
	PSE("P",sx,sy,0))
    for i in range(0, problem.getWidth() + 2):
      for j in range(0, problem.getHeight() + 2):
	if (i,j) != (sx,sy):
	  pycoSAT_args.append(
	      ~PSE("P",i,j, 0))

    # ENCODE initial action exclusion axioms
    pycoSAT_args.append(
	exactlyOne(
	  [PSE(a, 0) for a in ALL_ACTIONS]))

    # PERFORM ITERATIVE DEEPENING.
    # start depth at minimum possible timestep count
    # to save time.
    model = False
    (gx,gy) = problem.getGoalState()
    depth = abs(sx-gx) + abs(sy-gy) 
    
    while (model == False) and (depth < T_MAX):
      copy = [logic.expr(s) for s in pycoSAT_args]
      model = depthLimitedPlan(problem, copy, depth)
      depth += 1
    
    return extractActionSequence(model, ALL_ACTIONS)

def depthLimitedPlan(problem, initial_expr_list, depth):
    (gx,gy) = problem.getGoalState()
    goal_state_axioms = []

    for t in range(1, depth):

      # UPDATE goal state axioms
      updatePositionPlanGoalStates(goal_state_axioms, gx, gy, t)

      # UPDATE action exclusion axioms
      initial_expr_list.append(
	  exactlyOne(
	    [PSE(a, t) for a in ALL_ACTIONS]))

      for i in range(0,problem.getWidth() + 2):
        for j in range(0,problem.getHeight() + 2):
	  if problem.isWall((i,j)):
	    initial_expr_list.append(~P(i,j,t))

      # ENCODE successor states 
      for i in range(1,problem.getWidth() + 1):
        for j in range(1,problem.getHeight() + 1):
	  updatePositionPlanSuccStates(initial_expr_list,i,j,t)

    # ENCODE goal state axioms
    initial_expr_list.append(exactlyOne(goal_state_axioms))
    model = logic.pycoSAT(initial_expr_list)
    return model 

def updatePositionPlanGoalStates(goal_state_list,gx,gy,t):
  goal_state_list.append(PSE("P", gx, gy, t))

def updatePositionPlanSuccStates(expr_list,i,j,t):

  if P(i,j,t) not in pos_succ_state_dict:
    pos_succ_state_dict[P(i,j,t)] = CNF(
	P(i,j,t) % \
	  (P(i-1, j,t-1) & A(dE,t-1)) |
	  (P(i+1, j,t-1) & A(dW,t-1)) |
	  (P(i, j-1,t-1) & A(dN,t-1)) |
	  (P(i, j+1,t-1) & A(dS,t-1)))
  expr_list.append(pos_succ_state_dict[P(i,j,t)])

def foodLogicPlan(problem):
    """
    Given an instance of a FoodSearchProblem, return a list of actions that help Pacman
    eat all of the food.
    Available actions are game.Directions.{NORTH,SOUTH,EAST,WEST}
    Note that STOP is not an available action.
    """
    "*** YOUR CODE HERE ***"
    T_MAX = 51

    pycoSAT_args = []

    # ENCODE start state axioms
    # there's only one start state 
    (sx,sy) = problem.getStartState()[0]
    pycoSAT_args.append(
	PSE("P",sx,sy,0))
    for i in range(0, problem.getWidth() + 2):
      for j in range(0, problem.getHeight() + 2):
	if (i,j) != (sx,sy):
	  pycoSAT_args.append(
	      ~PSE("P",i,j, 0))

    # food start states
    foodGrid = problem.getStartState()[1]
    for i in range(1,problem.getWidth() + 1):
        for j in range(1,problem.getHeight() + 1):
            if foodGrid[i][j]:
                pycoSAT_args.append(F(i,j,0))
            else:
                pycoSAT_args.append(~F(i,j,0))

    # ENCODE initial action exclusion axioms
    pycoSAT_args.append(
	exactlyOne(
	  [PSE(a, 0) for a in ALL_ACTIONS]))
    pycoSAT_args.append(A(dW,0))

    # PERFORM ITERATIVE DEEPENING.
    # start depth at minimum possible timestep count
    # to save time.
    model = False
    depth = foodGrid.count() + 1 
    
    while (model == False) and (depth < T_MAX):
      copy = [logic.expr(s) for s in pycoSAT_args]
      model = foodDepthLimitedPlan(problem, copy, depth)
      depth += 1
    
    return extractActionSequence(model, ALL_ACTIONS)

def foodDepthLimitedPlan(problem, initial_expr_list, depth):
    goal_state_axioms = []

    for t in range(1, depth):

      # UPDATE goal state axioms
      updateFoodPlanGoalStates(initial_expr_list,goal_state_axioms, problem, t)

      # UPDATE action exclusion axioms
      initial_expr_list.append(
	  exactlyOne(
	    [PSE(a, t) for a in ALL_ACTIONS]))

      # UPDATE walls
      for i in range(0,problem.getWidth() + 2):
        for j in range(0,problem.getHeight() + 2):
	  if problem.isWall((i,j)):
	    initial_expr_list.append(~P(i,j,t))

      # ENCODE successor states 
      for i in range(1,problem.getWidth() + 1):
        for j in range(1,problem.getHeight() + 1):
	  updateFoodPlanSuccStates(initial_expr_list,i,j,t)

    # ENCODE goal state axioms
    initial_expr_list.append(CNF(exactlyOne(goal_state_axioms)))
    model = logic.pycoSAT(initial_expr_list)
    return model 

def updateFoodPlanGoalStates(expr_list,goal_state_list,problem,t):
    foods = []
    for i in range(1,problem.getWidth() + 1):
        for j in range(1,problem.getHeight() + 1):
            foods.append(F(i,j,t))
    expr_list.append(CNF(G(t) % CNF(~atLeastOne(foods))))
    goal_state_list.append(G(t))

def updateFoodPlanSuccStates(expr_list,i,j,t):
  expr_list.append(CNF(
      P(i,j,t) % \
	(P(i-1, j,t-1) & A(dE,t-1)) |
	(P(i+1, j,t-1) & A(dW,t-1)) |
	(P(i, j-1,t-1) & A(dN,t-1)) |
	(P(i, j+1,t-1) & A(dS,t-1))))
  # food successor states
  expr_list.append(CNF(~F(i,j,t) % (~F(i,j,t-1) | P(i,j,t))))

# shortcuts
CNF = logic.to_cnf
PSE = logic.PropSymbolExpr

# Enemy (ghost)
def E(x,y,t):
  return PSE("E",x,y,t)

# Goal
def G(t):
  return PSE("G",t)

# Pacman's position
def P(x,y,t):
  return PSE("P",x,y,t)

# Action
def A(s,t):
  return PSE(s,t)

# Food
def F(x,y,t):
  return PSE("F",x,y,t)

# Walls
def W(x,y):
  return PSE("W",x,y)

# ghost east
def GE(x,y,t):
  return PSE("GE",x,y,t)

# ghost west
def GW(x,y,t):
  return PSE("GW",x,y,t)


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

    T_MAX = 5 

    pycoSAT_args = []

    # ENCODE start state axioms
    # there's only one start state 
    (sx,sy) = problem.getStartState()[0]
    pycoSAT_args.append(
	PSE("P",sx,sy,0))
    for i in range(0, problem.getWidth() + 2):
      for j in range(0, problem.getHeight() + 2):
	if (i,j) != (sx,sy):
	  pycoSAT_args.append(
	      ~PSE("P",i,j, 0))

    # food start states
    foodGrid = problem.getStartState()[1]
    for i in range(1,problem.getWidth() + 1):
        for j in range(1,problem.getHeight() + 1):
            if foodGrid[i][j]:
                pycoSAT_args.append(F(i,j,0))
            else:
                pycoSAT_args.append(~F(i,j,0))

    for i in range(problem.getWidth() + 2):
      for j in range(problem.getHeight() + 2):
	if problem.isWall((i,j)):
	  pycoSAT_args.append(W(i,j))
	else:
	  pycoSAT_args.append(~W(i,j))

    # ghost start states
    '''
    ghost_start_positions = []
    for g in problem.getGhostStartStates():
      ghost_start_positions.append(g.getPosition())
    for sx,sy in ghost_start_positions:
      pycoSAT_args.append(E(sx,sy,0))
      if problem.isWall((sx + 1,sy)): 
	pycoSAT_args.append(GW(sx,sy,0))
	pycoSAT_args.append(~GE(sx,sy,0))
      else:
	pycoSAT_args.append(~GW(sx,sy,0))
	pycoSAT_args.append(GE(sx,sy,0))
      pycoSAT_args.append(atMostOne([GE(sx,sy,0), GW(sx,sy,0)]))
    '''
    '''
    for i in range(problem.getWidth() + 2):
      for j in range(problem.getHeight() + 2):
	if (i,j) not in ghost_start_positions:
	  pycoSAT_args.append(~E(i,j,0))
    '''	   
    print(pycoSAT_args)

    # ENCODE initial action exclusion axioms
    pycoSAT_args.append(
	exactlyOne(
	  [PSE(a, 0) for a in ALL_ACTIONS]))

    # PERFORM ITERATIVE DEEPENING.
    # start depth at minimum possible timestep count
    # to save time.
    model = False
    depth = foodGrid.count() + 1
    
    while (model == False) and (depth < T_MAX):
      copy = [logic.expr(s) for s in pycoSAT_args]
      model = ghostDepthLimitedPlan(problem, copy, depth)
      depth += 1
    
    return extractActionSequence(model, ALL_ACTIONS)

def ghostDepthLimitedPlan(problem, initial_expr_list, depth):
    goal_state_axioms = []

    for t in range(1, depth):

      # UPDATE goal state axioms
      updateGhostPlanGoalStates(initial_expr_list,goal_state_axioms, problem, t)

      # UPDATE action exclusion axioms
      initial_expr_list.append(
	  exactlyOne(
	    [PSE(a, t) for a in ALL_ACTIONS]))

      # UPDATE walls
      for i in range(0,problem.getWidth() + 2):
        for j in range(0,problem.getHeight() + 2):
	  if problem.isWall((i,j)):
	    initial_expr_list.append(~P(i,j,t))
	    #initial_expr_list.append(~E(i,j,t))
     
      # ENCODE successor states 
      for i in range(1,problem.getWidth() + 1):
        for j in range(1,problem.getHeight() + 1):
	  updateGhostPlanSuccStates(initial_expr_list,i,j,t)
          #initial_expr_list.append(atMostOne([GE(i,j,t), GW(i,j,t)]))

    # ENCODE goal state axioms
    initial_expr_list.append(CNF(exactlyOne(goal_state_axioms)))
    model = logic.pycoSAT(initial_expr_list)
    print model
    return model 

def updateGhostPlanGoalStates(expr_list,goal_state_list,problem,t):
    foods = []
    for i in range(1,problem.getWidth() + 1):
        for j in range(1,problem.getHeight() + 1):
            foods.append(F(i,j,t))
    expr_list.append(CNF(G(t) % CNF(~atLeastOne(foods))))
    goal_state_list.append(G(t))

def updateGhostPlanSuccStates(expr_list,i,j,t):
  expr_list.append(CNF(
      P(i,j,t) % \
	(P(i-1, j,t-1) & A(dE,t-1)) |
	(P(i+1, j,t-1) & A(dW,t-1)) |
	(P(i, j-1,t-1) & A(dN,t-1)) |
	(P(i, j+1,t-1) & A(dS,t-1))))
  '''
  expr_list.append(CNF(E(i,j,t) % \
      (GE(i-1,j,t-1) & E(i-1,j,t-1)) |
      (GW(i+1,j,t-1) & E(i+1,j,t-1))))
  '''

  '''
  expr_list.append(CNF(
  GE(i,j,t) % \
    (~W(i+1,j) & GE(i-1,j,t-1) & E(i-1,j,t-1)) | (GW(i,j,t-1) & W(i-1,j) & E(i,j,t))))

  '''
  '''
  expr_list.append(CNF(
    GE(i,j,t) % \
      (~W(i+1,j) & GE(i-1,j,t-1) & E(i-1,j,t-1)) | (W(i-1,j) & E(i,j,t))))

  expr_list.append(CNF(
    GW(i,j,t) % \
      (~W(i-1,j) & GW(i-1,j,t-1) & E(i+1,j,t-1)) | (W(i+1,j) & E(i,j,t))))
  '''
  '''
  expr_list.append(CNF(GW(i,j,t) % \
      (~W(i-1,j) & GW(i+1,j,t-1) & E(i+1,j,t-1)) |  (W(i+1,j) & GE(i,j,t-1) & E(i,j,t))))
  '''
  #expr_list.append(CNF(P(i,j,t) % (~E(i,j,t+1) & ~E(i,j,t))))
  #expr_list.append(CNF(P(i,j,t) >> ~E(i,j,t-1)))
  expr_list.append(A(dW,1))

  # food successor states
  expr_list.append(CNF(~F(i,j,t) % (~F(i,j,t-1) | P(i,j,t))))


# Abbreviations
plp = positionLogicPlan
flp = foodLogicPlan
#fglp = foodGhostLogicPlan
fglp = foodLogicPlan

# Some for the logic module uses pretty deep recursion on long expressions
sys.setrecursionlimit(100000)



