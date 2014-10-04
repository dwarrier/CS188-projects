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
    # only one action per timestep
    # only one goal state
    # action_at_t <=> position_at_t & position_at_t+1
    # position_at_t => no_wall_at_t

    initial_state_axioms = []
    goal_state_axioms = []
    precondition_axioms = []
    successor_axioms = []
    action_exclusion_axioms = []
    expr = []
    all_actions = [
      game.Directions.NORTH, 
      game.Directions.SOUTH,
      game.Directions.EAST,
      game.Directions.WEST]
    dN,dS,dE,dW = all_actions

    T_MAX = 25 

    (sx,sy) = problem.getStartState()
    (gx,gy) = problem.getGoalState()
    expr.append(
	logic.PropSymbolExpr("P",sx,sy,0))
    # there's only one start state 
    '''
    for i in range(1, problem.getWidth() + 1):
      for j in range(1, problem.getHeight() + 1):
    '''
    for i in range(0, problem.getWidth() + 2):
      for j in range(0, problem.getHeight() + 2):
	if (i,j) != (sx,sy):
	  expr.append(
	      ~logic.PropSymbolExpr("P",i,j, 0))
    # walls.
    '''
    for i in range(1, problem.getWidth() + 1):
      for j in range(1, problem.getHeight() + 1):
    '''
    for i in range(0, problem.getWidth() + 2):
      for j in range(0, problem.getHeight() + 2):
	if problem.isWall((i,j)) \
	   | (i > problem.getWidth()) \
	   | (j > problem.getHeight()):
	  expr.append(W(i,j))
	else:
	  expr.append(~W(i,j))
    # update action exclusion axioms
    expr.append(
	exactlyOne(
	  [logic.PropSymbolExpr(a, 0) for a in all_actions]))
    model = False
    # start depth at minimum possible timestep count
    depth = abs(sx-gx) + abs(sy-gy) 
    
    while (model == False) and (depth < T_MAX):
      copy = [logic.expr(s) for s in expr]
      model = depthLimitedPlan(problem, all_actions, copy, depth)
      depth += 1
    
    return extractActionSequence(model, all_actions)

def depthLimitedPlan(problem, all_actions, expr, depth):
    (gx,gy) = problem.getGoalState()
    goal_state_axioms = []
    dN,dS,dE,dW = all_actions
    for t in range(1, depth):
      # update goal state axioms
      goal_state_axioms.append(
	  logic.PropSymbolExpr("P", gx, gy, t))
      # update action exclusion axioms
      expr.append(
	  exactlyOne(
	    [logic.PropSymbolExpr(a, t) for a in all_actions]))
      # update successor states 
      for i in range(1,problem.getWidth() + 1):
        for j in range(1,problem.getHeight() + 1):

	  expr.append( logic.to_cnf(
	  P(i,j,t) % 
	    (~W(i,j) & (
	    cnf((P(i-1, j,t-1) & A(dE,t-1) & ~W(i-1,j))) |
	    cnf((P(i+1, j,t-1) & A(dW,t-1) & ~W(i+1,j))) |
	    cnf((P(i, j-1,t-1) & A(dN,t-1) & ~W(i,j-1))) |
	    cnf((P(i, j+1,t-1) & A(dS,t-1) & ~W(i,j+1))))
	    )))
    expr.append(
	exactlyOne(goal_state_axioms))
    model = logic.pycoSAT(expr)
    return model 


cnf = logic.to_cnf
def P(x,y,t):
  return logic.PropSymbolExpr("P",x,y,t)

def A(s,t):
  return logic.PropSymbolExpr(s,t)

def W(x,y):
  return logic.PropSymbolExpr("W",x,y)

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



