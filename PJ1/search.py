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
import copy

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

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
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

def breadthFirstSearch(problem):
    """
    Search the shallowest nodes in the search tree first.

    You are not required to implement this, but you may find it useful for Q5.
    """
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()

def depthLimitedSearch(problem, maxDepth):
    """
    Search the deepest nodes in the search tree first. Only search to indicated depth.
    """
    closedList = set()
    lifo = util.Stack()
    frontier = set()

    startState = problem.getStartState()
    lifo.push((startState,[],1))
    frontier.add(startState)
    while not lifo.isEmpty():
        (state,actions,depth) = lifo.pop() #to be expanded next
        frontier = frontier - set([state])
        # if state in closedList: #if in closed list, aka already expanded, move on
        #     print(state)
        #     print("is in closedList")
        #     continue
        # print(state)
        # print("not in closedList")
        if problem.isGoalState(state): #if reached goal state, done!
            return actions
        closedList.add(state)
        if depth < maxDepth: #continue going deeper
            successors = problem.getSuccessors(state);
            for (state,action,cost) in successors:
                if state not in closedList and state not in frontier:
                    lifo.push((state,actions + [action],depth + 1))
                    frontier.add(state)
    return None


def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def iterativeDeepeningSearch(problem):
    """
    Perform DFS with increasingly larger depth.

    Begin with a depth of 1 and increment depth by 1 at every step.
    """
    "*** YOUR CODE HERE ***"
    depth = 1
    solution = None
    while solution == None:
        solution = depthLimitedSearch(problem,depth)
        depth += 1
    return solution
    # for depth in range(1,6):
    #     solution = depthLimitedSearch(problem,depth)
    #     if solution != None:
    #         return solution
    # return solution


def aStarSearch(problem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    "*** YOUR CODE HERE ***"
    # setup priority queue
    estimate_func = lambda((state, action, cost)) : heuristic(state, problem) + cost
    frontier = util.PriorityQueueWithFunction(estimate_func)
    visited = []
    # initial actions are empty, inital step cost is 0
    frontier.push((problem.getStartState(), [], 0))
    while frontier.count != 0:
      # pop state 
      (state, action_list, curr_cost) = frontier.pop()
      # need to expand?
      if state in visited:
        continue;
      if problem.isGoalState(state):
	return action_list
      # expand state
      for (child_state,action,step_cost) in problem.getSuccessors(state):
	frontier.push((child_state, action_list + [action], curr_cost + step_cost))
      # mark as visited
      visited.append(state)

# Abbreviations
bfs = breadthFirstSearch
astar = aStarSearch
ids = iterativeDeepeningSearch
