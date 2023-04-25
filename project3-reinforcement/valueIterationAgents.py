# valueIterationAgents.py
# -----------------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


# valueIterationAgents.py
# -----------------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


import mdp, util

from learningAgents import ValueEstimationAgent
import collections

class ValueIterationAgent(ValueEstimationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A ValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100):
        """
          Your value iteration agent should take an mdp on
          construction, run the indicated number of iterations
          and then act according to the resulting policy.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state, action, nextState)
              mdp.isTerminal(state)
        """
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations
        self.values = util.Counter() # A Counter is a dict with default 0
        self.runValueIteration()

    def runValueIteration(self):

        while self.iterations!=0:
            self.iterations = self.iterations - 1
            valueList = util.Counter()
            for state in self.mdp.getStates():
                qValue = -100000
                for action in self.mdp.getPossibleActions(state):
                    curr = self.computeQValueFromValues(state, action)
                    qValue = max(qValue, curr)

                if qValue != -100000:
                    valueList[state] = qValue
            self.values = valueList.copy()
     
            
    def getValue(self, state):
        """
          Return the value of the state (computed in __init__).
        """
        return self.values[state]


    def computeQValueFromValues(self, state, action):
        """
          Compute the Q-value of action in state from the
          value function stored in self.values.
        """
        "*** YOUR CODE HERE ***"
        transitionProbs = self.mdp.getTransitionStatesAndProbs(state, action)
        curr = 0
        for nextState, probability in transitionProbs:
            curr += probability * (self.mdp.getReward(state, action, nextState) + self.discount * self.values[nextState])    
        return curr 

    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        "*** YOUR CODE HERE ***"
        # allActions = util.Counter
        # if self.mdp.isTerminal(state):
        #     return None
        # else:
        #     for action in self.mdp.getPossibleActions(state):
        #         allActions[action] = self.getQValue(state, action)
        #     return allActions.argMax()
        
        allActions = util.Counter()
        if self.mdp.isTerminal(state):
          return None
        else:
            for action in self.mdp.getPossibleActions(state):
                allActions[action] = self.getQValue(state, action)
                
            return allActions.argMax()

    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)

class AsynchronousValueIterationAgent(ValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        An AsynchronousValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs cyclic value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 1000):
        """
          Your cyclic value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy. Each iteration
          updates the value of only one state, which cycles through
          the states list. If the chosen state is terminal, nothing
          happens in that iteration.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state)
              mdp.isTerminal(state)
        """
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        "*** YOUR CODE HERE ***"

        # Get the list of all states in the MDP
        states_list = self.mdp.getStates()
        num_states = len(states_list) #total number of states

        # Loop through the specified number of iterations
        for state_idx in range(self.iterations): 
            #Select a state based on the current index of the iteration
            state = states_list[state_idx % num_states] 

            if not self.mdp.isTerminal(state): #if state is not terminal
                 # Select an action to take in the current state
                action = self.getAction(state)
                qval = self.getQValue(state, action) #get the qval for that action at that state
                #Update the value of the current state with the calculated qval
                self.values[state] = qval


class PrioritizedSweepingValueIterationAgent(AsynchronousValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A PrioritizedSweepingValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs prioritized sweeping value iteration
        for a given number of iterations using the supplied parameters.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100, theta = 1e-5):
        """
          Your prioritized sweeping value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy.
        """
        self.theta = theta
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        "*** YOUR CODE HERE ***"
        
        #Get list of predecessors of all states 
        predecessors = {}  
        for state in self.mdp.getStates():
            if not self.mdp.isTerminal(state):
                # For each non-terminal state, get all the possible actions and their resulting next states and probabilities
                for action in self.mdp.getPossibleActions(state):
                    for next_state, prob in self.mdp.getTransitionStatesAndProbs(state, action):
                        # Add the current state as a predecessor to the next state
                        predecessors.setdefault(next_state, set()).add(state)

        
        state_priority_queue = util.PriorityQueue()
        for state in self.mdp.getStates():
            if not self.mdp.isTerminal(state):
                # For each non-terminal state, compute the difference between the current value and the maximum Q-value
                qvalues = []
                for action in self.mdp.getPossibleActions(state): 
                    qval = self.getQValue(state, action)
                    qvalues.append(qval)

                difference = abs(self.values[state] - max(qvalues, default=0))
                # Add the state and its priority (negative difference) to the priority queue
                state_priority_queue.update(state, -difference)

        
        for _ in range(self.iterations):
            if state_priority_queue.isEmpty():
                break

            # Pop the state with the highest priority (smallest negative difference)
            current_state = state_priority_queue.pop()

            if not self.mdp.isTerminal(current_state):
                # Compute the value of the current state as the maximum Q-value among all possible actions
                qvalues = []
                for action in self.mdp.getPossibleActions(current_state):
                    qval = self.getQValue(current_state, action)
                    qvalues.append(qval)

                self.values[current_state] = max(qvalues, default=0)

                # Update the priority queue for all predecessors of the current state
                for predecessor in predecessors.get(current_state, []):
                    qvalues = []
                    for action in self.mdp.getPossibleActions(predecessor): 
                        qval = self.getQValue(predecessor, action)
                        qvalues.append(qval)

                    difference = abs(self.values[predecessor] - max(qvalues, default=0))
                    # If the difference exceeds the threshold, update the priority of the predecessor in the priority queue
                    if difference > self.theta:
                        state_priority_queue.update(predecessor, -difference)


