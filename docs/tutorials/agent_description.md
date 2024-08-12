The concept of an agent within the olfactory navigation framework is a an abstract concept.

Fundamentally, to create an agent, the user has to create a [subclass](https://www.w3schools.com/python/python_inheritance.asp) of the generic [Agent](../reference/agent.md) class. Along with this, the various functions need to be overwritten as needed. This article will describe the various components of the Agent class and how to make one for yourself with some examples.

The various components can be seperated between the [attributes](#attributes) and [methods or functions](#functions). The functions can be grouped in the following categories [training](#training), [testing](#testing), and [general](#general) functions. As part of the testing function, the functions required to interact with the [*run_test()*](../reference/simulation.md#olfactory_navigation.simulation.run_test) function.

Table Of Content:

- [Attributes](#attributes)
- [Functions](#functions)
    - [Training](#training)
    - [Testing](#testing)
        - [State initialization](#state-initialization)
        - [Action choice](#action-choice)
        - [State update](#state-update)
        - [Agent pruning](#agent-pruning)
    - [General](#general)
        - [Save/load](#saveload)
        - [GPU support](#gpu-support)



## Attributes

The agent has a set of attributes by default. Here is a list of some important attributes:

- *environment*: The [Environment](../reference/environment.md) the agent should be trained on.
- *threshold*: The olfactory sensitivity of the agent. A list of multiple threshold can also be set to represent the agent being able to smell different concentrations of odors.
- *action_set*: A list of the actions available to the agent. It should be an array of different action vectors.

On top of these attributes, other ones are also present such as *name*, or *action_labels* which help with more general functions, they are not directly involved with the core functionality of the agent.

Following this, other attributes can be defined within the init function. Those attribute can be of one of three categories:

- Static: Those attribute should be there to help within the functions of the agent. For example, we can imagine some conversion matrix the agent can use to update its state.
- Trainable: Those attribute typically should be set to *None* within the init and defined after the *train()* function has been run.
- Status: Those attributes are dynamic variables that will change over the course of the simulation. They should reprensent the status of the agent during the simulation (for example, where the agent believes he is in the space, or an internal clock, or even a memory).



## Functions

### Training

Function: *train()* <sub>[reference](../reference/agent.md#olfactory_navigation.agent.Agent.train)</sub>

For the training part, the concept of the agent is trained. We define the brain of the agent. It is optional to overwrite this function as we can imagine an agent that has some hard-coded behavior and therefore does not require training.



### Testing

For the testing, the agent needs to be able to interact be interacted with from the [*run_test()*](../reference/simulation.md#olfactory_navigation.simulation.run_test) function of the [simulation](../reference/simulation.md) module.

For this the following functions need to be implemented:

1. *initialize_state(n)* <sub>[reference](../reference/agent.md#olfactory_navigation.agent.Agent.initialize_state)</sub>
2. *choose_action()* <sub>[reference](../reference/agent.md#olfactory_navigation.agent.Agent.choose_action)</sub>
3. *update_state(observation, source_reached)* <sub>[reference](../reference/agent.md#olfactory_navigation.agent.Agent.update_state)</sub>
4. *kill(simulations_to_kill)* <sub>[reference](../reference/agent.md#olfactory_navigation.agent.Agent.kill)</sub>

#### State initialization

Function: *initialize_state(n)* <sub>[reference](../reference/agent.md#olfactory_navigation.agent.Agent.initialize_state)</sub>

In this function, the way the initial state of an agent should be has to be implemented. For example, say the agent concept you are defining relies on a clock system, at this step, the clocks for the *n* agents would be initialized for example at zero. Another example is for the POMDP-based agents, the initial "belief" of the agent has to be set.

Note: This step can be skipped with a *skip_initialization* parameter of the *run_test()* function that can be useful if want to initialize the agent in a particular manner compared to the default agent's initialization.

#### Action choice

Function: *choose_action()* <sub>[reference](../reference/agent.md#olfactory_navigation.agent.Agent.choose_action)</sub>

In the step of the simulation process, it is asked of the agent(s) to choose an action to make based on their internal state (clock or something else), hence why no parameter is being passed here.

The action chosen should be a vector (or a list of vectors) with the first component of the vector being the olfactory layer to be querried; then the other components should be the displacements in each dimension of the environment. Note that, the layer component is not needed in the case the environment is not a layered environment. Then, the amount of dispacement compenents should also match the amount of physical dimensions of the environment.

#### State update

Function: *update_state(observation, source_reached)* <sub>[reference](../reference/agent.md#olfactory_navigation.agent.Agent.update_state)</sub>

Following the action of the agent being applied, an observation is returned to it along with a cue of whether or not it has found the source of the odor. The odor cue that is returned to the agent depends on the actual position of the agent relative to the odor source (which is unknown to him). The odor cue returned to the agent doesn't only depend on its physical position, it also depend on the time of the simulation. Unless specified otherwise using the *time_shift* parameter of the *run_test()* function, the time starts at zero for all of the agents. 

This function is therefore meant for the agent to update its internal state based on the observation it received. For example, if the agent's state is a clock measuring the amount of time since it last received an odor cue, if the agent receives an odor cue above its olfaction threshold, the clock would be reset to zero, else it would be incremented by one.

#### Agent pruning

Function: *kill(simulations_to_kill)* <sub>[reference](../reference/agent.md#olfactory_navigation.agent.Agent.kill)</sub>

When a agent reached the source, the simulation will not keep track if it anymore and will prune it's position its memory. This is done to speed up the simulation as more and more agents reach the source. The simulation therefore returns to agent which is to so the states of the agents that have reached the source can also be pruned.

This is done by sending a boolean array of which agent's states are the be pruned/killed to the *kill()* function.



### General

#### Save/load

The agent must be able to be saved to memory for reproducability's sake. Related functions:

- *save()* <sub>[reference](../reference/agent.md#olfactory_navigation.agent.Agent.save)</sub>
- *load()* <sub>[reference](../reference/agent.md#olfactory_navigation.agent.Agent.load)</sub>

By default, the agent can be saved as a [pickle](https://realpython.com/python-pickle-module/) file. But it is recommended to define your own *save()* and *load()* functions as a pickle file is a blackbox and may save too many attributes, so not very space efficient. For instance, for the agent to be re-used later, only the trainable variables are needed.

#### GPU support

To speedup large operations, the arrays associated to the agent can be sent to the GPU for the operations to be performed there. Related functions:

- *to_gpu()* <sub>[reference](../reference/agent.md#olfactory_navigation.agent.Agent.to_gpu)</sub>
- *to_cpu()* <sub>[reference](../reference/agent.md#olfactory_navigation.agent.Agent.to_cpu)</sub>

These functions are to be defined to send the various componenents to and from the gpu memory. If some components are not directly Numpy arrays but contain Numpy arrays, these components should be handled specifically by overwriting the *to_gpu()* function.

By default, the *to_cpu()* function returns the instance of the agent before it was sent to GPU using the *to_gpu()* function. If a different behavior is expected, it can be defined explicitly by overwriting the *to_cpu()* function.
