In this page, we will describe how to create an [Agent](reference/agent.md) by describing the various parts required to create a valid agent.


## The Agent Concept

The concept of an agent within the olfactory navigation framework is a an abstract concept. It should be able to perform two things:

1. [Be Trained](#training)
2. [Be Tested](#testing)

These two steps will be defined below.

Fundamentally, to create an agent, the user has to create a subclass of the generic [Agent](reference/agent.md) class. Along with this, the various functions need to be overwritten as needed.


## Attributes

The agent has already a set of attributes by default:

- environment: The [Environment](reference/environment.md) the agent should be trained on
- name: By default the class name
- threshold: The olfactory sensitivity of the agent

Additionally, other attributes can be defined within the init function. Those attribute should be of one of three categories:

- Static: Those attribute should be there to help within the functions of the agent.
- Trainable: Those attribute typically should be set to *None* within the init and set to something after the *train()* function has been run.
- Status: Those attributes are dynamic variables that will change over the course of the simulation. They should reprensent the status of the agent during the simulation (for example, where the agent believes he is in the space, or an internal clock, or even a memory).


## Training

For the training part, the **concept** of the agent is trained. We define the brain of the agent.

For this, the function ['Agent.train()'](reference/agent.md#olfactory_navigation.agent.Agent.train) needs to be defined (overwritten).


## Testing

For the testing, the agent needs to be able to interact be interacted with from the *run_test()* function of the [simulation](reference/simulation.md) module.

For this the following functions need to be implemented:

- initialize_state()
- choose_action()
- update_state(observation, source_reached)
- kill()

## Saving

The agent must be able to be saved to memory for reproducability's sake. Related functions

By default, the agent can be saved as a [pickle](https://realpython.com/python-pickle-module/) file. But it is remended to define your own *save()* and *load()* functions as a pickle file is a blackbox and may save too many attributes, so not very space efficient. For instance, for the agent to be re-used later, only the **Trainable** variables are needed.


## GPU support

To speedup large operations, the 
