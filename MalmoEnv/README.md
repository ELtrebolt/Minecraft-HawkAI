# Minecraft HawkAI

## MalmoEnv

MalmoEnv is an OpenAI "gym" like Python Environment for Malmo/Minecraft, directly implemented Python to Java Minecraft.

A python "gym env" is created and used to run an agent in a Malmo mission. Each env has a remote Minecraft instance
associated to it (by DNS name or IP and Port). For multi-agent missions, the first agent's (role 0) Minecraft
client instance is used as a coordinator to allow all agents to rendezvous on mission starts (i.e. on env resets).

As it's python only, you just need this one package, its direct dependencies and (Java) Minecraft!

## Files Overview

The following are the files within the codebase that we have created/modified:

- `custom_env/custom_env.py` - defines a CustomEnv class that makes the MalmoEnv gym-like environment compatible with Stable Baselines3 RL Modeling
- `custom_env/constants.py` - defines all the constants used for setting up the custom environment
- `custom_env/info_parser.py` - defines the parser class that parses information on the states and interactions within the environment
- `missions/hawkAI.xml` - the MalmoEnv mission file that initializes the Minecraft world that the agent interacts with
- `runmultiagent.py` - the main file that runs the simulations, builds the DQN model, and writes logs for evaluation

## Setting up MalmoEnv and Running the Agent

Install dependencies:

Java8 JDK, python3, git

In the root directory: `pip install -r requirements.txt`

To prepare Minecraft:

`cd Minecraft`

`(echo -n "malmo.version=" && cat ../VERSION) > ./src/main/resources/version.properties`

Running the agent in our custom mission environment (run each command in different cmd prompt/shells):

`./launchClient.sh -port 9000 -env` or (On Windows) `launchClient.bat -port 9000 -env`

(In another shell) `cd MalmoEnv` optionally run `python3 setup.py install`

`python3 runmultiagent.py --mission missions/hawkAI.xml --port 9000`

## Final Model Simulation

![](https://github.com/ELtrebolt/Minecraft-HawkAI/blob/main/MalmoEnv/gifs/successful_run.gif)
