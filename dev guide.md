RL-v2
=====

Architecture
------------

- There is a manager
  - Entry Point - has the __main__ thing.
  - Configures logger
  - Accepts Env and Algorithm
  - Runs the RL loop. Resets env, calls `onReset` of agents, calls `act` of agents, steps the env etc.
    - Call `post_act`
    - Call `post_episode`
    - etc.
- There are lots of agents. Can do whatever during their turn.
- Runner also does the following:
  - Configure log folder and root logger
  - Create env.
  - Wrap it with whatever.
  - Pass it to runner.
  - Create and register relevant agents to the runner (based on algorithm)
  - Start the runner.
  - A list of agents configured in some way.
- Each of these three pull their arguments from command line.
- Logging handled by python logging module. But who configures it?


## Sequence

1. Runner gets created with env_id, algo_id
2. In runner constructor
   1. env is created using gym factory.
   2. algo is created using algo factory (but algo agents not created yet). Algo given access to runner.
   3. env is wrapped using algo
   4. algo told to create



Principles:
- One instance of the program is always assumed to belong to one algo run
  - Multiple instances of algo using multiple instances of python.
- The library **must always** be used using `python -m RL env_id algo_id`
- **Really avoid running an algo without proving an algo suffix. And really avoid running an algo again with same suffix using the `--overwrite` flag**
- logger always used using `logging.getLogger(__name__)`.
  - Any object specific info can be explicitly included in the log statement.

Logging norms:
- print and logging separate. only WARNINGs+ should go to stdout as well.
- INFO level for episode granularity events
- DEBUG level for frame granularity events
- Both of the above should not go to console
- WARNING for reporting potentially unintented behaviour or discouraged behaviour, but non-lethal behaviour
- ERROR for code crash behaviour

- When logging from inside a function foo, use language like 'doing foo'.
  - Typically log at top of the function.
- When logging just before calling foo, log 'calling foo'.
- When logging after a function call or from inside the function at the end, log 'done foo'.
  - Log from outside or inside? Anywhere it is guaranteed that the log will happen.
  - Calling from inside might be risky because functions get overriden.


Todo:
- [X] Make sure all exceptions, errors and warnings logged properly in core.
- [X] Printing basics to console? An agent or hard coded?
- [] Parallel envs code might not be done properly in core yet.
- [] All the TODO tags in the code


Housekeeping:
- [X] Video Record Unwrapped - can couple with renderer .. or monitor?
  - Let's do monitor because can upload monitor logs to gym site then. Also, manager records the stats of the wrapped env. Might need standardized (by openai) raw stats of unwrapped env.
- [ ] Video Record Obs
  - how? 
- [ ] Other Wrappers can record as required