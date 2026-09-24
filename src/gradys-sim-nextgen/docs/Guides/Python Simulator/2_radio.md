# Multi-radio systems (Radio Extension)

This guide shows how to use the Radio extension in prototype-mode to override 
communication medium settings (range, delay, failure rate) on a per-message basis.
It complements the standard 
[CommunicationHandler][gradysim.simulator.handler.communication.CommunicationHandler] 
by letting a protocol use multiple radios with distinct characteristics.

The guide will follow an example protocol that uses the radio extension to 
simulate a node with two radios: a short-range radio and a long-range radio.

## What the Radio does

- Creates one or more logical radios inside a protocol
- Each radio has its own [CommunicationMedium][gradysim.simulator.handler.communication.CommunicationMedium]
  - transmission_range (m)
  - delay (s)
  - failure_rate (0..1)
- Messages sent through a radio use that radio’s configuration, other messages
  are unaffected
- Only available in the Python simulator; requires a [CommunicationHandler][gradysim.simulator.handler.communication.CommunicationHandler]

[API reference][gradysim.simulator.extension.radio.Radio]

## Instantiation

Create Radio objects in `initialize()` or later. The `initialize` method marks
the simulation's initialization. Radios instantiated before
the simulation starts raise an error.

```py title="Protocol.initialize: create radios"
--8<--
docs/Guides/radio example/radio_protocol.py:27:30
--8<--
```

!!!warning
    Extensions only work in the Python simulator. Instantiating a Radio in 
    other environments raises an error.

## Configuring

Each Radio's configuration starts with the handler’s default medium 
(same range, delay and failure as the global 
[CommunicationHandler][gradysim.simulator.handler.communication.CommunicationHandler]). 
You can then override only the fields you need by calling `.set_configuration`. 
The parameters that `set_configuration` accepts are the same as those of the
[CommunicationMedium][gradysim.simulator.handler.communication.CommunicationMedium].
All parameters are optional, with unspecified ones retaining their previous value.

In this example we're configuring two radios with different ranges, one with
a short range (10 m) and one with a long range (100 m). 

```py title="Per-radio range overrides"
--8<--
docs/Guides/radio example/radio_protocol.py:32:33
--8<--
```

## Sending messages

Messages sent through a Radio use that radio’s medium:

```py title="Broadcast via short-range radio"
--8<--
docs/Guides/radio example/radio_protocol.py:37:43
--8<--
```

```py title="Unicast via long-range radio"
--8<--
docs/Guides/radio example/radio_protocol.py:45:52
--8<--
```

Messages sent through the provider are unaffected by radios and use the 
handler’s default medium:

```py title="Provider send (unaffected by radios)"
--8<--
docs/Guides/radio example/radio_protocol.py:54:61
--8<--
```

In this example we send three messages:
- A broadcast via the short-range radio (10 m)
- A unicast via the long-range radio (100 m)
- A unicast via the provider (default range 60 m)

Three nodes are placed such that:
- Node 1 is within 10 m of Node 0
- Node 2 is within 100 m of Node 0, but outside 10 m

The expected result in this scenario is that Node 1 receives all three messages,
while Node 2 only receives the long-range and provider messages. This example
illustrates how protocols can use multiple radios with different configurations 
to simulate multi-radio systems with different capabilities.

We chose to only override the transmission range in this example, but you can 
also override delay and failure rate per-radio if needed, or any other field
of the [CommunicationMedium][gradysim.simulator.handler.communication.CommunicationMedium].

As expected, running the example yields the following output:

```plaintext
INFO:root:[--------- Simulation started ---------]
INFO     [--------- Simulation started ---------]
INFO:root:Node 0 received: []
INFO     [it=3 time=0:00:00 | RadioProtocol 0 Finalization] Node 0 received: []
INFO:root:Node 1 received: ['hello_short']
INFO     [it=3 time=0:00:00 | RadioProtocol 1 Finalization] Node 1 received: ['hello_short']
INFO:root:Node 2 received: ['ping_long', 'via_provider']
INFO     [it=3 time=0:00:00 | RadioProtocol 2 Finalization] Node 2 received: ['ping_long', 'via_provider']
INFO:root:[--------- Simulation finished ---------]
INFO     [--------- Simulation finished ---------]
INFO:root:Real time elapsed: 0:00:00	Total iterations: 3	Simulation time: 0:00:00
INFO     Real time elapsed: 0:00:00	Total iterations: 3	Simulation time: 0:00:00
```

## Full example

??? example "Full protocol code"
    ```py
    --8<--
    docs/Guides/radio example/radio_protocol.py
    --8<--
    ```

??? example "Full execution code"
    ```py
    --8<--
    docs/Guides/radio example/main.py
    --8<--
    ```