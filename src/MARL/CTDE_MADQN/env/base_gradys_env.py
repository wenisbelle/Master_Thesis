"""
Stepping of a GrADyS-SIM NextGen simulation for a reinforcement learning environment.

The simulation runs event by event, the same way run_stepped_simulation of
Analytical/swarm/main.py does it: nothing is added to the event loop. That
matters, gradysim orders the events of the same instant by their position in
the heap, so an extra event (the old version scheduled one every iteration)
changes the order of the others and the run diverges from the analytical one.

An environment step (advance) stops:
    - right after an event that asked for a decision (decision_requests not empty).
      The environment applies the actions before the next event, so the decision
      takes no simulated time, as in the analytical protocol where it is taken
      inside the event;
    - when the simulated time reaches the next multiple of
      algorithm_iteration_interval, so the rewards keep flowing between decisions;
    - when the simulation ends.

The swarm is sampled every sample_interval seconds, at the same instants as
run_stepped_simulation samples it (observers, see notify): once after the first
event, after the first event at or past every multiple of sample_interval, and
once more on the event that ends the simulation. A sample due while a decision
is pending is taken after the decision is applied, as in the analytical
protocol, where the decision is applied inside the event that precedes it.
"""

from dataclasses import dataclass
from abc import ABC, abstractmethod
from time import sleep
from gradysim.simulator.simulation import SimulationBuilder, Simulator, SimulationConfiguration


@dataclass
class SimulationStatus:
    has_ended: bool


class BaseGrADySEnvironment(ABC):
    """
    A generic TorchRL compatible environment for simulations using the GrADyS-SIM NextGen framework.
    This class serves as a base for creating custom environments that leverage the GrADyS-SIM NextGen simulation capabilities.
    """

    simulator: Simulator

    def __init__(self, algorithm_iteration_interval: float, sample_interval: float = 1.0,
                 visual_mode: bool = False):
        """
        Args:
            algorithm_iteration_interval: longest environment step, in simulated seconds.
            sample_interval: simulated seconds between two samples of the swarm, see notify.
            visual_mode: the simulation runs in real time, for the gradysim visualization.
        """
        self._algorithm_iteration_interval = algorithm_iteration_interval
        self._sample_interval = sample_interval
        self.visual_mode = visual_mode
        ##### Filled by the protocols, see protocol.request_decision #####
        self.decision_requests = []
        self._observers = []
        self._pending_samples = []
        self._ended = True

    @abstractmethod
    def _build_simulation(self, builder: SimulationBuilder):
        """
        Adds the handlers and the nodes to the builder.
        """
        pass

    def _make_observers(self) -> list:
        """Callables observe(time) notified on every sample, see notify. Built after the simulation."""
        return []

    def reset_simulation(self, simulation_configuration: SimulationConfiguration):
        """
        Builds a new simulation and runs its first step, which initializes every protocol
        and processes the first event. It is followed by the first sample.
        """
        self.finalize_simulation()
        self.decision_requests.clear()
        builder = SimulationBuilder(simulation_configuration)
        self._build_simulation(builder)
        self.simulator = builder.build()
        self._observers = self._make_observers()

        running = self.simulator.step_simulation()
        self._ended = not running
        self._pending_samples = [self.simulator._current_timestamp]
        self._next_sample = self._sample_interval
        self._next_iteration = self._algorithm_iteration_interval
        if not running:
            self.decision_requests.clear()

    @property
    def current_time(self) -> float:
        return self.simulator._current_timestamp

    def _notify(self) -> None:
        """Takes the samples that are due, see the module docstring."""
        while self._pending_samples:
            time = self._pending_samples.pop(0)
            for observe in self._observers:
                observe(time)

    def step_simulation(self) -> SimulationStatus:
        """
        Advances the simulation until a decision is asked for, the next iteration
        instant, or the end. See the module docstring.
        """
        while True:
            if self.decision_requests:
                return SimulationStatus(has_ended=False)
            self._notify()
            if self._ended:
                return SimulationStatus(has_ended=True)
            if self.simulator._current_timestamp >= self._next_iteration:
                while self._next_iteration <= self.simulator._current_timestamp:
                    self._next_iteration += self._algorithm_iteration_interval
                return SimulationStatus(has_ended=False)

            if self.visual_mode:
                next_event = self.simulator._event_loop.peek_event()
                if next_event is not None:
                    time_until_next_event = next_event.timestamp - self.simulator._current_timestamp
                    if time_until_next_event > 0:
                        sleep(time_until_next_event)

            running = self.simulator.step_simulation()
            time = self.simulator._current_timestamp
            if not running:
                ##### As run_stepped_simulation: the event that ends the run is sampled once, #####
                ##### on its own time, and the decision it may have asked for is dropped.    #####
                self._ended = True
                self.decision_requests.clear()
                self._pending_samples.append(time)
                continue
            while self._next_sample <= time:
                self._pending_samples.append(self._next_sample)
                self._next_sample += self._sample_interval

    def finalize_simulation(self):
        """
        Finalizes the simulation and releases any resources.
        """
        if hasattr(self, "simulator"):
            self.simulator._finalize_simulation()
