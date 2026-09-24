# This is an example file to test the RAFT protocol
# It creates 40 nodes at random positions on the ground level
# and executes consensus among them


# Import the necessary general libraries
import sys
import random
from pathlib import Path

# Ensure project root is on sys.path when running as a script
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Import the necessary Gradysim libraries, the protocol of the nodes and other necessary libraries
from gradysim.simulator.handler.communication import CommunicationHandler, CommunicationMedium  # noqa: E402
from gradysim.simulator.handler.mobility import MobilityHandler, MobilityConfiguration  # noqa: E402
from gradysim.simulator.handler.timer import TimerHandler  # noqa: E402
from gradysim.simulator.handler.visualization import VisualizationHandler, VisualizationConfiguration  # noqa: E402
from gradysim.simulator.simulation import SimulationBuilder, SimulationConfiguration  # noqa: E402
try:
    from .protocol import RaftProtocol  # noqa: E402
except ImportError:
    from protocol import RaftProtocol  # noqa: E402

# Main function to execute the simulation
def main():

    # Simulation parameters
    duration = 5 # Simulation duration in seconds
    debug = False # Simulation debug mode
    real_time = True # Simulation real time mode
    builder = SimulationBuilder(SimulationConfiguration(duration=duration, debug=debug, real_time=real_time))

    # Add the communication handler
    transmission_range = 200 # Communication transmission range in meters
    delay = 0.0 # Communication delay in seconds
    failure_rate = 0.0 # Communication failure rate in percentage
    medium = CommunicationMedium(transmission_range=transmission_range, delay=delay, failure_rate=failure_rate)
    builder.add_handler(CommunicationHandler(medium))

    # Add the timer handler
    builder.add_handler(TimerHandler())

    # Add the mobility handler
    update_rate = 0.01 # Mobility update rate in seconds
    builder.add_handler(MobilityHandler(MobilityConfiguration(update_rate=update_rate)))

    # Add the visualization handler
    open_browser = True # Visualization open browser - True: open browser, False: do not open browser
    builder.add_handler(VisualizationHandler(VisualizationConfiguration(open_browser=open_browser, )))

    # Add 40 nodes at random positions between -50 and +50
    num_nodes = 40
    for i in range(num_nodes):
        x = random.uniform(-50, 50)
        y = random.uniform(-50, 50)
        builder.add_node(RaftProtocol, (x, y, 0))

    simulation = builder.build()
    simulation.start_simulation()


if __name__ == "__main__":
    main() 