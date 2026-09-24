import sys

from gradysim.simulator.handler.communication import CommunicationHandler, CommunicationMedium
from gradysim.simulator.handler.ardupilot_mobility import ArdupilotMobilityHandler, ArdupilotMobilityConfiguration
from gradysim.simulator.handler.timer import TimerHandler
from gradysim.simulator.handler.visualization import VisualizationHandler
from gradysim.simulator.simulation import SimulationBuilder, SimulationConfiguration
from protocol_sensor import SimpleProtocolSensor
from protocol_mobile import SimpleProtocolMobile
from protocol_ground import SimpleProtocolGround


def run_simulation(ardupilot_path: str):
    builder = SimulationBuilder(SimulationConfiguration(
        duration=180,
        real_time=True
    ))

    builder.add_handler(CommunicationHandler(CommunicationMedium(transmission_range=30)))
    builder.add_handler(TimerHandler())
    builder.add_handler(ArdupilotMobilityHandler(ArdupilotMobilityConfiguration(
        simulate_drones=True,
        ardupilot_path=ardupilot_path,
        starting_api_port=8000,
        update_rate=0.5,
        default_speed=10,
        generate_report=True
    )))
    builder.add_handler(VisualizationHandler())

    # Ground station at origin
    builder.add_node(SimpleProtocolGround, (0, 0, 20))

    # Sensor located away from ground station
    builder.add_node(SimpleProtocolSensor, (100, 0, 20))

    # Drone that flies between ground station and sensor
    builder.add_node(SimpleProtocolMobile, (0, 0, 20))

    simulation = builder.build()
    simulation.start_simulation()


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python main.py <path_to_ardupilot_repository>")
        sys.exit(1)

    run_simulation(sys.argv[1])
