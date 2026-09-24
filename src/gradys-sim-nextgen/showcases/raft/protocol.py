import logging

from gradysim.protocol.interface import IProtocol
from gradysim.protocol.messages.telemetry import Telemetry
from gradysim.simulator.extension.visualization_controller import VisualizationController
from gradysim.protocol.plugin.raft import RaftConfig, RaftMode, RaftConsensusPlugin, RaftState


class RaftProtocol(IProtocol):
    def initialize(self):
        """Configure the Raft consensus plugin and schedule showcase timers."""
        self.counter = 0
        self.node_id = self.provider.get_id()

        # Schedule application timers using absolute simulation time
        now = self.provider.current_time()
        self.provider.schedule_timer("counter_timer", now + 1.0)
        self.provider.schedule_timer("failure_simulation_timer", now + 2.0)
        self.provider.schedule_timer("recovery_simulation_timer", now + 4.0)

        self.visualization_controller = VisualizationController(self)
        self.visualization_controller.paint_node(self.node_id, (0, 0, 255))

        # Build the consensus configuration
        config = RaftConfig()
        config.set_election_timeout(150, 300)  # milliseconds
        config.set_heartbeat_interval(50)      # milliseconds
        config.add_consensus_variable("v_int", int)
        config.set_logging(enable=True, level="INFO")

        # For instability tests you can uncomment the line below to fall back to classic mode
        # config.set_raft_mode(RaftMode.CLASSIC)
        config.set_raft_mode(RaftMode.FAULT_TOLERANT)

        # Configure failure detection parameters when running in fault tolerant mode
        failure_config = config.get_failure_config()
        failure_config.set_failure_threshold(2)
        failure_config.set_recovery_threshold(3)
        failure_config.set_detection_interval(2)
        failure_config.set_heartbeat_timeout(4)  # multiplier of heartbeat interval

        # Initialize the Raft consensus plugin
        self.consensus = RaftConsensusPlugin(
            config=config,
            protocol=self
        )

        # Set known nodes and start consensus
        total_nodes = 40
        known_nodes = list(range(total_nodes))
        self.consensus.set_known_nodes(known_nodes)
        self.consensus.start()

    def handle_timer(self, timer: str):
        # Handle user specific timers
        if timer == "counter_timer":
            self.counter += 1
            self.provider.schedule_timer(
                "counter_timer",
                self.provider.current_time() + 1.0
            )

        if timer == "failure_simulation_timer":
            inactive_nodes = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
            for node in inactive_nodes:
                self.consensus.set_simulation_active(node, False)

        if timer == "recovery_simulation_timer":
            recovery_nodes = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
            for node in recovery_nodes:
                self.consensus.set_simulation_active(node, True)

    def handle_packet(self, message: str):
        pass

    def handle_telemetry(self, telemetry: Telemetry) -> None:
        # If this node is leader, propose a value occasionally
        if self.consensus and self.consensus.is_leader():
            if self.counter % 2 == 0:  # Every two iterations
                self.consensus.propose_value("v_int", 1)

        # Update visualization and log node state for debugging
        current_state = self.consensus.get_current_state()
        if current_state == RaftState.LEADER:
            color = (0, 255, 0)  # Green leader
            state_label = "LEADER"
        elif current_state == RaftState.CANDIDATE:
            color = (255, 255, 0)  # Yellow candidate
            state_label = "CANDIDATE"
        else:
            if self.consensus.is_simulation_active(self.node_id):
                color = (0, 0, 255)  # Blue active follower
                state_label = "ACTIVE FOLLOWER"
            else:
                color = (255, 0, 0)  # Red inactive
                state_label = "INACTIVE"

        self.visualization_controller.paint_node(self.node_id, color)
        print(f"Node {self.node_id} is {state_label}")

    def finish(self):
        logging.info(f"Counter: {self.counter}")

        if self.consensus:
            self.consensus.stop()
