import unittest

from gradysim.simulator.event import EventLoop
from gradysim.protocol.messages.communication import CommunicationCommand, CommunicationCommandType
from gradysim.simulator.node import Node
from gradysim.simulator.handler.communication import CommunicationMedium, CommunicationHandler, \
    CommunicationException, can_transmit


def handle_command_helper(command: CommunicationCommand):
    medium = CommunicationMedium(transmission_range=10)

    received = 0

    class DummyEncapsulator:
        def handle_packet(self, _message: dict):
            nonlocal received
            received += 1

    node1 = Node()
    node1.id = 1
    node1.position = (0, 0, 0)
    node1.protocol_encapsulator = DummyEncapsulator()

    node2 = Node()
    node2.id = 2
    node2.position = (5, 5, 5)
    node2.protocol_encapsulator = DummyEncapsulator()

    comm_handler = CommunicationHandler(medium)
    event_loop = EventLoop()
    comm_handler.inject(event_loop)

    comm_handler.register_node(node1)
    comm_handler.register_node(node2)

    # Testing successful command
    comm_handler.handle_command(command, node1)

    event_loop.pop_event().callback()
    return received


class TestCommunication(unittest.TestCase):
    def test_transmission_range(self):
        medium = CommunicationMedium(transmission_range=10)
        # Exact boundary
        self.assertTrue(can_transmit((0, 0, 0), (10, 0, 0), medium))
        # Same position
        self.assertTrue(can_transmit((0, 0, 0), (0, 0, 0), medium))
        # Inside radius
        self.assertTrue(can_transmit((0, 0, 0), (-3, -3, 0), medium))

        # Just outside boundary
        self.assertFalse(can_transmit((0, 0, 0), (10.001, 0, 0), medium))
        # Far apart
        self.assertFalse(can_transmit((10, 0, 0), (-10, 0, 0), medium))
        self.assertFalse(can_transmit((0, 0, 0), (30, 0, 0), medium))
        self.assertFalse(can_transmit((0, 0, 0), (8, 8, 8), medium))

    def test_successful_send_command(self):
        command = CommunicationCommand(
            CommunicationCommandType.SEND,
            "",
            2
        )
        received = handle_command_helper(command)
        self.assertEqual(received, 1)

    def test_successful_broadcast_command(self):
        command = CommunicationCommand(
            CommunicationCommandType.BROADCAST,
            ""
        )
        received = handle_command_helper(command)
        self.assertEqual(received, 1)

    def test_failure_no_destination_send(self):
        command = CommunicationCommand(
            CommunicationCommandType.SEND,
            ""
        )
        self.assertRaises(CommunicationException, handle_command_helper, command)

    def test_failure_self_message(self):
        command = CommunicationCommand(
            CommunicationCommandType.SEND,
            "",
            1
        )
        self.assertRaises(CommunicationException, handle_command_helper, command)

    def test_failure_non_existing_target(self):
        command = CommunicationCommand(
            CommunicationCommandType.SEND,
            "",
            10
        )
        self.assertRaises(CommunicationException, handle_command_helper, command)

    def test_register_not_injected(self):
        node1 = Node()
        node1.id = 1
        node1.position = (0, 0, 0)

        medium = CommunicationMedium()
        comm_handler = CommunicationHandler(medium)

        with self.assertRaises(CommunicationException):
            comm_handler.register_node(node1)

    def test_delay(self):
        # Setting up nodes
        received = 0
        class DummyEncapsulator:
            def handle_packet(self, _message: dict):
                nonlocal received
                received += 1

        node1 = Node()
        node1.id = 1
        node1.position = (0, 0, 0)
        node1.protocol_encapsulator = DummyEncapsulator()

        node2 = Node()
        node2.id = 2
        node2.position = (5, 5, 5)
        node2.protocol_encapsulator = DummyEncapsulator()

        # Setting up comm_handler
        delay = 1
        medium = CommunicationMedium(delay=delay)
        event_loop = EventLoop()
        comm_handler = CommunicationHandler(medium)
        comm_handler.inject(event_loop)

        comm_handler.register_node(node1)
        comm_handler.register_node(node2)

        # Sending message
        command = CommunicationCommand(
            CommunicationCommandType.SEND,
            "",
            2
        )
        comm_handler.handle_command(command, node1)

        # Assert that event has been queued
        self.assertEqual(len(event_loop), 1)

        # Assert that timestamp is consistent with delay
        current_time = event_loop.current_time
        event = event_loop.pop_event()
        self.assertEqual(event.timestamp, current_time + delay)

        # Assert that message is received
        event.callback()
        self.assertEqual(received, 1)

    def test_failure(self):
        # Failure rate 0: should transmit
        medium = CommunicationMedium(failure_rate=0)
        self.assertTrue(can_transmit((0, 0, 0), (0, 0, 0), medium))

        # Failure rate 1: should never transmit
        medium = CommunicationMedium(failure_rate=1)
        self.assertFalse(can_transmit((0, 0, 0), (0, 0, 0), medium))

    def test_override_failure_rate(self):
        # Default medium: no failures
        default_medium = CommunicationMedium(failure_rate=0)
        handler = CommunicationHandler(default_medium)
        event_loop = EventLoop()
        handler.inject(event_loop)

        received = 0
        class DummyEncapsulator:
            def handle_packet(self, _message: dict):
                nonlocal received
                received += 1

        node1 = Node()
        node1.id = 1
        node1.position = (0, 0, 0)
        node1.protocol_encapsulator = DummyEncapsulator()

        node2 = Node()
        node2.id = 2
        node2.position = (0, 0, 0)
        node2.protocol_encapsulator = DummyEncapsulator()

        handler.register_node(node1)
        handler.register_node(node2)

        command = CommunicationCommand(
            CommunicationCommandType.SEND,
            "",
            2
        )

        # By default should be able to transmit
        handler.handle_command(command, node1)
        self.assertEqual(len(event_loop), 1)
        event_loop.pop_event().callback()
        self.assertEqual(received, 1)

        # Override to always fail for this command
        handler.handle_command(command, node1, medium=CommunicationMedium(failure_rate=1))
        self.assertEqual(len(event_loop), 0)
        self.assertEqual(received, 1)

    def test_override_delay(self):
        # Setup nodes and received counter
        received = 0

        class DummyEncapsulator:
            def handle_packet(self, _message: dict):
                nonlocal received
                received += 1

        node1 = Node()
        node1.id = 1
        node1.position = (0, 0, 0)
        node1.protocol_encapsulator = DummyEncapsulator()

        node2 = Node()
        node2.id = 2
        node2.position = (1, 0, 0)
        node2.protocol_encapsulator = DummyEncapsulator()

        # Default medium has no delay
        default_medium = CommunicationMedium(delay=0)
        comm_handler = CommunicationHandler(default_medium)
        event_loop = EventLoop()
        comm_handler.inject(event_loop)

        comm_handler.register_node(node1)
        comm_handler.register_node(node2)

        # Override delay for this command
        delay = 1.5

        # Send message from node1 to node2 with override
        command = CommunicationCommand(
            CommunicationCommandType.SEND,
            "",
            2
        )
        comm_handler.handle_command(command, node1, medium=CommunicationMedium(delay=delay))

        # An event should be queued
        self.assertEqual(len(event_loop), 1)

        current_time = event_loop.current_time
        event = event_loop.pop_event()
        self.assertEqual(event.timestamp, current_time + delay)

        # Execute event and assert message received
        event.callback()
        self.assertEqual(received, 1)

    def test_broadcast_range_filtering(self):
        # Two recipients: one in range, one out of range
        received2 = 0
        received3 = 0

        class Enc2:
            def handle_packet(self, _message: dict):
                nonlocal received2
                received2 += 1
        class Enc3:
            def handle_packet(self, _message: dict):
                nonlocal received3
                received3 += 1

        node1 = Node()
        node1.id = 1
        node1.position = (0, 0, 0)
        node2 = Node()
        node2.id = 2
        node2.position = (5, 0, 0)
        node2.protocol_encapsulator = Enc2()
        node3 = Node()
        node3.id = 3
        node3.position = (15, 0, 0)
        node3.protocol_encapsulator = Enc3()

        medium = CommunicationMedium(transmission_range=10, delay=0)
        event_loop = EventLoop()
        handler = CommunicationHandler(medium)
        handler.inject(event_loop)
        handler.register_node(node1)
        handler.register_node(node2)
        handler.register_node(node3)

        command = CommunicationCommand(CommunicationCommandType.BROADCAST, "msg")
        handler.handle_command(command, node1)

        # Only node2 should be scheduled
        self.assertEqual(len(event_loop), 1)
        event_loop.pop_event().callback()
        self.assertEqual(received2, 1)
        self.assertEqual(received3, 0)

    def test_broadcast_delay_applies_to_all_in_range(self):
        # Both recipients in range; both should get events with same delay
        received2 = 0
        received3 = 0

        class Enc2:
            def handle_packet(self, _message: dict):
                nonlocal received2
                received2 += 1
        class Enc3:
            def handle_packet(self, _message: dict):
                nonlocal received3
                received3 += 1

        node1 = Node()
        node1.id = 1
        node1.position = (0, 0, 0)
        node2 = Node()
        node2.id = 2
        node2.position = (5, 0, 0)
        node2.protocol_encapsulator = Enc2()
        node3 = Node()
        node3.id = 3
        node3.position = (8, 0, 0)
        node3.protocol_encapsulator = Enc3()

        delay = 1.2
        medium = CommunicationMedium(transmission_range=20, delay=delay)
        event_loop = EventLoop()
        handler = CommunicationHandler(medium)
        handler.inject(event_loop)
        handler.register_node(node1)
        handler.register_node(node2)
        handler.register_node(node3)

        command = CommunicationCommand(CommunicationCommandType.BROADCAST, "msg")
        handler.handle_command(command, node1)

        # Two events should be queued with the same timestamp
        self.assertEqual(len(event_loop), 2)
        t0 = event_loop.current_time
        e1 = event_loop.pop_event()
        e2 = event_loop.pop_event()
        self.assertEqual(e1.timestamp, t0 + delay)
        self.assertEqual(e2.timestamp, t0 + delay)

        # Execute to ensure messages are received
        e1.callback()
        e2.callback()
        self.assertEqual(received2, 1)
        self.assertEqual(received3, 1)

    def test_per_command_medium_override_range(self):
        # Default range too small; override per-command to succeed
        received = 0
        class Enc:
            def handle_packet(self, _message: dict):
                nonlocal received
                received += 1
        node1 = Node()
        node1.id = 1
        node1.position = (0, 0, 0)
        node2 = Node()
        node2.id = 2
        node2.position = (8, 0, 0)
        node2.protocol_encapsulator = Enc()

        default = CommunicationMedium(transmission_range=5)
        event_loop = EventLoop()
        handler = CommunicationHandler(default)
        handler.inject(event_loop)
        handler.register_node(node1)
        handler.register_node(node2)

        cmd = CommunicationCommand(CommunicationCommandType.SEND, "", 2)
        handler.handle_command(cmd, node1)  # should not schedule
        self.assertEqual(len(event_loop), 0)

        handler.handle_command(cmd, node1, medium=CommunicationMedium(transmission_range=10))
        self.assertEqual(len(event_loop), 1)
        event_loop.pop_event().callback()
        self.assertEqual(received, 1)

    def test_negative_delay_schedules_immediately(self):
        received = 0
        class Enc:
            def handle_packet(self, _message: dict):
                nonlocal received
                received += 1
        node1 = Node()
        node1.id = 1
        node1.position = (0, 0, 0)
        node2 = Node()
        node2.id = 2
        node2.position = (0, 0, 0)
        node2.protocol_encapsulator = Enc()

        event_loop = EventLoop()
        handler = CommunicationHandler(CommunicationMedium(delay=-1))
        handler.inject(event_loop)
        handler.register_node(node1)
        handler.register_node(node2)
        cmd = CommunicationCommand(CommunicationCommandType.SEND, "", 2)
        handler.handle_command(cmd, node1)
        self.assertEqual(len(event_loop), 1)
        e = event_loop.pop_event()
        self.assertEqual(e.timestamp, event_loop.current_time)
        e.callback()
        self.assertEqual(received, 1)

