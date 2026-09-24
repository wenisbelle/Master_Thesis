# RAFT Consensus Plugin

The RAFT consensus plugin ships with GrADyS-SIM NextGen and provides a fault-tolerant coordination layer for protocols that need to agree on shared values. This guide walks through the essentials for configuring the plugin, wiring it into a protocol, and validating the showcase scenario.

## Key Capabilities

- **Fault-Tolerant Consensus**: Robust leader election with automatic recovery
- **Active Node Discovery**: Dynamic cluster size detection for accurate majority calculations
- **Heartbeat-Based Failure Detection**: Automatic detection of failed nodes
- **Massive Failure Recovery**: Recovers from scenarios where massive quantity of nodes fail
- **Scalable Architecture**: Works with any number of nodes (10, 50, 100+)
- **GradySim Integration**: Seamless integration with GrADyS-SIM NextGen simulation framework
- **No Log Replication**: Lightweight implementation focused on consensus values
- **Dynamic Majority Calculation**: Thresholds adapt to actual active nodes
- **Dual Operation Modes**: Classic Raft and Fault-Tolerant modes
- **Dispatcher Integration**: Automatic handling of RAFT-specific packets and timers
- **Optional Statistics**: State inspection helpers for debugging

## Quick Start

### Basic Setup

```python
from gradysim.protocol.plugin.raft import RaftConfig, RaftMode, RaftConsensusPlugin

class RaftProtocol(IProtocol):
    def initialize(self) -> None:
        # Configure consensus
        config = RaftConfig()
        config.set_election_timeout(150, 300)  # 150-300ms election timeout
        config.set_heartbeat_interval(50)      # 50ms heartbeat interval
        config.add_consensus_variable("sequence", int)
        config.add_consensus_variable("leader_position", str)
        config.set_logging(enable=True, level="INFO")

        # Choose operation mode
        config.set_raft_mode(RaftMode.FAULT_TOLERANT)  # Default: fault-tolerant mode
        # config.set_raft_mode(RaftMode.CLASSIC)      # Alternative: classic mode

        # Initialize the Raft consensus plugin
        self.consensus = RaftConsensusPlugin(config=config, protocol=self)
        
        # Set known nodes and start consensus
        self.consensus.set_known_nodes(list(range(10)))
        self.consensus.start()

    def handle_timer(self, timer: str) -> None:
        # Non RAFT timers continue to arrive here
        if timer == "app_timer":
            self.provider.schedule_timer("app_timer", self.provider.current_time() + 1)

    def handle_telemetry(self, telemetry: Telemetry) -> None:
        # Propose values if leader
        if self.consensus.is_leader():
            self.consensus.propose_value("sequence", int(telemetry.simulation_time))
            self.consensus.propose_value("leader_position", "north")

    def finish(self) -> None:
        self.consensus.stop()
```

### Advanced Usage with Failure Detection

```python
def initialize(self) -> None:
    config = RaftConfig()
    config.set_election_timeout(150, 300)
    config.set_heartbeat_interval(50)
    config.add_consensus_variable("sequence", int)
    config.set_raft_mode(RaftMode.FAULT_TOLERANT)

    # Configure failure detection parameters
    failure_config = config.get_failure_config()
    failure_config.set_failure_threshold(2)      # 2 failed heartbeats to mark as failed
    failure_config.set_recovery_threshold(3)     # 3 successful heartbeats to recover
    failure_config.set_detection_interval(2)     # Check every 2 heartbeats
    failure_config.set_heartbeat_timeout(4)      # 4× heartbeat_interval = 200ms timeout

    self.consensus = RaftConsensusPlugin(config=config, protocol=self)
    self.consensus.set_known_nodes(list(range(10)))
    self.consensus.start()
```

### Dispatcher Integration

The plugin automatically prefixes every internal message and timer with `__RAFT__` when talking to the provider. Dispatcher hooks intercept any packet or timer that carries this prefix and stop it from reaching the protocol. All other events pass through unchanged, so protocol logic stays clean.

## Architecture Overview

### **Component Relationships**

```
   RaftConsensus           RaftNode            GradySim      
     (Facade)     ◄──►   (Core Logic)  ◄──►    (Platform)    
         │                     │                  │
         ▼                     ▼                  ▼
   RaftConfig          HeartbeatDetector       Simulation    
   (Settings)             (Internal)           Framework     
```

**Component Relationships:**

- **RaftConsensus**: Public facade for consensus operations

- **RaftNode**: Core Raft algorithm implementation

- **GradySim**: Simulation framework integration

- **RaftConfig**: Configuration and settings management

- **HeartbeatDetector**: Internal failure detection (Fault-Tolerant mode only)

### **Message Flow**

1. **Consensus Request**: `RaftConsensus.propose_value()`
2. **Internal Processing**: `RaftNode` processes the request
3. **Platform Communication**: Uses GradySim provider to send messages
4. **Response Handling**: Processes responses via dispatcher
5. **State Update**: Updates consensus state and notifies callers

### **Key Innovations**

#### **1. Active Node Discovery System**

The RAFT plugin introduces a **discovery phase** before elections that solves the fundamental problem of unknown cluster size after failures:

```python
# Traditional Raft: Uses fixed cluster size
majority = (total_nodes // 2) + 1  # Always uses total_nodes

# RAFT Plugin: Discovers actual active nodes
discovered_active = discover_active_nodes()  # Dynamic discovery
majority = (discovered_active // 2) + 1     # Uses actual active nodes
```

**Benefits:**

- **Accurate majority calculations** after failures

- **Faster leader election** with correct thresholds

- **Automatic adaptation** to cluster changes

- **No manual configuration** needed

#### **2. Massive Failure Recovery**

The plugin can recover from scenarios that would break traditional Raft implementations:

```python
# Scenario: 10 nodes, 8 fail (including leader)
# Traditional Raft: Stuck - can't calculate majority
# RAFT Plugin: Discovers 2 active nodes, calculates majority = 2, elects leader
```

#### **3. Dynamic Cluster Adaptation**

The system automatically adapts to cluster changes:

```python
# Initial: 10 nodes active
# After failures: 3 nodes active
# RAFT Plugin automatically adjusts majority threshold from 6 to 2
```

#### **4. Dispatcher Integration**

The plugin automatically handles RAFT-specific communication without affecting your protocol:

```python
# The plugin automatically prefixes internal messages and timers with "__RAFT__"
# Your protocol only sees non-RAFT events
def handle_timer(self, timer: str) -> None:
    # RAFT timers are automatically intercepted
    # Only your application timers reach here
    if timer == "app_timer":
        self.provider.schedule_timer("app_timer", self.provider.current_time() + 1)
```

## Configuring the Plugin

### Election and Heartbeat

- `set_election_timeout(min_ms, max_ms)`: 

random election timeout window. Typical values are 150-300 ms.

- `set_heartbeat_interval(interval_ms)`: 

heartbeat cadence. Keep it well below the election timeout (for example 50 ms).

### Consensus Variables

Declare everything that must reach agreement before starting the consensus:

```python
config.add_consensus_variable("sequence", int)
config.add_consensus_variable("formation_info", dict)
config.add_consensus_variable("leader_position", str)
config.add_consensus_variable("temperature", float)
config.add_consensus_variable("is_active", bool)
```

**Supported Variable Types:**

- **Primitive Types**: `int`, `float`, `str`, `bool`

- **Complex Types**: `dict`, `list`, `tuple`

- **None Values**: `None` is also supported

- **Custom Objects**: Any object that can be JSON serialized

Committed values are available through `get_committed_value(name)` or `get_all_committed_values()`.

### Operating Modes

The RAFT plugin supports two distinct operation modes, allowing you to choose between classic Raft behavior and fault-tolerant enhancements:

#### **Classic Raft Mode (`RaftMode.CLASSIC`)**

Classic Raft Mode implements the standard Raft protocol as specified by Ongaro and Ousterhout. The only divergence in our system is replication: the plugin targets agreement on consensus values rather than maintaining a replicated log.

**Characteristics:**

- **Fixed Cluster Size**: Uses total known nodes for all majority calculations

- **Direct Elections**: No discovery phase - immediate election on timeout

- **Standard Behavior**: Classic Raft consensus semantics

- **Better Performance**: Lower overhead (no discovery phase)

- **Deterministic**: Predictable behavior for static clusters

- **Manual Failure Handling**: Requires manual cluster reconfiguration after failures

**Use Cases:**

- Static clusters with fixed, known node count

- Environments where performance is critical

- Systems requiring standard Raft semantics

- Scenarios with manual failure management

#### **Fault-Tolerant Raft Mode (`RaftMode.FAULT_TOLERANT`)**

Enhanced Raft implementation with automatic fault tolerance and dynamic cluster adaptation.

**Characteristics:**

- **Dynamic Active Node Detection**: Discovery phase before elections

- **Adaptive Majority Calculation**: Uses active node count for thresholds

- **Massive Failure Recovery**: Automatic recovery from massive quantity of node failures

- **Self-Healing**: Automatic adaptation to cluster changes

- **Fault Tolerance**: No manual intervention needed

- **Active Node Information**: Real-time access to node information

- **Discovery Overhead**: Small performance cost for discovery phase

**Use Cases:**

- Dynamic environments with node failures

- Distributed systems requiring high availability

- Scenarios with unpredictable node failures

- Systems that need automatic fault recovery

#### **Mode Comparison**

| Feature | Classic Mode | Fault-Tolerant Mode |
|---------|--------------|---------------------|
| **Discovery Phase** | ❌ Disabled | ✅ Enabled |
| **Majority Calculation** | Fixed (total nodes) | Dynamic (active nodes) |
| **Failure Recovery** | ❌ Manual | ✅ Automatic |
| **Performance** | ✅ Faster | ⚡ Slight overhead |
| **Massive Failures** | ❌ Requires intervention | ✅ Auto-recovery |
| **Active Node Info** | ❌ Not available | ✅ Available |
| **Use Case** | Static clusters | Dynamic clusters |
| **Standard Raft** | ✅ Compliant | Enhanced |

#### **Configuration Examples**

**Classic Raft Mode Configuration:**
```python
config = RaftConfig()
config.set_raft_mode(RaftMode.CLASSIC)
config.set_election_timeout(150, 300)
config.set_heartbeat_interval(50)

# Classic mode: Uses total known nodes for majority
# No discovery phase - direct elections
# Standard Raft semantics
```

**Fault-Tolerant Raft Mode Configuration:**
```python
config = RaftConfig()
config.set_raft_mode(RaftMode.FAULT_TOLERANT)  # Default mode
config.set_election_timeout(200, 400)  # Longer timeouts for discovery
config.set_heartbeat_interval(50)

# Fault-tolerant mode: Uses active node discovery
# Dynamic majority calculations
# Automatic fault recovery
```

#### **Runtime Mode Detection**

```python
# Check current mode
if config.is_classic_mode():
    print("Running in Classic Raft mode")
    # No discovery, standard Raft behavior
elif config.is_fault_tolerant_mode():
    print("Running in Fault-Tolerant Raft mode")  
    # Discovery enabled, dynamic fault tolerance

# Get mode programmatically
current_mode = config.get_raft_mode()
print(f"Current mode: {current_mode.value}")
```

#### **Choosing the Right Mode**

**Choose Classic Mode When:**
- Cluster size is fixed and known
- Performance is critical
- Standard Raft behavior is required
- Manual failure management is acceptable

**Choose Fault-Tolerant Mode When:**
- Nodes may fail unexpectedly
- Automatic fault recovery is needed
- Cluster size varies dynamically
- High availability is required
- Active node information is needed

Switch modes with `config.set_raft_mode(RaftMode.CLASSIC)` or `config.set_raft_mode(RaftMode.FAULT_TOLERANT)`.

### Failure Detection

`config.get_failure_config()` exposes thresholds for the heartbeat detector used in fault tolerant mode:

```python
failure = config.get_failure_config()
failure.set_failure_threshold(2)      # 2 failed heartbeats to mark as failed
failure.set_recovery_threshold(3)     # 3 successful heartbeats to recover
failure.set_detection_interval(2)     # Check every 2 heartbeats
failure.set_heartbeat_timeout(4)      # 4 * heartbeat interval
```

**Failure Detection Configuration Options:**

```python
# Conservative Detection (High reliability, slower detection)
failure.set_failure_threshold(5)      # 5 consecutive failures
failure.set_recovery_threshold(3)     # 3 consecutive successes
failure.set_detection_interval(3)     # Check every 3 heartbeats
failure.set_heartbeat_timeout(6)      # 6× heartbeat_interval = 300ms timeout

# Aggressive Detection (Fast detection, may have false positives)
failure.set_failure_threshold(2)      # 2 consecutive failures
failure.set_recovery_threshold(1)     # 1 consecutive success
failure.set_detection_interval(1)     # Check every heartbeat
failure.set_absolute_timeout(100)     # 100ms absolute timeout

# Balanced Detection (Recommended default)
failure.set_failure_threshold(3)      # 3 consecutive failures
failure.set_recovery_threshold(2)     # 2 consecutive successes
failure.set_detection_interval(2)     # Check every 2 heartbeats
failure.set_heartbeat_timeout(4)      # 4× heartbeat_interval = 200ms timeout
```

### Logging

The RAFT plugin provides comprehensive logging capabilities for debugging and monitoring consensus behavior. Logging is configured per node and can be customized based on your needs.

#### **Basic Logging Configuration**

```python
# Enable logging with default INFO level
config.set_logging(enable=True)

# Enable logging with specific level
config.set_logging(enable=True, level="DEBUG")
config.set_logging(enable=True, level="INFO")
config.set_logging(enable=True, level="WARNING")
config.set_logging(enable=True, level="ERROR")

# Disable logging (only errors will be shown)
config.set_logging(enable=False)
```

#### **Available Log Levels**

| Level | Description | Use Case |
|-------|-------------|----------|
| `DEBUG` | Detailed information for diagnosing problems | Development and debugging |
| `INFO` | General information about consensus operations | Normal operation monitoring |
| `WARNING` | Warning messages for potential issues | Production monitoring |
| `ERROR` | Error messages for serious problems | Error tracking |

#### **Logging Examples**

**Development and Debugging:**
```python
config = RaftConfig()
config.set_logging(enable=True, level="DEBUG")

# This will show:
# - Detailed election processes
# - Message exchanges
# - Timer events
# - State transitions
# - Failure detection details
```

**Production Monitoring:**
```python
config = RaftConfig()
config.set_logging(enable=True, level="INFO")

# This will show:
# - Election results
# - Leadership changes
# - Consensus value commits
# - Major state changes
```

**Minimal Logging:**
```python
config = RaftConfig()
config.set_logging(enable=True, level="WARNING")

# This will show:
# - Only warnings and errors
# - Critical consensus issues
# - Failure scenarios
```

**Silent Mode:**
```python
config = RaftConfig()
config.set_logging(enable=False)

# This will show:
# - Only critical errors
# - System failures
```

#### **Log Output Examples**

When logging is enabled, you'll see output like:

```
INFO:RaftConsensus-1: RaftConsensus initialized for node 1
INFO:RaftNode-1: Starting Raft consensus for node 1
INFO:RaftNode-1: Node 1 became candidate in term 1
INFO:RaftNode-1: Node 1 became leader in term 1
INFO:RaftNode-1: Proposing value for variable 'sequence': 42
INFO:RaftNode-1: Value committed for variable 'sequence': 42
```

#### **Logger Names**

The plugin creates separate loggers for different components:

- **`RaftConsensus-{node_id}`**: Main consensus facade logger
- **`RaftNode-{node_id}`**: Internal node implementation logger

This allows you to configure different log levels for different components if needed.

#### **Advanced Logging Configuration**

You can also configure logging at the Python logging level:

```python
import logging

# Configure root logger
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Configure specific RAFT loggers
logging.getLogger("RaftConsensus").setLevel(logging.INFO)
logging.getLogger("RaftNode").setLevel(logging.DEBUG)
```

## Working with the Consensus Instance

### Basic Operations

**Leadership and State:**
- `is_leader()` - Check if this node is the current leader
- `get_leader_id()` - Get the current leader ID
- `get_current_state()` - Get current Raft state (FOLLOWER, CANDIDATE, LEADER)
- `get_current_term()` - Get current term number

**Value Management:**
- `propose_value(name, value)` - Propose new value (leader only)
- `get_committed_value(name)` - Get committed value for a variable
- `get_all_committed_values()` - Get all committed values

**Node Management:**
- `set_simulation_active(node_id, bool)` - Set node as active/inactive in simulation
- `is_simulation_active(node_id)` - Check if node is active in simulation
- `get_failed_nodes()` - Get failed nodes (fault tolerant mode)
- `get_active_nodes()` - Get active nodes (fault tolerant mode)

**Monitoring and Debugging:**
- `get_statistics()` - Get consensus statistics and counters
- `get_state_info()` - Get detailed state information
- `get_active_nodes_info()` - Get detailed active node information (fault tolerant mode)

### Method Usage Examples

```python
def handle_telemetry(self, telemetry: Telemetry) -> None:
    # Check if we're the leader before proposing
    if self.consensus.is_leader():
        # Propose new values
        success = self.consensus.propose_value("sequence", int(telemetry.simulation_time))
        if success:
            print("✅ Value proposed successfully")
        else:
            print("❌ Failed to propose value")
        
        # Propose multiple values
        self.consensus.propose_value("leader_position", "north")
        self.consensus.propose_value("temperature", 25.5)
    
    # All nodes can retrieve committed values
    sequence_value = self.consensus.get_committed_value("sequence")
    position_value = self.consensus.get_committed_value("leader_position")
    
    if sequence_value is not None:
        print(f"Current sequence: {sequence_value}")
    
    # Get all committed values
    all_values = self.consensus.get_all_committed_values()
    print(f"All committed values: {all_values}")
    
    # Check consensus state
    is_leader = self.consensus.is_leader()
    leader_id = self.consensus.get_leader_id()
    current_term = self.consensus.get_current_term()
    
    print(f"Leader: {is_leader}, Leader ID: {leader_id}, Term: {current_term}")
```

### Active Node Information (Fault-Tolerant Mode)

```python
def handle_telemetry(self, telemetry: Telemetry) -> None:
    # Get detailed information about active nodes
    active_info = self.consensus.get_active_nodes_info()
    
    print(f"Active nodes: {active_info['active_nodes']}")
    print(f"Total nodes: {active_info['total_known']}")
    print(f"Failed nodes: {active_info['failed_nodes']}")
    print(f"Has majority: {active_info['has_majority']}")
    print(f"Detection method: {active_info['detection_method']}")
    
    # Check if we have quorum
    if self.consensus.has_quorum():
        print("✅ Cluster has quorum")
    else:
        print("❌ Cluster lacks quorum")
    
    # Check specific nodes
    if self.consensus.is_node_failed(3):
        print("Node 3 is currently failed")
    
    if self.consensus.is_simulation_active(5):
        print("Node 5 is active in simulation")
```

### Node Failure Simulation

```python
def handle_timer(self, timer: str) -> None:
    if timer == "failure_simulation":
        # Simulate failure of specific nodes
        nodes_to_fail = [1, 2, 3, 4, 5]
        for node_id in nodes_to_fail:
            self.consensus.set_simulation_active(node_id, False)
        print(f"Simulated failure of nodes: {nodes_to_fail}")
    
    elif timer == "recovery_simulation":
        # Simulate recovery of specific nodes
        nodes_to_recover = [1, 2, 3, 4, 5]
        for node_id in nodes_to_recover:
            self.consensus.set_simulation_active(node_id, True)
        print(f"Simulated recovery of nodes: {nodes_to_recover}")
    
    # Forward other timers to consensus
    if self.consensus:
        self.consensus.handle_timer(timer)
```

Always stop the plugin with `consensus.stop()` during `finish()` to clean up timers.

## Running the Showcase

The repository ships with a ready-to-run demonstration under `showcases/raft`. It sets up 40 nodes and exercises failure and recovery flows.

```bash
python -m showcases.raft.main
```

This command loads `showcases/raft/protocol.py`, which already uses `RaftConsensusPlugin` directly. Monitor the console output to track leadership changes, proposed values, and simulated failures.

## Best Practices

### Configuration Guidelines

#### **Election Timeouts**
```python
# For stable networks
config.set_election_timeout(150, 300)    # 150-300ms

# For unstable networks
config.set_election_timeout(200, 400)    # 200-400ms

# For very unstable networks
config.set_election_timeout(300, 600)    # 300-600ms
```

#### **Heartbeat Intervals**
```python
# Standard interval
config.set_heartbeat_interval(50)        # 50ms

# For high-frequency updates
config.set_heartbeat_interval(25)        # 25ms

# For low-frequency updates
config.set_heartbeat_interval(100)       # 100ms
```

#### **Failure Detection Configuration**

```python
# Conservative Detection (High reliability, slower detection)
failure_config.set_failure_threshold(5)      # 5 consecutive failures
failure_config.set_recovery_threshold(3)     # 3 consecutive successes
failure_config.set_detection_interval(3)     # Check every 3 heartbeats
failure_config.set_heartbeat_timeout(6)      # 6× heartbeat_interval = 300ms timeout

# Aggressive Detection (Fast detection, may have false positives)
failure_config.set_failure_threshold(2)      # 2 consecutive failures
failure_config.set_recovery_threshold(1)     # 1 consecutive success
failure_config.set_detection_interval(1)     # Check every heartbeat
failure_config.set_absolute_timeout(100)     # 100ms absolute timeout

# Balanced Detection (Recommended default)
failure_config.set_failure_threshold(3)      # 3 consecutive failures
failure_config.set_recovery_threshold(2)     # 2 consecutive successes
failure_config.set_detection_interval(2)     # Check every 2 heartbeats
failure_config.set_heartbeat_timeout(4)      # 4× heartbeat_interval = 200ms timeout
```

### Mode Selection

#### **Use Classic Mode When:**
- Cluster size is fixed and known
- Performance is critical
- Standard Raft behavior is required
- Manual failure management is acceptable

#### **Use Fault-Tolerant Mode When:**
- Nodes may fail unexpectedly
- Automatic fault recovery is needed
- Cluster size varies dynamically
- High availability is required
- Active node information is needed

### Error Handling

```python
# Always check if consensus is ready
if self.consensus.is_ready():
    # Proceed with operations
    pass

# Handle leader-only operations
if self.consensus.is_leader():
    success = self.consensus.propose_value("var", value)
    if not success:
        print("Failed to propose value")

# Check quorum before critical operations
if self.consensus.has_quorum():
    # Proceed with critical operation
    pass
else:
    print("No quorum available")
```

## Troubleshooting

### Common Issues

#### **1. No Leader Elected**
```python
# Check if nodes are properly configured
print(f"Known nodes: {self.consensus.get_known_nodes()}")
print(f"Active nodes: {self.consensus.get_active_nodes()}")

# Check if cluster has quorum
if self.consensus.has_quorum():
    print("Cluster has quorum")
else:
    print("Cluster lacks quorum - add more nodes")
```

#### **2. Values Not Committing**
```python
# Check if we're the leader
if not self.consensus.is_leader():
    print("Not the leader - cannot propose values")

# Check if we have quorum
if not self.consensus.has_quorum():
    print("No quorum - cannot commit values")

# Check consensus variables
variables = self.consensus.get_consensus_variables()
print(f"Available variables: {list(variables.keys())}")
```

#### **3. High Election Frequency**
```python
# Increase election timeout
config.set_election_timeout(300, 600)    # Longer timeouts

# Increase heartbeat frequency
config.set_heartbeat_interval(25)       # More frequent heartbeats

# Check network stability
failed_nodes = self.consensus.get_failed_nodes()
print(f"Failed nodes: {failed_nodes}")
```

### ⚠️ Known Issue: Fault Tolerance Communication Range

**Important Warning**: When the RAFT plugin is operating in **Fault-Tolerant Mode** (`RaftMode.FAULT_TOLERANT`) and the communication range is not large enough to reach all nodes in the swarm, the system may not converge to stability and consecutive elections may be triggered repeatedly.

**Problem Description:**
- In Fault-Tolerant mode, nodes perform active node discovery before elections
- When communication range is limited, the network may form multiple disconnected clusters
- **Edge nodes** that can be reached by multiple clusters simultaneously become a critical issue
- These edge nodes may receive conflicting election requests from different clusters
- This leads to incorrect majority calculations and failed leader elections
- The system may enter a cycle of repeated elections without reaching consensus
- **Most likely cause**: Edge nodes being simultaneously accessible by multiple clusters, creating conflicting consensus states

**Current Status:**
This is an **open research problem** for which a complete solution is not yet known. The issue occurs in distributed systems where:
- Nodes have limited communication range
- The network topology creates disconnected components
- The consensus algorithm requires global knowledge for proper operation

**Impact:**
- ❌ System may not converge to a stable leader
- ❌ Consecutive election cycles may occur
- ❌ Consensus values may not be committed
- ❌ Performance degradation due to constant re-elections

**Workarounds:**
1. **Increase Communication Range**: Ensure all nodes can communicate with each other
2. **Use Classic Mode**: Switch to `RaftMode.CLASSIC` for scenarios with limited communication range
3. **Network Topology Design**: Design network topology to ensure full connectivity

**Call for Contributions:**
This remains an **open research challenge** in distributed consensus algorithms. We invite researchers and developers to:

- **Research Solutions**: Investigate novel approaches to consensus in limited-range networks
- **Propose Algorithms**: Develop new consensus algorithms for disconnected networks
- **Test Scenarios**: Create test cases and benchmarks for this problem
- **Document Solutions**: Share findings and potential solutions
- **Collaborate**: Work together to find a robust solution

**Contact for Contributions:**
If you have ideas, solutions, or want to contribute to solving this problem, please contact:
- **Email**: [llucchesi@inf.puc-rio.br](mailto:llucchesi@inf.puc-rio.br)
- **Subject**: "RAFT Plugin - Fault Tolerance Communication Range Issue"

### Debugging Tips

#### **Enable Logging**
```python
config = RaftConfig()
config.set_logging(enable=True, level="DEBUG")
```

#### **Monitor Node States**
```python
# Get detailed state information
state_info = self.consensus.get_state_info()
print(f"Current state: {state_info['state']}")
print(f"Current term: {state_info['current_term']}")
print(f"Leader ID: {state_info['leader_id']}")
```

#### **Check Network Connectivity**
```python
# In fault-tolerant mode, check active nodes
active_info = self.consensus.get_active_nodes_info()
print(f"Active nodes: {active_info['active_nodes']}")
print(f"Failed nodes: {active_info['failed_nodes']}")

# Check majority calculation
majority_info = self.consensus.get_majority_info()
print(f"Majority needed: {majority_info['majority_needed']}")
print(f"Has majority: {majority_info['has_majority']}")
```

### Performance Monitoring

```python
# Get consensus statistics
stats = self.consensus.get_statistics()
print(f"Proposals made: {stats['proposals_made']}")
print(f"Values committed: {stats['values_committed']}")
print(f"Election attempts: {stats['election_attempts']}")

# Get failure detection metrics
metrics = self.consensus.get_failure_detection_metrics()
print(f"Detection latency: {metrics['detection_latency_ms']}ms")
print(f"Success rate: {metrics['success_rate_percent']}%")
```

## API Quick Reference

This section provides a comprehensive reference of all available methods in the RAFT consensus plugin, organized by functionality.

### **Configuration Methods**

**Import Required:**
```python
from gradysim.protocol.plugin.raft import RaftConfig, RaftMode
```

#### **RaftConfig Methods**

| Method | Description | Example |
|--------|-------------|---------|
| `set_election_timeout(min_ms, max_ms)` | Set randomized election timeout range | `config.set_election_timeout(150, 300)` |
| `set_heartbeat_interval(ms)` | Set leader heartbeat interval | `config.set_heartbeat_interval(50)` |
| `add_consensus_variable(name, type)` | Add a consensus variable | `config.add_consensus_variable("sequence", int)` |
| `set_logging(enable, level)` | Configure logging | `config.set_logging(True, "INFO")` |
| `set_raft_mode(mode)` | Set operation mode | `config.set_raft_mode(RaftMode.FAULT_TOLERANT)` |
| `get_failure_config()` | Get failure detection config | `failure = config.get_failure_config()` |

### **Lifecycle Methods**

**Import Required:**
```python
from gradysim.protocol.plugin.raft import RaftConsensusPlugin
```

| Method | Description | Example |
|--------|-------------|---------|
| `start()` | Start consensus process | `consensus.start()` |
| `stop()` | Stop consensus process | `consensus.stop()` |
| `is_ready()` | Check if system is ready | `if consensus.is_ready():` |

### **Leadership and State Methods**

**Import Required:**
```python
from gradysim.protocol.plugin.raft import RaftConsensusPlugin, RaftState
```

| Method | Description | Example |
|--------|-------------|---------|
| `is_leader()` | Check if this node is leader | `if consensus.is_leader():` |
| `get_leader_id()` | Get current leader ID | `leader_id = consensus.get_leader_id()` |
| `get_current_state()` | Get current Raft state | `state = consensus.get_current_state()` |
| `get_current_term()` | Get current term number | `term = consensus.get_current_term()` |

### **Value Management Methods**

**Import Required:**
```python
from gradysim.protocol.plugin.raft import RaftConsensusPlugin
```

| Method | Description | Example |
|--------|-------------|---------|
| `propose_value(name, value)` | Propose new value (leader only) | `consensus.propose_value("sequence", 42)` |
| `get_committed_value(name)` | Get committed value | `value = consensus.get_committed_value("sequence")` |
| `get_all_committed_values()` | Get all committed values | `values = consensus.get_all_committed_values()` |
| `get_consensus_variables()` | Get configured variables | `vars = consensus.get_consensus_variables()` |
| `has_consensus_variable(name)` | Check if variable exists | `if consensus.has_consensus_variable("sequence"):` |

### **Node Management Methods**

**Import Required:**
```python
from gradysim.protocol.plugin.raft import RaftConsensusPlugin
```

| Method | Description | Example |
|--------|-------------|---------|
| `set_known_nodes(node_ids)` | Set known node IDs | `consensus.set_known_nodes([1, 2, 3, 4, 5])` |
| `set_simulation_active(node_id, active)` | Set node simulation state | `consensus.set_simulation_active(1, False)` |
| `is_simulation_active(node_id)` | Check if node is active | `if consensus.is_simulation_active(1):` |
| `get_failed_nodes()` | Get failed nodes | `failed = consensus.get_failed_nodes()` |
| `get_active_nodes()` | Get active nodes | `active = consensus.get_active_nodes()` |
| `is_node_failed(node_id)` | Check if node failed | `if consensus.is_node_failed(1):` |

### **Active Node Information Methods (Fault-Tolerant Mode)**

**Import Required:**
```python
from gradysim.protocol.plugin.raft import RaftConsensusPlugin
```

| Method | Description | Example |
|--------|-------------|---------|
| `get_active_nodes_info()` | Get detailed active node info | `info = consensus.get_active_nodes_info()` |
| `get_communication_failed_nodes()` | Get communication failed nodes | `failed = consensus.get_communication_failed_nodes()` |
| `get_communication_active_nodes()` | Get communication active nodes | `active = consensus.get_communication_active_nodes()` |
| `is_communication_failed(node_id)` | Check communication failure | `if consensus.is_communication_failed(1):` |

### **Quorum and Majority Methods**

**Import Required:**
```python
from gradysim.protocol.plugin.raft import RaftConsensusPlugin
```

| Method | Description | Example |
|--------|-------------|---------|
| `has_quorum()` | Check if cluster has quorum | `if consensus.has_quorum():` |
| `has_majority_votes()` | Check if has majority votes | `if consensus.has_majority_votes():` |
| `has_majority_confirmation()` | Check majority confirmation | `if consensus.has_majority_confirmation():` |
| `get_majority_info()` | Get majority information | `info = consensus.get_majority_info()` |

### **Monitoring and Debugging Methods**

**Import Required:**
```python
from gradysim.protocol.plugin.raft import RaftConsensusPlugin
```

| Method | Description | Example |
|--------|-------------|---------|
| `get_statistics()` | Get consensus statistics | `stats = consensus.get_statistics()` |
| `get_state_info()` | Get detailed state info | `info = consensus.get_state_info()` |
| `get_configuration()` | Get current configuration | `config = consensus.get_configuration()` |
| `get_failure_detection_metrics()` | Get failure detection metrics | `metrics = consensus.get_failure_detection_metrics()` |

### **Cluster Management Methods**

**Import Required:**
```python
from gradysim.protocol.plugin.raft import RaftConsensusPlugin
```

| Method | Description | Example |
|--------|-------------|---------|
| `set_cluster_id(cluster_id)` | Set cluster ID | `consensus.set_cluster_id(1)` |
| `get_cluster_id()` | Get cluster ID | `id = consensus.get_cluster_id()` |
| `is_in_same_cluster(node_id)` | Check if in same cluster | `if consensus.is_in_same_cluster(2):` |

### **Complete Usage Example**

```python
from gradysim.protocol.plugin.raft import RaftConfig, RaftMode, RaftConsensusPlugin

class RaftProtocol(IProtocol):
    def initialize(self) -> None:
        # Configuration
        config = RaftConfig()
        config.set_election_timeout(150, 300)
        config.set_heartbeat_interval(50)
        config.add_consensus_variable("sequence", int)
        config.set_logging(enable=True, level="INFO")
        config.set_raft_mode(RaftMode.FAULT_TOLERANT)
        
        # Initialize consensus
        self.consensus = RaftConsensusPlugin(config=config, protocol=self)
        self.consensus.set_known_nodes(list(range(10)))
        self.consensus.start()
    
    def handle_telemetry(self, telemetry: Telemetry) -> None:
        # Leadership and value management
        if self.consensus.is_leader():
            self.consensus.propose_value("sequence", int(telemetry.simulation_time))
        
        # Get committed values
        sequence = self.consensus.get_committed_value("sequence")
        if sequence is not None:
            print(f"Current sequence: {sequence}")
        
        # Check consensus state
        if self.consensus.is_leader():
            print(f"Node {self.consensus.get_leader_id()} is leader in term {self.consensus.get_current_term()}")
        
        # Monitor active nodes (fault-tolerant mode)
        active_info = self.consensus.get_active_nodes_info()
        print(f"Active nodes: {active_info['active_nodes']}")
        print(f"Has majority: {active_info['has_majority']}")
        
        # Check quorum
        if not self.consensus.has_quorum():
            print("Warning: No quorum available")
    
    def finish(self) -> None:
        # Always stop consensus
        self.consensus.stop()
```

