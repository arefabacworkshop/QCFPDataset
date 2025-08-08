"""
Complex Event Processing (CEP) Engine for Kimera SWM

This module implements microsecond-latency event processing with quantum-enhanced
pattern matching capabilities, integrated with Kimera's cognitive architecture.
"""

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, Set, Callable
from datetime import datetime, timedelta
from collections import defaultdict, deque
from enum import Enum
import numpy as np
import time
from concurrent.futures import ThreadPoolExecutor
import heapq

# Quantum imports
try:
    import cirq
    import qiskit
    from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
    # Fix Qiskit import
    try:
        from qiskit_aer import Aer
        QUANTUM_AVAILABLE = True
    except ImportError:
        try:
            from qiskit import Aer
            QUANTUM_AVAILABLE = True
        except ImportError:
            QUANTUM_AVAILABLE = False
            logger.warning("Qiskit Aer not available - quantum features disabled")
except ImportError:
    QUANTUM_AVAILABLE = False
    logging.warning("Quantum libraries not available. Pattern matching will use classical methods.")

# Local imports
from src.core.geoid import GeoidState as Geoid
from src.engines.cognitive_field_dynamics import CognitiveFieldDynamics as CognitiveFieldDynamicsEngine
from src.engines.thermodynamic_engine import ThermodynamicEngine
from src.engines.contradiction_engine import ContradictionEngine

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EventPriority(Enum):
    """Event priority levels"""
    CRITICAL = 0  # Highest priority
    HIGH = 1
    MEDIUM = 2
    LOW = 3
    BACKGROUND = 4  # Lowest priority


@dataclass
class MarketEvent:
    """Represents a market event for processing"""
    event_id: str
    timestamp: datetime
    event_type: str
    symbol: str
    data: Dict[str, Any]
    priority: str = "medium"
    correlation_id: Optional[str] = None
    source: str = "unknown"
    
    def __lt__(self, other):
        """For priority queue ordering"""
        priority_map = {
            "critical": 0,
            "high": 1,
            "medium": 2,
            "low": 3,
            "background": 4
        }
        return priority_map.get(self.priority, 2) < priority_map.get(other.priority, 2)


@dataclass
class EventPattern:
    """Detected event pattern"""
    pattern_id: str
    pattern_type: str
    confidence: float
    events: List[MarketEvent]
    timestamp: datetime
    quantum_enhanced: bool = False
    cognitive_resonance: float = 0.0


@dataclass
class CorrelationResult:
    """Result of event correlation analysis"""
    correlation_id: str
    event_pairs: List[Tuple[str, str]]
    correlation_strength: float
    temporal_distance: timedelta
    spatial_similarity: float
    semantic_similarity: float


class ComplexEventProcessor:
    """
    Complex Event Processing Engine with quantum enhancement
    
    Features:
    - Microsecond latency processing
    - Quantum pattern matching
    - Multi-dimensional correlation
    - Cognitive field integration
    - Priority-based processing
    """
    
    def __init__(self,
                 cognitive_field: Optional[CognitiveFieldDynamicsEngine] = None,
                 thermodynamic_engine: Optional[ThermodynamicEngine] = None,
                 contradiction_engine: Optional[ContradictionEngine] = None):
        """Initialize CEP Engine"""
        self.cognitive_field = cognitive_field
        self.thermodynamic_engine = thermodynamic_engine
        self.contradiction_engine = contradiction_engine
        
        # Event storage
        self.event_store: Dict[str, MarketEvent] = {}
        self.event_index: Dict[str, List[str]] = defaultdict(list)  # Index by symbol
        self.event_timeline: deque = deque(maxlen=100000)  # Sliding window
        
        # Pattern detection
        self.pattern_library: Dict[str, Callable] = self._initialize_pattern_library()
        self.detected_patterns: List[EventPattern] = []
        
        # Priority queues for different event priorities
        self.priority_queues: Dict[EventPriority, List[MarketEvent]] = {
            priority: [] for priority in EventPriority
        }
        
        # Performance tracking
        self.events_processed = 0
        self.patterns_detected = 0
        self.processing_times = deque(maxlen=10000)
        
        # Quantum pattern matcher
        self.quantum_matcher = QuantumPatternMatcher() if QUANTUM_AVAILABLE else None
        
        # Thread pool for parallel processing
        self.executor = ThreadPoolExecutor(max_workers=4)
        
        # Background processing control
        self.running = False
        self.background_task = None
        
        logger.info("Complex Event Processor initialized")
        
    def start_background_processing(self):
        """Start background event processing"""
        if not self.running:
            self.running = True
            # Only create task if we're in an async context
            try:
                loop = asyncio.get_running_loop()
                self.background_task = loop.create_task(self._process_event_queues())
            except RuntimeError:
                # No event loop running, background processing will be manual
                pass
        
    def _initialize_pattern_library(self) -> Dict[str, Callable]:
        """Initialize pattern detection functions"""
        return {
            'price_spike': self._detect_price_spike,
            'volume_surge': self._detect_volume_surge,
            'momentum_shift': self._detect_momentum_shift,
            'correlation_break': self._detect_correlation_break,
            'flash_crash': self._detect_flash_crash,
            'liquidity_drain': self._detect_liquidity_drain,
            'order_clustering': self._detect_order_clustering,
            'market_manipulation': self._detect_market_manipulation
        }
        
    async def process_event(self, event: MarketEvent) -> None:
        """
        Process incoming market event with microsecond latency
        
        Args:
            event: Market event to process
        """
        start_time = time.perf_counter()
        
        try:
            # Store event
            self.event_store[event.event_id] = event
            self.event_index[event.symbol].append(event.event_id)
            self.event_timeline.append(event)
            
            # Add to priority queue
            priority = EventPriority[event.priority.upper()]
            heapq.heappush(self.priority_queues[priority], event)
            
            # Trigger pattern detection for critical events
            if event.priority == "critical":
                await self._immediate_pattern_detection(event)
                
            # Track processing time
            processing_time = (time.perf_counter() - start_time) * 1_000_000  # Convert to microseconds
            self.processing_times.append(processing_time)
            self.events_processed += 1
            
        except Exception as e:
            logger.error(f"Event processing error: {e}")
            
    async def _process_event_queues(self):
        """Background task to process event queues"""
        while self.running:
            try:
                # Process events by priority
                for priority in EventPriority:
                    queue = self.priority_queues[priority]
                    
                    if queue:
                        # Process batch of events
                        batch_size = 10 if priority == EventPriority.CRITICAL else 50
                        events_to_process = []
                        
                        for _ in range(min(batch_size, len(queue))):
                            if queue:
                                events_to_process.append(heapq.heappop(queue))
                                
                        if events_to_process:
                            await self._process_event_batch(events_to_process)
                            
                await asyncio.sleep(0.001)  # 1ms between batches
                
            except Exception as e:
                logger.error(f"Queue processing error: {e}")
                
    async def _process_event_batch(self, events: List[MarketEvent]):
        """Process a batch of events"""
        # Group by symbol for efficient processing
        symbol_groups = defaultdict(list)
        for event in events:
            symbol_groups[event.symbol].append(event)
            
        # Process each symbol group
        for symbol, symbol_events in symbol_groups.items():
            await self._detect_patterns_for_symbol(symbol, symbol_events)
            
    async def _immediate_pattern_detection(self, event: MarketEvent):
        """Immediate pattern detection for critical events"""
        # Check all pattern types
        for pattern_name, pattern_func in self.pattern_library.items():
            pattern = await pattern_func(event)
            if pattern:
                self.detected_patterns.append(pattern)
                self.patterns_detected += 1
                
                # Integrate with cognitive field
                if self.cognitive_field:
                    await self._integrate_pattern_with_cognitive_field(pattern)
                    
    async def detect_patterns(self, 
                            symbol: str,
                            time_window: timedelta) -> List[EventPattern]:
        """
        Detect patterns in events for a symbol within time window
        
        Args:
            symbol: Symbol to analyze
            time_window: Time window for pattern detection
            
        Returns:
            List of detected patterns
        """
        # Get relevant events
        cutoff_time = datetime.now() - time_window
        relevant_events = [
            self.event_store[event_id]
            for event_id in self.event_index[symbol]
            if self.event_store[event_id].timestamp > cutoff_time
        ]
        
        if not relevant_events:
            return []
            
        patterns = []
        
        # Run pattern detection
        for pattern_name, pattern_func in self.pattern_library.items():
            detected = await self._run_pattern_detection(
                pattern_name,
                pattern_func,
                relevant_events
            )
            patterns.extend(detected)
            
        # Quantum enhancement if available
        if self.quantum_matcher and len(relevant_events) > 5:
            quantum_patterns = await self._quantum_pattern_matching(relevant_events)
            patterns.extend(quantum_patterns)
            
        return patterns
        
    async def _run_pattern_detection(self,
                                   pattern_name: str,
                                   pattern_func: Callable,
                                   events: List[MarketEvent]) -> List[EventPattern]:
        """Run pattern detection function"""
        try:
            # Group events by type for pattern detection
            event_groups = defaultdict(list)
            for event in events:
                event_groups[event.event_type].append(event)
                
            patterns = []
            
            # Detect patterns in each group
            for event_type, group_events in event_groups.items():
                if len(group_events) >= 2:  # Need at least 2 events
                    pattern = await pattern_func(group_events)
                    if pattern:
                        patterns.append(pattern)
                        
            return patterns
            
        except Exception as e:
            logger.error(f"Pattern detection error for {pattern_name}: {e}")
            return []
            
    async def _quantum_pattern_matching(self, 
                                      events: List[MarketEvent]) -> List[EventPattern]:
        """Use quantum computing for enhanced pattern matching"""
        if not self.quantum_matcher:
            return []
            
        try:
            # Encode events for quantum processing
            encoded_events = self._encode_events_for_quantum(events)
            
            # Run quantum pattern matching
            quantum_results = await self.quantum_matcher.find_patterns(encoded_events)
            
            # Convert results to patterns
            patterns = []
            for result in quantum_results:
                pattern = EventPattern(
                    pattern_id=f"quantum_{datetime.now().timestamp()}",
                    pattern_type=result['type'],
                    confidence=result['confidence'],
                    events=result['events'],
                    timestamp=datetime.now(),
                    quantum_enhanced=True
                )
                patterns.append(pattern)
                
            return patterns
            
        except Exception as e:
            logger.error(f"Quantum pattern matching error: {e}")
            return []
            
    def _encode_events_for_quantum(self, events: List[MarketEvent]) -> np.ndarray:
        """Encode events for quantum processing"""
        # Simple encoding: price changes and volumes
        encoded = []
        for event in events:
            if 'price' in event.data and 'volume' in event.data:
                encoded.append([
                    event.data['price'],
                    event.data['volume'],
                    event.timestamp.timestamp()
                ])
                
        return np.array(encoded) if encoded else np.array([])
        
    # Pattern detection functions
    async def _detect_price_spike(self, events: List[MarketEvent]) -> Optional[EventPattern]:
        """Detect sudden price spikes"""
        if len(events) < 2:
            return None
            
        # Extract prices
        prices = []
        for event in events:
            if 'price' in event.data:
                prices.append((event.timestamp, event.data['price']))
                
        if len(prices) < 2:
            return None
            
        # Calculate price changes
        price_changes = []
        for i in range(1, len(prices)):
            time_diff = (prices[i][0] - prices[i-1][0]).total_seconds()
            if time_diff > 0:
                price_change = abs(prices[i][1] - prices[i-1][1]) / prices[i-1][1]
                rate = price_change / time_diff
                price_changes.append(rate)
                
        # Detect spike
        if price_changes and max(price_changes) > 0.001:  # 0.1% per second
            return EventPattern(
                pattern_id=f"spike_{datetime.now().timestamp()}",
                pattern_type="price_spike",
                confidence=min(max(price_changes) * 100, 1.0),
                events=events,
                timestamp=datetime.now()
            )
            
        return None
        
    async def _detect_volume_surge(self, events: List[MarketEvent]) -> Optional[EventPattern]:
        """Detect unusual volume surges"""
        volumes = []
        for event in events:
            if 'volume' in event.data:
                volumes.append(event.data['volume'])
                
        if len(volumes) < 5:
            return None
            
        avg_volume = np.mean(volumes[:-1])
        last_volume = volumes[-1]
        
        if last_volume > avg_volume * 3:  # 3x average
            return EventPattern(
                pattern_id=f"surge_{datetime.now().timestamp()}",
                pattern_type="volume_surge",
                confidence=min(last_volume / avg_volume / 3, 1.0),
                events=events[-5:],
                timestamp=datetime.now()
            )
            
        return None
        
    async def _detect_momentum_shift(self, events: List[MarketEvent]) -> Optional[EventPattern]:
        """Detect momentum shifts"""
        # Simplified momentum detection
        if len(events) < 10:
            return None
            
        prices = [e.data.get('price', 0) for e in events if 'price' in e.data]
        if len(prices) < 10:
            return None
            
        # Calculate momentum
        first_half = np.mean(prices[:len(prices)//2])
        second_half = np.mean(prices[len(prices)//2:])
        
        momentum_change = (second_half - first_half) / first_half
        
        if abs(momentum_change) > 0.02:  # 2% shift
            return EventPattern(
                pattern_id=f"momentum_{datetime.now().timestamp()}",
                pattern_type="momentum_shift",
                confidence=min(abs(momentum_change) * 50, 1.0),
                events=events,
                timestamp=datetime.now()
            )
            
        return None
        
    async def _detect_correlation_break(self, events: List[MarketEvent]) -> Optional[EventPattern]:
        """Detect correlation breaks between assets"""
        # This would need events from multiple symbols
        # Simplified for single symbol
        return None
        
    async def _detect_flash_crash(self, events: List[MarketEvent]) -> Optional[EventPattern]:
        """Detect flash crash patterns"""
        if len(events) < 5:
            return None
            
        prices = [(e.timestamp, e.data.get('price', 0)) for e in events if 'price' in e.data]
        if len(prices) < 5:
            return None
            
        # Check for rapid price decline
        max_price = max(p[1] for p in prices[:3])
        min_price = min(p[1] for p in prices[-3:])
        
        if max_price > 0:
            decline = (max_price - min_price) / max_price
            time_span = (prices[-1][0] - prices[0][0]).total_seconds()
            
            if decline > 0.05 and time_span < 60:  # 5% drop in under 60 seconds
                return EventPattern(
                    pattern_id=f"flash_{datetime.now().timestamp()}",
                    pattern_type="flash_crash",
                    confidence=min(decline * 10, 1.0),
                    events=events,
                    timestamp=datetime.now()
                )
                
        return None
        
    async def _detect_liquidity_drain(self, events: List[MarketEvent]) -> Optional[EventPattern]:
        """Detect liquidity drainage"""
        # Would need order book depth data
        return None
        
    async def _detect_order_clustering(self, events: List[MarketEvent]) -> Optional[EventPattern]:
        """Detect unusual order clustering"""
        # Would need order flow data
        return None
        
    async def _detect_market_manipulation(self, events: List[MarketEvent]) -> Optional[EventPattern]:
        """Detect potential market manipulation"""
        # Would need sophisticated analysis
        return None
        
    async def correlate_events(self,
                             time_window: timedelta,
                             correlation_threshold: float = 0.7) -> List[CorrelationResult]:
        """
        Correlate events across multiple dimensions
        
        Args:
            time_window: Time window for correlation
            correlation_threshold: Minimum correlation strength
            
        Returns:
            List of correlation results
        """
        correlations = []
        
        # Get recent events
        cutoff_time = datetime.now() - time_window
        recent_events = [e for e in self.event_timeline if e.timestamp > cutoff_time]
        
        # Perform pairwise correlation
        for i in range(len(recent_events)):
            for j in range(i + 1, len(recent_events)):
                event1, event2 = recent_events[i], recent_events[j]
                
                # Calculate correlation metrics
                temporal_distance = abs((event2.timestamp - event1.timestamp).total_seconds())
                
                # Spatial similarity (same symbol or related)
                spatial_sim = 1.0 if event1.symbol == event2.symbol else 0.3
                
                # Semantic similarity (similar event types)
                semantic_sim = 1.0 if event1.event_type == event2.event_type else 0.5
                
                # Overall correlation
                correlation_strength = (spatial_sim + semantic_sim) / 2
                
                if correlation_strength >= correlation_threshold:
                    correlations.append(CorrelationResult(
                        correlation_id=f"corr_{event1.event_id}_{event2.event_id}",
                        event_pairs=[(event1.event_id, event2.event_id)],
                        correlation_strength=correlation_strength,
                        temporal_distance=timedelta(seconds=temporal_distance),
                        spatial_similarity=spatial_sim,
                        semantic_similarity=semantic_sim
                    ))
                    
        return correlations
        
    async def _integrate_pattern_with_cognitive_field(self, pattern: EventPattern):
        """Integrate detected pattern with cognitive field"""
        if not self.cognitive_field:
            return
            
        # Create pattern geoid
        pattern_geoid = Geoid(
            geoid_id=f"pattern_{pattern.pattern_id}",
            semantic_state={
                'type': 1.0,  # market_pattern
                'pattern_type': hash(pattern.pattern_type) % 100 / 100,
                'confidence': pattern.confidence,
                'symbol': hash(pattern.events[0].symbol if pattern.events else 'unknown') % 100 / 100
            },
            symbolic_state={'content': f"Pattern: {pattern.pattern_type}"}
        )
        
        # Calculate cognitive resonance
        await self.cognitive_field.integrate_geoid(pattern_geoid)
        
        # Update pattern with cognitive resonance
        pattern.cognitive_resonance = await self.cognitive_field.calculate_coherence(pattern_geoid)
        
    async def get_performance_metrics(self) -> Dict[str, Any]:
        """Get CEP performance metrics"""
        avg_latency = np.mean(self.processing_times) if self.processing_times else 0
        
        return {
            'events_processed': self.events_processed,
            'patterns_detected': self.patterns_detected,
            'average_latency_us': avg_latency,
            'p99_latency_us': np.percentile(self.processing_times, 99) if self.processing_times else 0,
            'active_patterns': len(self.pattern_library),
            'quantum_enabled': self.quantum_matcher is not None,
            'event_store_size': len(self.event_store),
            'queue_depths': {
                priority.name: len(queue) 
                for priority, queue in self.priority_queues.items()
            }
        }
        
    async def optimize_processing(self):
        """Optimize CEP processing"""
        # Clear old events
        cutoff_time = datetime.now() - timedelta(hours=1)
        old_events = [
            event_id for event_id, event in self.event_store.items()
            if event.timestamp < cutoff_time
        ]
        
        for event_id in old_events:
            event = self.event_store.pop(event_id)
            self.event_index[event.symbol].remove(event_id)
            
        # Clear old patterns
        self.detected_patterns = [
            p for p in self.detected_patterns
            if p.timestamp > cutoff_time
        ]
        
    async def monitor_performance(self):
        """Monitor CEP performance continuously"""
        while self.running:
            metrics = await self.get_performance_metrics()
            
            # Log performance
            if metrics['average_latency_us'] > 100:
                logger.warning(f"High CEP latency: {metrics['average_latency_us']:.2f} μs")
                
            # Optimize if needed
            if metrics['event_store_size'] > 50000:
                await self.optimize_processing()
                
            await asyncio.sleep(10)  # Monitor every 10 seconds
            
    def shutdown(self):
        """Shutdown CEP engine"""
        self.running = False
        self.executor.shutdown()


class QuantumPatternMatcher:
    """Quantum-enhanced pattern matching"""
    
    def __init__(self):
        """Initialize the quantum pattern matcher"""
        if QUANTUM_AVAILABLE:
            try:
                # Initialize quantum backend
                self.backend = Aer.get_backend('qasm_simulator')
                self.quantum_available = True
                logger.info("✅ Quantum pattern matcher initialized with Aer backend")
            except Exception as e:
                logger.warning(f"⚠️ Quantum backend initialization failed: {e}")
                self.quantum_available = False
        else:
            self.quantum_available = False
            logger.warning("⚠️ Quantum pattern matcher disabled - Qiskit not available")
            
    async def find_patterns(self, encoded_events: np.ndarray) -> List[Dict[str, Any]]:
        """Find patterns using quantum algorithms"""
        if not QUANTUM_AVAILABLE or encoded_events.size == 0:
            return []
            
        try:
            # Create quantum circuit
            n_qubits = min(int(np.log2(len(encoded_events))) + 1, 10)
            qc = QuantumCircuit(n_qubits, n_qubits)
            
            # Encode data into quantum state
            # Simplified encoding
            for i in range(min(len(encoded_events), n_qubits)):
                if encoded_events[i, 0] > np.mean(encoded_events[:, 0]):
                    qc.x(i)
                    
            # Apply quantum operations for pattern detection
            for i in range(n_qubits - 1):
                qc.cx(i, i + 1)
                
            qc.h(range(n_qubits))
            
            # Measure
            qc.measure(range(n_qubits), range(n_qubits))
            
            # Execute
            from qiskit import execute
            job = execute(qc, self.backend, shots=1024)
            result = job.result()
            counts = result.get_counts()
            
            # Interpret results as patterns
            patterns = []
            for bitstring, count in counts.items():
                if count > 100:  # Significant occurrence
                    patterns.append({
                        'type': f'quantum_pattern_{bitstring}',
                        'confidence': count / 1024,
                        'events': []  # Would map back to original events
                    })
                    
            return patterns
            
        except Exception as e:
            logger.error(f"Quantum pattern matching error: {e}")
            return []


def create_complex_event_processor(cognitive_field=None,
                                 thermodynamic_engine=None,
                                 contradiction_engine=None) -> ComplexEventProcessor:
    """Factory function to create CEP engine"""
    return ComplexEventProcessor(
        cognitive_field=cognitive_field,
        thermodynamic_engine=thermodynamic_engine,
        contradiction_engine=contradiction_engine
    ) 