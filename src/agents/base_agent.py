from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Callable
from datetime import datetime
from enum import Enum
import uuid
import logging
import asyncio
from queue import Queue
import json

logger = logging.getLogger(__name__)

class MessageType(Enum):
    """Types of messages agents can exchange"""
    TASK_REQUEST = "task_request"
    TASK_RESPONSE = "task_response"
    DATA_UPDATE = "data_update"
    VALIDATION_REQUEST = "validation_request"
    VALIDATION_RESPONSE = "validation_response"
    ERROR = "error"
    STATUS_UPDATE = "status_update"
    DISCOVERY = "discovery"
    CONTRADICTION = "contradiction"
    GAP_IDENTIFIED = "gap_identified"

@dataclass
class AgentMessage:
    """Message structure for inter-agent communication"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    sender_id: str = ""
    recipient_id: Optional[str] = None  # None = broadcast
    message_type: MessageType = MessageType.TASK_REQUEST
    payload: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    correlation_id: Optional[str] = None  # For request-response tracking
    priority: int = 5  # 1-10, higher = more important
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "id": self.id,
            "sender_id": self.sender_id,
            "recipient_id": self.recipient_id,
            "message_type": self.message_type.value,
            "payload": self.payload,
            "timestamp": self.timestamp.isoformat(),
            "correlation_id": self.correlation_id,
            "priority": self.priority
        }

class BaseAgent(ABC):
    """
    Abstract base class for all specialized agents
    
    Implements stateless design with message-based communication
    """
    
    def __init__(self, agent_id: str, agent_type: str, config: Optional[Dict[str, Any]] = None):
        self.agent_id = agent_id
        self.agent_type = agent_type
        self.config = config or {}
        self.logger = logging.getLogger(f"{__name__}.{agent_type}.{agent_id}")
        self._message_handlers: Dict[MessageType, Callable] = {}
        self._register_default_handlers()
        
    def _register_default_handlers(self):
        """Register default message handlers"""
        self._message_handlers[MessageType.STATUS_UPDATE] = self._handle_status_update
        self._message_handlers[MessageType.ERROR] = self._handle_error
        
    async def process_message(self, message: AgentMessage) -> Optional[AgentMessage]:
        """
        Process an incoming message
        
        Returns response message if applicable
        """
        self.logger.debug(f"Processing message {message.id} of type {message.message_type}")
        
        # Check if message is for this agent
        if message.recipient_id and message.recipient_id != self.agent_id:
            return None
            
        # Get appropriate handler
        handler = self._message_handlers.get(message.message_type)
        if handler:
            try:
                return await handler(message)
            except Exception as e:
                self.logger.error(f"Error processing message {message.id}: {e}")
                return self._create_error_response(message, str(e))
        else:
            self.logger.warning(f"No handler for message type {message.message_type}")
            return None
    
    @abstractmethod
    async def execute_task(self, task_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute the agent's primary task
        
        Must be implemented by subclasses
        """
        pass
    
    def register_handler(self, message_type: MessageType, handler: Callable):
        """Register a message handler for a specific message type"""
        self._message_handlers[message_type] = handler
        
    def create_message(
        self,
        message_type: MessageType,
        payload: Dict[str, Any],
        recipient_id: Optional[str] = None,
        correlation_id: Optional[str] = None,
        priority: int = 5
    ) -> AgentMessage:
        """Create a new message from this agent"""
        return AgentMessage(
            sender_id=self.agent_id,
            recipient_id=recipient_id,
            message_type=message_type,
            payload=payload,
            correlation_id=correlation_id,
            priority=priority
        )
    
    def _create_error_response(self, original_message: AgentMessage, error: str) -> AgentMessage:
        """Create an error response message"""
        return self.create_message(
            MessageType.ERROR,
            {
                "error": error,
                "original_message_id": original_message.id,
                "original_message_type": original_message.message_type.value
            },
            recipient_id=original_message.sender_id,
            correlation_id=original_message.correlation_id
        )
    
    async def _handle_status_update(self, message: AgentMessage) -> None:
        """Default handler for status updates"""
        self.logger.info(f"Status update from {message.sender_id}: {message.payload.get('status', 'unknown')}")
        
    async def _handle_error(self, message: AgentMessage) -> None:
        """Default handler for errors"""
        self.logger.error(f"Error from {message.sender_id}: {message.payload.get('error', 'unknown')}")

class MessageBroker:
    """
    Simple in-memory message broker for agent communication
    
    In production, this would be replaced with Kafka/RabbitMQ
    """
    
    def __init__(self):
        self.agents: Dict[str, BaseAgent] = {}
        self.message_queue: asyncio.Queue = asyncio.Queue()
        self.logger = logging.getLogger(f"{__name__}.MessageBroker")
        self._running = False
        
    def register_agent(self, agent: BaseAgent):
        """Register an agent with the broker"""
        self.agents[agent.agent_id] = agent
        self.logger.info(f"Registered agent {agent.agent_id} of type {agent.agent_type}")
        
    def unregister_agent(self, agent_id: str):
        """Unregister an agent"""
        if agent_id in self.agents:
            del self.agents[agent_id]
            self.logger.info(f"Unregistered agent {agent_id}")
            
    async def publish(self, message: AgentMessage):
        """Publish a message to the broker"""
        await self.message_queue.put(message)
        self.logger.debug(f"Published message {message.id} from {message.sender_id}")
        
    async def start(self):
        """Start the message broker"""
        self._running = True
        self.logger.info("Message broker started")
        
        while self._running:
            try:
                # Get message with timeout to allow checking _running flag
                message = await asyncio.wait_for(self.message_queue.get(), timeout=1.0)
                await self._route_message(message)
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                self.logger.error(f"Error processing message: {e}")
                
    async def stop(self):
        """Stop the message broker"""
        self._running = False
        self.logger.info("Message broker stopped")
        
    async def _route_message(self, message: AgentMessage):
        """Route message to appropriate agent(s)"""
        if message.recipient_id:
            # Direct message
            if message.recipient_id in self.agents:
                agent = self.agents[message.recipient_id]
                response = await agent.process_message(message)
                if response:
                    await self.publish(response)
            else:
                self.logger.warning(f"Agent {message.recipient_id} not found")
        else:
            # Broadcast message
            tasks = []
            for agent in self.agents.values():
                if agent.agent_id != message.sender_id:  # Don't send to self
                    tasks.append(agent.process_message(message))
                    
            responses = await asyncio.gather(*tasks)
            for response in responses:
                if response:
                    await self.publish(response)

class AgentOrchestrator:
    """
    Orchestrates multi-agent workflows
    
    Manages agent lifecycle and coordinates complex tasks
    """
    
    def __init__(self, broker: MessageBroker):
        self.broker = broker
        self.logger = logging.getLogger(f"{__name__}.AgentOrchestrator")
        self.workflow_results: Dict[str, Any] = {}
        
    async def execute_workflow(
        self,
        workflow_name: str,
        workflow_steps: List[Dict[str, Any]],
        initial_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Execute a multi-step workflow across agents
        
        Args:
            workflow_name: Name of the workflow
            workflow_steps: List of steps with agent_type and task_data
            initial_data: Initial data for the workflow
            
        Returns:
            Workflow results
        """
        workflow_id = str(uuid.uuid4())
        self.logger.info(f"Starting workflow {workflow_name} (ID: {workflow_id})")
        
        current_data = initial_data.copy()
        results = {
            "workflow_id": workflow_id,
            "workflow_name": workflow_name,
            "start_time": datetime.now().isoformat(),
            "steps": []
        }
        
        for step_idx, step in enumerate(workflow_steps):
            agent_type = step["agent_type"]
            task_data = {**current_data, **step.get("task_data", {})}
            
            self.logger.info(f"Executing step {step_idx + 1}: {agent_type}")
            
            # Find agent of the required type
            agent = self._find_agent_by_type(agent_type)
            if not agent:
                error_msg = f"No agent of type {agent_type} available"
                self.logger.error(error_msg)
                results["error"] = error_msg
                break
                
            # Execute task
            try:
                step_result = await agent.execute_task(task_data)
                results["steps"].append({
                    "step": step_idx + 1,
                    "agent_type": agent_type,
                    "agent_id": agent.agent_id,
                    "status": "success",
                    "result": step_result
                })
                
                # Update current data with step results
                if isinstance(step_result, dict):
                    current_data.update(step_result)
                    
            except Exception as e:
                self.logger.error(f"Step {step_idx + 1} failed: {e}")
                results["steps"].append({
                    "step": step_idx + 1,
                    "agent_type": agent_type,
                    "status": "failed",
                    "error": str(e)
                })
                results["error"] = f"Workflow failed at step {step_idx + 1}"
                break
                
        results["end_time"] = datetime.now().isoformat()
        results["final_data"] = current_data
        
        self.workflow_results[workflow_id] = results
        return results
        
    def _find_agent_by_type(self, agent_type: str) -> Optional[BaseAgent]:
        """Find an available agent of the specified type"""
        for agent in self.broker.agents.values():
            if agent.agent_type == agent_type:
                return agent
        return None
        
    def get_workflow_status(self, workflow_id: str) -> Optional[Dict[str, Any]]:
        """Get status of a workflow"""
        return self.workflow_results.get(workflow_id) 