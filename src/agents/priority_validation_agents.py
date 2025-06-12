"""
Priority Validation Agents
Specialized agents for validating priority data (safety, cost, practical info)
"""
import asyncio
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime
import re
from dataclasses import dataclass

from .base_agent import BaseAgent, AgentMessage, MessageType
from ..core.safe_dict_utils import safe_get, safe_get_confidence_value, safe_get_nested, safe_get_dict
from src.schemas import PriorityMetrics

logger = logging.getLogger(__name__)


class SafetyValidationAgent(BaseAgent):
    """Agent specialized in validating safety-related information"""
    
    def __init__(self, agent_id: str):
        super().__init__(agent_id, "Safety Validation")
        self.safety_patterns = {
            "crime_index": r"crime\s+(index|rate).*?(\d+\.?\d*)",
            "safety_rating": r"safety\s+(rating|score).*?(\d+\.?\d*)",
            "emergency_numbers": r"emergency\s+(number|contact).*?(\d+)",
            "travel_advisory": r"(level\s*[1-4]|exercise\s+\w+\s+caution)",
            "tourist_police": r"tourist\s+police",
            "safe_areas": r"(safe\s+(area|neighborhood|district))",
            "dangerous_areas": r"(avoid|dangerous|unsafe)\s+(area|neighborhood|district)"
        }
        # Register message handlers
        self._message_handlers[MessageType.VALIDATION_REQUEST] = self._handle_validation_request
    
    async def _handle_validation_request(self, message: AgentMessage) -> Optional[AgentMessage]:
        """Handle validation request messages"""
        data = message.payload
        
        if not isinstance(data, dict) or "safety" not in data:
            return self.create_message(
                MessageType.ERROR,
                {"error": "Invalid safety data format"},
                recipient_id=message.sender_id,
                correlation_id=message.correlation_id
            )
        
        safety_data = data["safety"]
        validation_results = {
            "validated": True,
            "confidence": 1.0,
            "issues": [],
            "corrections": {}
        }
        
        # Validate crime index
        if safe_get(safety_data, "crime_index") is not None:
            crime_index = safety_data["crime_index"]
            if not (0 <= crime_index <= 100):
                validation_results["issues"].append(f"Crime index {crime_index} out of valid range (0-100)")
                validation_results["corrections"]["crime_index"] = max(0, min(100, crime_index))
                validation_results["validated"] = False
        
        # Validate safety rating
        if safe_get(safety_data, "safety_rating") is not None:
            safety_rating = safety_data["safety_rating"]
            if not (0 <= safety_rating <= 10):
                validation_results["issues"].append(f"Safety rating {safety_rating} out of valid range (0-10)")
                validation_results["corrections"]["safety_rating"] = max(0, min(10, safety_rating))
                validation_results["validated"] = False
        
        # Cross-validate crime index and safety rating
        if safe_get(safety_data, "crime_index") and safe_get(safety_data, "safety_rating"):
            # They should be inversely correlated
            expected_safety = 10 - (safety_data["crime_index"] / 10)
            if abs(safety_data["safety_rating"] - expected_safety) > 3:
                validation_results["issues"].append("Crime index and safety rating appear inconsistent")
                validation_results["confidence"] *= 0.8
        
        # Validate travel advisory level
        if safe_get(safety_data, "travel_advisory_level"):
            level = safety_data["travel_advisory_level"]
            if level not in [1, 2, 3, 4]:
                validation_results["issues"].append(f"Invalid travel advisory level: {level}")
                validation_results["validated"] = False
        
        # Validate emergency contacts
        if safe_get(safety_data, "emergency_contacts"):
            for contact_type, number in safety_data["emergency_contacts"].items():
                if not re.match(r"^\+?\d+$", str(number).replace("-", "").replace(" ", "")):
                    validation_results["issues"].append(f"Invalid emergency number format: {number}")
                    validation_results["confidence"] *= 0.9
        
        # Calculate overall confidence
        validation_results["confidence"] = round(validation_results["confidence"], 2)
        
        return self.create_message(
            MessageType.VALIDATION_RESPONSE,
            {
                "category": "safety",
                "original_data": safety_data,
                "validation": validation_results
            },
            recipient_id=message.sender_id,
            correlation_id=message.correlation_id
        )

    async def execute_task(self, task_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a safety validation task"""
        # Create a validation request message
        validation_message = self.create_message(
            MessageType.VALIDATION_REQUEST,
            task_data
        )
        
        # Process the validation request
        response = await self._handle_validation_request(validation_message)
        
        if response and response.message_type != MessageType.ERROR:
            return response.payload
        else:
            return {
                "error": "Validation failed",
                "details": response.payload if response else None
            }


class CostValidationAgent(BaseAgent):
    """Agent specialized in validating cost-related information"""
    
    def __init__(self, agent_id: str):
        super().__init__(agent_id, "Cost Validation")
        self.currency_rates = {
            "USD": 1.0,
            "EUR": 1.1,
            "GBP": 1.3,
            "JPY": 0.007,
            "CNY": 0.14,
            "AUD": 0.65,
            "CAD": 0.75
        }
        # Register message handlers
        self._message_handlers[MessageType.VALIDATION_REQUEST] = self._handle_validation_request
    
    async def _handle_validation_request(self, message: AgentMessage) -> Optional[AgentMessage]:
        """Handle validation request messages"""
        data = message.payload
        
        if not isinstance(data, dict) or "cost" not in data:
            return self.create_message(
                MessageType.ERROR,
                {"error": "Invalid cost data format"},
                recipient_id=message.sender_id,
                correlation_id=message.correlation_id
            )
        
        cost_data = data["cost"]
        validation_results = {
            "validated": True,
            "confidence": 1.0,
            "issues": [],
            "corrections": {}
        }
        
        # Validate budget ranges
        budget_fields = ["budget_per_day_low", "budget_per_day_mid", "budget_per_day_high"]
        budget_values = []
        
        for field in budget_fields:
            if safe_get(cost_data, field) is not None:
                value = cost_data[field]
                if value < 0:
                    validation_results["issues"].append(f"{field} cannot be negative: {value}")
                    validation_results["validated"] = False
                elif value > 2000:  # Sanity check for daily budget
                    validation_results["issues"].append(f"{field} seems unusually high: ${value}")
                    validation_results["confidence"] *= 0.8
                budget_values.append(value)
        
        # Ensure budget values are in ascending order
        if len(budget_values) >= 2:
            if budget_values != sorted(budget_values):
                validation_results["issues"].append("Budget values not in ascending order (low < mid < high)")
                validation_results["validated"] = False
                validation_results["corrections"]["budget_order"] = sorted(budget_values)
        
        # Validate currency
        if safe_get(cost_data, "currency"):
            currency = cost_data["currency"]
            if currency not in self.currency_rates:
                validation_results["issues"].append(f"Unrecognized currency: {currency}")
                validation_results["confidence"] *= 0.7
        
        # Validate specific costs
        cost_items = [
            ("meal_cost_budget", 5, 50),
            ("meal_cost_mid", 10, 100),
            ("meal_cost_luxury", 20, 500),
            ("coffee_price", 1, 10),
            ("beer_price", 2, 15),
            ("public_transport_ticket", 0.5, 10),
            ("taxi_start", 2, 20),
            ("hotel_budget", 20, 200),
            ("hotel_mid", 50, 500),
            ("hotel_luxury", 100, 2000)
        ]
        
        for item, min_val, max_val in cost_items:
            if safe_get(cost_data, item) is not None:
                value = cost_data[item]
                if not (min_val <= value <= max_val):
                    validation_results["issues"].append(
                        f"{item} value ${value} outside expected range (${min_val}-${max_val})"
                    )
                    validation_results["confidence"] *= 0.9
        
        # Cross-validate meal costs with daily budget
        if safe_get(cost_data, "meal_cost_budget") and safe_get(cost_data, "budget_per_day_low"):
            meal_percentage = (cost_data["meal_cost_budget"] * 3) / cost_data["budget_per_day_low"]
            if meal_percentage > 0.8:
                validation_results["issues"].append(
                    "Meal costs seem too high relative to daily budget"
                )
                validation_results["confidence"] *= 0.85
        
        # Calculate overall confidence
        validation_results["confidence"] = round(validation_results["confidence"], 2)
        
        return self.create_message(
            MessageType.VALIDATION_RESPONSE,
            {
                "category": "cost",
                "original_data": cost_data,
                "validation": validation_results
            },
            recipient_id=message.sender_id,
            correlation_id=message.correlation_id
        )

    async def execute_task(self, task_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a cost validation task"""
        # Create a validation request message
        validation_message = self.create_message(
            MessageType.VALIDATION_REQUEST,
            task_data
        )
        
        # Process the validation request
        response = await self._handle_validation_request(validation_message)
        
        if response and response.message_type != MessageType.ERROR:
            return response.payload
        else:
            return {
                "error": "Validation failed",
                "details": response.payload if response else None
            }


class PracticalInfoAgent(BaseAgent):
    """Agent specialized in validating practical travel information"""
    
    def __init__(self, agent_id: str):
        super().__init__(agent_id, "Practical Info Validation")
        self.valid_languages = {
            "English", "Spanish", "French", "German", "Italian", "Portuguese",
            "Chinese", "Japanese", "Korean", "Arabic", "Russian", "Hindi"
        }
        self.valid_visa_types = {
            "visa_free", "visa_on_arrival", "e_visa", "embassy_visa", "not_allowed"
        }
        # Register message handlers
        self._message_handlers[MessageType.VALIDATION_REQUEST] = self._handle_validation_request
    
    async def _handle_validation_request(self, message: AgentMessage) -> Optional[AgentMessage]:
        """Handle validation request messages"""
        data = message.payload
        validation_results = {
            "validated": True,
            "confidence": 1.0,
            "issues": [],
            "corrections": {}
        }
        
        # Validate health data
        if "health" in data:
            health_data = data["health"]
            
            # Validate water safety
            if safe_get(health_data, "water_safety"):
                valid_water_statuses = ["safe to drink", "bottled water recommended", "not safe"]
                if health_data["water_safety"].lower() not in [s.lower() for s in valid_water_statuses]:
                    validation_results["issues"].append(f"Invalid water safety status: {health_data['water_safety']}")
                    validation_results["confidence"] *= 0.9
            
            # Validate vaccinations
            if safe_get(health_data, "required_vaccinations"):
                known_vaccines = {
                    "yellow fever", "hepatitis a", "hepatitis b", "typhoid",
                    "japanese encephalitis", "rabies", "malaria", "polio"
                }
                for vaccine in health_data["required_vaccinations"]:
                    if vaccine.lower() not in known_vaccines:
                        validation_results["issues"].append(f"Unrecognized vaccine: {vaccine}")
                        validation_results["confidence"] *= 0.95
            
            # Validate medical facility quality
            if safe_get(health_data, "medical_facility_quality"):
                valid_qualities = ["excellent", "good", "adequate", "limited", "poor"]
                if health_data["medical_facility_quality"].lower() not in valid_qualities:
                    validation_results["issues"].append(
                        f"Invalid medical facility quality: {health_data['medical_facility_quality']}"
                    )
                    validation_results["confidence"] *= 0.9
        
        # Validate accessibility data
        if "accessibility" in data:
            access_data = data["accessibility"]
            
            # Validate visa information
            if safe_get(access_data, "visa_required") is not None:
                if safe_get(access_data, "visa_cost") and not access_data["visa_required"]:
                    validation_results["issues"].append("Visa cost specified but visa not required")
                    validation_results["confidence"] *= 0.8
            
            # Validate infrastructure rating
            if safe_get(access_data, "infrastructure_rating") is not None:
                rating = access_data["infrastructure_rating"]
                if not (0 <= rating <= 5):
                    validation_results["issues"].append(f"Infrastructure rating {rating} out of range (0-5)")
                    validation_results["corrections"]["infrastructure_rating"] = max(0, min(5, rating))
                    validation_results["validated"] = False
            
            # Validate language information
            if safe_get(access_data, "primary_language"):
                if access_data["primary_language"] not in self.valid_languages:
                    validation_results["issues"].append(
                        f"Unrecognized language: {access_data['primary_language']}"
                    )
                    validation_results["confidence"] *= 0.9
            
            # Validate english proficiency
            if safe_get(access_data, "english_proficiency"):
                valid_levels = ["none", "basic", "moderate", "good", "excellent", "native"]
                if access_data["english_proficiency"].lower() not in valid_levels:
                    validation_results["issues"].append(
                        f"Invalid English proficiency level: {access_data['english_proficiency']}"
                    )
                    validation_results["confidence"] *= 0.9
        
        # Validate weather data
        if "weather" in data:
            weather_data = data["weather"]
            
            # Validate temperature ranges
            temp_fields = [
                "avg_temp_summer", "avg_temp_winter",
                "avg_high_summer", "avg_high_winter",
                "avg_low_summer", "avg_low_winter"
            ]
            
            for field in temp_fields:
                if safe_get(weather_data, field) is not None:
                    temp = weather_data[field]
                    if not (-50 <= temp <= 60):  # Celsius
                        validation_results["issues"].append(
                            f"Temperature {field} seems unrealistic: {temp}°C"
                        )
                        validation_results["confidence"] *= 0.8
            
            # Validate rainfall
            if safe_get(weather_data, "rainfall_mm_annual") is not None:
                rainfall = weather_data["rainfall_mm_annual"]
                if not (0 <= rainfall <= 5000):
                    validation_results["issues"].append(
                        f"Annual rainfall seems unrealistic: {rainfall}mm"
                    )
                    validation_results["confidence"] *= 0.8
        
        # Calculate overall confidence
        validation_results["confidence"] = round(validation_results["confidence"], 2)
        
        return self.create_message(
            MessageType.VALIDATION_RESPONSE,
            {
                "practical_info": data,
                "validation": validation_results
            },
            recipient_id=message.sender_id,
            correlation_id=message.correlation_id
        )

    async def execute_task(self, task_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a practical info validation task"""
        # Create a validation request message
        validation_message = self.create_message(
            MessageType.VALIDATION_REQUEST,
            task_data
        )
        
        # Process the validation request
        response = await self._handle_validation_request(validation_message)
        
        if response and response.message_type != MessageType.ERROR:
            return response.payload
        else:
            return {
                "error": "Validation failed",
                "details": response.payload if response else None
            }


class PriorityValidationOrchestrator:
    """Orchestrates priority validation across specialized agents"""
    
    def __init__(self, message_broker):
        self.broker = message_broker
        self.safety_agent = SafetyValidationAgent("safety_validator")
        self.cost_agent = CostValidationAgent("cost_validator")
        self.practical_agent = PracticalInfoAgent("practical_validator")
        
        # Register agents
        self.broker.register_agent(self.safety_agent)
        self.broker.register_agent(self.cost_agent)
        self.broker.register_agent(self.practical_agent)
    
    async def validate_priority_data(self, priority_data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate all priority data through specialized agents"""
        
        # Send data to all validators
        validation_message = AgentMessage(
            sender_id="orchestrator",
            recipient_id=None,  # Broadcasting to all
            message_type=MessageType.VALIDATION_REQUEST,
            payload=priority_data,
            correlation_id=str(datetime.now().timestamp())
        )
        await self.broker.publish(validation_message)
        
        # Collect results
        validation_results = {
            "overall_valid": True,
            "overall_confidence": 1.0,
            "category_results": {},
            "all_issues": []
        }
        
        # Process safety validation
        if "safety" in priority_data:
            safety_result = await self.safety_agent.process_message(validation_message)
            if safety_result:
                validation_results["category_results"]["safety"] = safety_result.payload["validation"]
                if not safety_result.payload["validation"]["validated"]:
                    validation_results["overall_valid"] = False
                validation_results["overall_confidence"] *= safety_result.payload["validation"]["confidence"]
                validation_results["all_issues"].extend(safety_result.payload["validation"]["issues"])
        
        # Process cost validation
        if "cost" in priority_data:
            cost_result = await self.cost_agent.process_message(validation_message)
            if cost_result:
                validation_results["category_results"]["cost"] = cost_result.payload["validation"]
                if not cost_result.payload["validation"]["validated"]:
                    validation_results["overall_valid"] = False
                validation_results["overall_confidence"] *= cost_result.payload["validation"]["confidence"]
                validation_results["all_issues"].extend(cost_result.payload["validation"]["issues"])
        
        # Process practical info validation
        practical_result = await self.practical_agent.process_message(validation_message)
        if practical_result:
            validation_results["category_results"]["practical"] = practical_result.payload["validation"]
            if not practical_result.payload["validation"]["validated"]:
                validation_results["overall_valid"] = False
            validation_results["overall_confidence"] *= practical_result.payload["validation"]["confidence"]
            validation_results["all_issues"].extend(practical_result.payload["validation"]["issues"])
        
        validation_results["overall_confidence"] = round(validation_results["overall_confidence"], 2)
        
        return validation_results 