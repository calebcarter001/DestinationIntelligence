#!/usr/bin/env python3
"""
Enhanced Query Tool for Priority Insights
Provides advanced querying and display of priority data with traveler decision support
"""
import sqlite3
import json
from datetime import datetime
from typing import Dict, List, Optional, Any
from tabulate import tabulate
import click
from colorama import init, Fore, Style

from src.core.priority_export_manager import PriorityExportManager
from src.schemas import PriorityMetrics

# Initialize colorama for colored output
init(autoreset=True)


class PriorityQueryTool:
    """Enhanced query tool for priority insights"""
    
    def __init__(self, db_path: str = "real_destination_intelligence.db"):
        self.db_path = db_path
        self.export_manager = PriorityExportManager()
    
    def query_priority_summary(self, destination_name: str) -> Dict[str, Any]:
        """Query priority summary for a destination"""
        
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            # Get priority metrics
            cursor.execute("""
                SELECT * FROM priority_metrics 
                WHERE destination_name = ? 
                ORDER BY created_at DESC 
                LIMIT 1
            """, (destination_name,))
            
            metrics_row = cursor.fetchone()
            if not metrics_row:
                return {"error": f"No priority data found for {destination_name}"}
            
            # Get priority insights
            cursor.execute("""
                SELECT * FROM priority_insights 
                WHERE destination_name = ? 
                ORDER BY confidence_score DESC
            """, (destination_name,))
            
            insights = cursor.fetchall()
            
            # Build response
            return {
                "destination": destination_name,
                "metrics": dict(metrics_row),
                "insights": [dict(i) for i in insights],
                "summary": self._generate_summary(metrics_row, insights)
            }
    
    def compare_destinations(
        self,
        destinations: List[str],
        criteria: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Compare multiple destinations on priority criteria"""
        
        if not criteria:
            criteria = ["safety", "cost", "health", "accessibility"]
        
        comparison_data = []
        
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            for dest in destinations:
                # Get metrics
                cursor.execute("""
                    SELECT * FROM priority_metrics 
                    WHERE destination_name = ? 
                    ORDER BY created_at DESC 
                    LIMIT 1
                """, (dest,))
                
                metrics_row = cursor.fetchone()
                if metrics_row:
                    metrics = PriorityMetrics(**dict(metrics_row))
                    
                    # Generate scorecard
                    scorecard = self.export_manager.generate_traveler_scorecard(
                        dest, metrics, []
                    )
                    
                    comparison_data.append({
                        "destination": dest,
                        "priority_metrics": metrics,
                        "scorecard": scorecard
                    })
        
        # Export comparison
        if len(comparison_data) >= 2:
            comparison_file = self.export_manager.export_comparative_analysis(
                comparison_data, criteria
            )
            
            # Load and return comparison
            with open(comparison_file, 'r') as f:
                return json.load(f)
        
        return {"error": "Not enough destinations with priority data for comparison"}
    
    def search_by_criteria(
        self,
        max_crime_index: Optional[float] = None,
        max_daily_budget: Optional[float] = None,
        visa_free: Optional[bool] = None,
        safe_water: Optional[bool] = None,
        min_infrastructure: Optional[float] = None
    ) -> List[Dict[str, Any]]:
        """Search destinations by priority criteria"""
        
        conditions = []
        params = []
        
        if max_crime_index is not None:
            conditions.append("crime_index <= ?")
            params.append(max_crime_index)
        
        if max_daily_budget is not None:
            conditions.append("budget_per_day_low <= ?")
            params.append(max_daily_budget)
        
        if visa_free is True:
            conditions.append("(visa_required IS NULL OR visa_required = 0)")
        
        if safe_water is True:
            conditions.append("water_safety LIKE '%safe%' AND water_safety NOT LIKE '%not safe%'")
        
        if min_infrastructure is not None:
            conditions.append("infrastructure_rating >= ?")
            params.append(min_infrastructure)
        
        where_clause = " AND ".join(conditions) if conditions else "1=1"
        
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            query = f"""
                SELECT DISTINCT destination_name, crime_index, budget_per_day_low,
                       visa_required, water_safety, infrastructure_rating
                FROM priority_metrics
                WHERE {where_clause}
                ORDER BY crime_index ASC, budget_per_day_low ASC
            """
            
            cursor.execute(query, params)
            results = cursor.fetchall()
            
            return [dict(r) for r in results]
    
    def get_traveler_recommendations(
        self,
        traveler_profile: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Get personalized recommendations based on traveler profile"""
        
        # Extract profile preferences
        max_budget = traveler_profile.get("daily_budget", 150)
        safety_threshold = traveler_profile.get("min_safety_score", 7)
        visa_preference = traveler_profile.get("visa_free_only", False)
        
        # Query suitable destinations
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            # Build query based on preferences
            conditions = ["budget_per_day_low <= ?"]
            params = [max_budget]
            
            if visa_preference:
                conditions.append("(visa_required IS NULL OR visa_required = 0)")
            
            query = f"""
                SELECT * FROM priority_metrics
                WHERE {' AND '.join(conditions)}
                ORDER BY crime_index ASC
            """
            
            cursor.execute(query, params)
            destinations = cursor.fetchall()
            
            recommendations = []
            
            for dest_row in destinations:
                metrics = PriorityMetrics(**dict(dest_row))
                scorecard = self.export_manager.generate_traveler_scorecard(
                    dest_row["destination_name"],
                    metrics,
                    [],
                    traveler_profile
                )
                
                # Filter by safety score
                if scorecard["category_scores"]["safety"] >= safety_threshold:
                    recommendations.append({
                        "destination": dest_row["destination_name"],
                        "match_score": scorecard["overall_score"],
                        "scorecard": scorecard,
                        "why_recommended": self._explain_recommendation(
                            metrics, scorecard, traveler_profile
                        )
                    })
            
            # Sort by match score
            recommendations.sort(key=lambda x: x["match_score"], reverse=True)
            
            return recommendations[:10]  # Top 10 recommendations
    
    def display_priority_summary(self, destination_name: str):
        """Display formatted priority summary"""
        
        data = self.query_priority_summary(destination_name)
        
        if "error" in data:
            print(f"{Fore.RED}Error: {data['error']}{Style.RESET_ALL}")
            return
        
        print(f"\n{Fore.CYAN}═══ Priority Analysis: {destination_name} ═══{Style.RESET_ALL}\n")
        
        # Display metrics
        metrics = data["metrics"]
        
        # Safety section
        print(f"{Fore.YELLOW}SAFETY{Style.RESET_ALL}")
        if metrics.get("crime_index"):
            color = Fore.GREEN if metrics["crime_index"] < 30 else Fore.YELLOW if metrics["crime_index"] < 60 else Fore.RED
            print(f"  Crime Index: {color}{metrics['crime_index']:.1f}/100{Style.RESET_ALL}")
        if metrics.get("tourist_police_available"):
            print(f"  Tourist Police: {Fore.GREEN}✓ Available{Style.RESET_ALL}")
        
        # Cost section
        print(f"\n{Fore.YELLOW}COST{Style.RESET_ALL}")
        if metrics.get("budget_per_day_low"):
            print(f"  Budget Range: ${metrics['budget_per_day_low']}-${metrics.get('budget_per_day_high', 'N/A')}/day")
        if metrics.get("currency"):
            print(f"  Currency: {metrics['currency']}")
        
        # Health section
        print(f"\n{Fore.YELLOW}HEALTH{Style.RESET_ALL}")
        if metrics.get("water_safety"):
            color = Fore.GREEN if "safe" in metrics["water_safety"].lower() and "not" not in metrics["water_safety"].lower() else Fore.YELLOW
            print(f"  Water: {color}{metrics['water_safety']}{Style.RESET_ALL}")
        if metrics.get("required_vaccinations"):
            print(f"  Required Vaccinations: {metrics['required_vaccinations']}")
        
        # Accessibility section
        print(f"\n{Fore.YELLOW}ACCESSIBILITY{Style.RESET_ALL}")
        if metrics.get("visa_required") is not None:
            if metrics["visa_required"]:
                print(f"  Visa: {Fore.YELLOW}Required (${metrics.get('visa_cost', 'N/A')}){Style.RESET_ALL}")
            else:
                print(f"  Visa: {Fore.GREEN}Not Required{Style.RESET_ALL}")
        if metrics.get("infrastructure_rating"):
            print(f"  Infrastructure: {metrics['infrastructure_rating']:.1f}/5")
        
        # Display insights
        if data["insights"]:
            print(f"\n{Fore.YELLOW}KEY INSIGHTS{Style.RESET_ALL}")
            for insight in data["insights"][:5]:
                impact_color = Fore.RED if insight["priority_impact"] == "high" else Fore.YELLOW
                print(f"  • {impact_color}[{insight['priority_impact'].upper()}]{Style.RESET_ALL} {insight['insight_name']}")
    
    def display_comparison_table(self, destinations: List[str]):
        """Display comparison table for multiple destinations"""
        
        comparison = self.compare_destinations(destinations)
        
        if "error" in comparison:
            print(f"{Fore.RED}Error: {comparison['error']}{Style.RESET_ALL}")
            return
        
        print(f"\n{Fore.CYAN}═══ Destination Comparison ═══{Style.RESET_ALL}\n")
        
        # Build table data
        headers = ["Destination", "Overall", "Safety", "Cost", "Health", "Access"]
        table_data = []
        
        for dest, data in comparison["comparison_matrix"].items():
            scores = data["category_scores"]
            row = [
                dest,
                f"{data['overall_score']:.1f}",
                self._score_with_color(scores.get("safety", 0)),
                self._score_with_color(scores.get("cost", 0)),
                self._score_with_color(scores.get("health", 0)),
                self._score_with_color(scores.get("accessibility", 0))
            ]
            table_data.append(row)
        
        print(tabulate(table_data, headers=headers, tablefmt="grid"))
        
        # Show rankings
        print(f"\n{Fore.YELLOW}RANKINGS{Style.RESET_ALL}")
        for criterion, ranking in comparison["rankings"].items():
            print(f"\n{criterion.capitalize()}:")
            for rank_data in ranking[:3]:  # Top 3
                print(f"  {rank_data['rank']}. {rank_data['destination']} ({rank_data['score']:.1f})")
        
        # Show best for categories
        if comparison.get("best_for"):
            print(f"\n{Fore.YELLOW}BEST FOR{Style.RESET_ALL}")
            for category, dests in comparison["best_for"].items():
                if dests:
                    print(f"  {category.replace('_', ' ').title()}: {', '.join(dests)}")
    
    def _generate_summary(self, metrics_row: sqlite3.Row, insights: List[sqlite3.Row]) -> Dict[str, Any]:
        """Generate summary from metrics and insights"""
        
        summary = {
            "safety_level": self._categorize_safety(metrics_row["crime_index"]),
            "cost_level": self._categorize_cost(metrics_row["budget_per_day_low"]),
            "health_preparedness": self._categorize_health(metrics_row),
            "accessibility_level": self._categorize_accessibility(metrics_row),
            "high_priority_concerns": len([i for i in insights if i["priority_impact"] == "high"])
        }
        
        return summary
    
    def _categorize_safety(self, crime_index: Optional[float]) -> str:
        """Categorize safety level"""
        if crime_index is None:
            return "Unknown"
        elif crime_index < 20:
            return "Very Safe"
        elif crime_index < 40:
            return "Safe"
        elif crime_index < 60:
            return "Moderate"
        else:
            return "Use Caution"
    
    def _categorize_cost(self, budget_low: Optional[float]) -> str:
        """Categorize cost level"""
        if budget_low is None:
            return "Unknown"
        elif budget_low < 50:
            return "Budget"
        elif budget_low < 100:
            return "Moderate"
        elif budget_low < 200:
            return "Mid-Range"
        else:
            return "Luxury"
    
    def _categorize_health(self, metrics: sqlite3.Row) -> str:
        """Categorize health preparedness needs"""
        if metrics["required_vaccinations"]:
            return "High Preparation"
        elif metrics["water_safety"] and "not safe" in metrics["water_safety"].lower():
            return "Moderate Preparation"
        else:
            return "Low Preparation"
    
    def _categorize_accessibility(self, metrics: sqlite3.Row) -> str:
        """Categorize accessibility level"""
        if not metrics["visa_required"] and metrics["infrastructure_rating"] and metrics["infrastructure_rating"] >= 4:
            return "Very Accessible"
        elif metrics["visa_required"]:
            return "Moderate"
        else:
            return "Good"
    
    def _score_with_color(self, score: float) -> str:
        """Format score with color"""
        if score >= 8:
            return f"{Fore.GREEN}{score:.1f}{Style.RESET_ALL}"
        elif score >= 6:
            return f"{Fore.YELLOW}{score:.1f}{Style.RESET_ALL}"
        else:
            return f"{Fore.RED}{score:.1f}{Style.RESET_ALL}"
    
    def _explain_recommendation(
        self,
        metrics: PriorityMetrics,
        scorecard: Dict[str, Any],
        profile: Dict[str, Any]
    ) -> List[str]:
        """Explain why a destination is recommended"""
        
        reasons = []
        
        # Budget match
        if metrics.budget_per_day_low and metrics.budget_per_day_low <= profile.get("daily_budget", 150):
            reasons.append(f"Within your budget (${metrics.budget_per_day_low}/day)")
        
        # Safety match
        if scorecard["category_scores"]["safety"] >= 8:
            reasons.append("Excellent safety record")
        
        # Visa preference
        if not metrics.visa_required and profile.get("visa_free_only"):
            reasons.append("No visa required")
        
        # Language preference
        if (metrics.english_proficiency and 
            metrics.english_proficiency.lower() in ["good", "excellent", "native"] and
            profile.get("english_speaking_preferred")):
            reasons.append("English widely spoken")
        
        return reasons


@click.group()
def cli():
    """Enhanced Priority Query Tool for Destination Intelligence"""
    pass


@cli.command()
@click.argument('destination')
def summary(destination):
    """Show priority summary for a destination"""
    tool = PriorityQueryTool()
    tool.display_priority_summary(destination)


@cli.command()
@click.argument('destinations', nargs=-1, required=True)
def compare(destinations):
    """Compare multiple destinations"""
    if len(destinations) < 2:
        click.echo("Please provide at least 2 destinations to compare")
        return
    tool = PriorityQueryTool()
    tool.display_comparison_table(list(destinations))


@cli.command()
@click.option('--max-crime', type=float, help='Maximum crime index')
@click.option('--max-budget', type=float, help='Maximum daily budget')
@click.option('--visa-free', is_flag=True, help='Only visa-free destinations')
@click.option('--safe-water', is_flag=True, help='Only destinations with safe tap water')
@click.option('--min-infrastructure', type=float, help='Minimum infrastructure rating')
def search(max_crime, max_budget, visa_free, safe_water, min_infrastructure):
    """Search destinations by criteria"""
    tool = PriorityQueryTool()
    results = tool.search_by_criteria(
        max_crime_index=max_crime,
        max_daily_budget=max_budget,
        visa_free=visa_free,
        safe_water=safe_water,
        min_infrastructure=min_infrastructure
    )
    
    if results:
        headers = ["Destination", "Crime Index", "Budget/Day", "Visa", "Water", "Infrastructure"]
        table_data = []
        
        for r in results:
            row = [
                r["destination_name"],
                f"{r['crime_index']:.1f}" if r["crime_index"] else "N/A",
                f"${r['budget_per_day_low']}" if r["budget_per_day_low"] else "N/A",
                "Required" if r["visa_required"] else "Not Required",
                r["water_safety"] or "N/A",
                f"{r['infrastructure_rating']:.1f}" if r["infrastructure_rating"] else "N/A"
            ]
            table_data.append(row)
        
        print(tabulate(table_data, headers=headers, tablefmt="grid"))
    else:
        print("No destinations found matching criteria")


@cli.command()
@click.option('--budget', type=float, default=150, help='Daily budget')
@click.option('--safety-min', type=float, default=7, help='Minimum safety score')
@click.option('--visa-free', is_flag=True, help='Prefer visa-free destinations')
@click.option('--english', is_flag=True, help='Prefer English-speaking destinations')
def recommend(budget, safety_min, visa_free, english):
    """Get personalized recommendations"""
    profile = {
        "daily_budget": budget,
        "min_safety_score": safety_min,
        "visa_free_only": visa_free,
        "english_speaking_preferred": english
    }
    
    tool = PriorityQueryTool()
    recommendations = tool.get_traveler_recommendations(profile)
    
    if recommendations:
        print(f"\n{Fore.CYAN}═══ Personalized Recommendations ═══{Style.RESET_ALL}\n")
        
        for i, rec in enumerate(recommendations[:5], 1):
            print(f"{Fore.YELLOW}{i}. {rec['destination']} (Match: {rec['match_score']:.1f}/10){Style.RESET_ALL}")
            for reason in rec["why_recommended"]:
                print(f"   • {reason}")
            print()
    else:
        print("No suitable destinations found for your preferences")


if __name__ == "__main__":
    cli() 