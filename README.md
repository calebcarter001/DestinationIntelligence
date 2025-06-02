# Enhanced Destination Intelligence System

An advanced AI-powered system for discovering, analyzing, and providing comprehensive insights about travel destinations, now with traveler priority features for safety, cost, health, and accessibility analysis.

## Overview

This system uses CrewAI agents, LangChain tools, and multi-agent validation to:
- Discover destination information from web sources
- Extract and validate priority travel data (safety, cost, health, accessibility)
- Analyze themes and attractions with evidence-based confidence scoring
- Track seasonal variations and provide temporal insights
- Generate personalized recommendations based on traveler profiles
- Store insights in a structured database with ChromaDB for semantic search

## Key Features

### Core Intelligence Features
- **Web Discovery**: Automated discovery of destination information from multiple sources
- **Theme Analysis**: Evidence-based theme extraction with confidence scoring
- **Multi-Agent Validation**: Specialized agents for data validation and contradiction detection
- **Cultural Perspective**: Analysis of cultural context and local insights
- **Temporal Analysis**: Seasonal tracking and pattern recognition

### Priority Features (NEW)
- **Safety Analysis**: Crime indices, tourist police availability, travel advisories
- **Cost Intelligence**: Budget ranges, meal costs, accommodation prices
- **Health Information**: Required vaccinations, water safety, medical facilities
- **Accessibility Data**: Visa requirements, language barriers, infrastructure ratings
- **Weather Tracking**: Seasonal patterns, temperature ranges, rainfall data

### Validation & Quality
- **Priority Validation Agents**: Specialized validators for each data category
- **Confidence Scoring**: Multi-level confidence assessment for all insights
- **Source Credibility**: Authority-based weighting of information sources
- **Contradiction Resolution**: Automatic detection and resolution of conflicting data

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd DestinationIntelligence
```

2. Create and activate a virtual environment:
```bash
python -m venv demo_env
source demo_env/bin/activate  # On Windows: demo_env\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up environment variables:
```bash
cp .env.example .env
# Edit .env with your API keys:
# - GEMINI_API_KEY
# - BRAVE_SEARCH_API_KEY
# - OPENAI_API_KEY (optional)
# - JINA_API_KEY (optional)
```

5. Configure settings:
```bash
cp config.yaml.example config.yaml
# Edit config.yaml for your preferences
```

## Usage

### Running the Enhanced Agent Application

```bash
python run_enhanced_agent_app.py [--provider gemini|openai] [--model MODEL_NAME]
```

The application will interactively prompt for:
- Destination name
- Number of pages to analyze
- Confirmation to proceed

### Querying Priority Insights

The system includes a comprehensive CLI tool for querying priority data:

```bash
# View priority summary for a destination
python query_priority_insights.py summary "Bend, Oregon"

# Compare multiple destinations
python query_priority_insights.py compare "Portland, Oregon" "Seattle, Washington" "Denver, Colorado"

# Search by criteria
python query_priority_insights.py search --max-crime 30 --max-budget 100 --visa-free --safe-water

# Get personalized recommendations
python query_priority_insights.py recommend --budget 150 --safety-min 8 --visa-free --english
```

### Querying Enhanced Data

```bash
# Query all insights for a destination
python query_enhanced_data.py "Bend, Oregon"

# Query with specific theme filter
python query_enhanced_data.py "Bend, Oregon" --theme "outdoor"

# Query priority metrics only
python query_enhanced_data.py "Bend, Oregon" --priority-only
```

## System Architecture

### Enhanced Components

1. **Priority Data Extraction** (`priority_data_extraction_tool.py`)
   - Extracts safety, cost, health, and accessibility data from web content
   - Uses regex patterns and NLP for data extraction
   - Assigns source credibility scores

2. **Priority Aggregation** (`priority_aggregation_tool.py`)
   - Aggregates data from multiple sources
   - Uses median-based aggregation for numeric values
   - Generates confidence scores based on source agreement

3. **Validation Agents** (`priority_validation_agents.py`)
   - `SafetyValidationAgent`: Validates safety metrics and cross-checks consistency
   - `CostValidationAgent`: Ensures cost data validity and currency handling
   - `PracticalInfoAgent`: Validates health, visa, and accessibility information
   - `PriorityValidationOrchestrator`: Coordinates all validators

4. **Seasonal Tracking** (`seasonal_tracking.py`)
   - Tracks metrics variations across seasons
   - Predicts seasonal values based on historical data
   - Generates seasonal recommendations

5. **Export Manager** (`priority_export_manager.py`)
   - Generates JSON exports with full analysis
   - Creates traveler scorecards (0-10 ratings)
   - Produces comparative analysis between destinations
   - Provides personalized recommendations

### Database Schema

The system uses SQLite with the following enhanced tables:

- `destination_insights`: Theme and attraction insights
- `page_contents`: Cached web content
- `priority_metrics`: Safety, cost, health, accessibility metrics
- `priority_insights`: Validated priority insights with evidence

### Data Flow

1. **Discovery Phase**: Web discovery finds relevant content
2. **Extraction Phase**: Priority data and themes extracted from content
3. **Validation Phase**: Multi-agent validation ensures data quality
4. **Aggregation Phase**: Data from multiple sources combined
5. **Storage Phase**: Validated data stored in database
6. **Query Phase**: Interactive tools for data exploration

## Query Examples

### Priority Summary Output
```
═══ Priority Analysis: Bend, Oregon ═══

SAFETY
  Crime Index: 25.3/100
  Tourist Police: ✓ Available

COST
  Budget Range: $80-$250/day
  Currency: USD

HEALTH
  Water: Safe to drink
  Required Vaccinations: None

ACCESSIBILITY
  Visa: Not Required
  Infrastructure: 4.2/5
```

### Comparison Output
```
═══ Destination Comparison ═══

Destination    Overall  Safety  Cost   Health  Access
Portland       8.2      7.5     8.0    8.5     8.5
Seattle        7.8      7.0     6.5    8.5     9.0
Denver         8.5      8.0     7.5    9.0     8.5
```

## Configuration

### Priority Settings (config.yaml)
```yaml
priority_settings:
  enable_priority_discovery: true
  confidence_threshold: 0.6
  aggregation_method: "median"
  
validation_settings:
  strict_mode: false
  min_sources: 2
```

## Testing

Run the test suite:
```bash
# Test priority features
python test_priority_features.py

# Test enhanced integration
python test_enhanced_integration.py
```

## Troubleshooting

### Common Issues

1. **No priority data found**: Ensure web discovery is finding relevant content
2. **Low confidence scores**: Check if enough credible sources are available
3. **Validation failures**: Review validation rules in `priority_validation_agents.py`

### Logging

Logs are stored in the `logs/` directory:
- Application logs: `enhanced_app_run_*.log`
- Agent traces: `enhanced_langchain_agent_trace_*.log`

## Future Enhancements

- Real-time price tracking integration
- Social media sentiment analysis
- Weather API integration for live updates
- Mobile app companion
- Multi-language support

## Contributing

Please read CONTRIBUTING.md for details on our code of conduct and the process for submitting pull requests.

## License

This project is licensed under the MIT License - see the LICENSE file for details.