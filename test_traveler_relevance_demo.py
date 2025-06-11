#!/usr/bin/env python3
"""
Demo script to test the enhanced traveler relevance algorithm
Shows how different types of themes would be ranked
"""

def calculate_traveler_relevance_demo(theme_name, theme_description, theme_category):
    """Demo version of the enhanced traveler relevance calculation"""
    combined_text = f"{theme_name.lower()} {theme_description.lower()}"
    theme_category = theme_category.lower()
    
    # Category-based base scoring
    base_relevance = 0.5
    
    high_tourist_categories = [
        "entertainment & nightlife", "cultural identity & atmosphere", "nature & outdoor",
        "authentic experiences", "distinctive features", "artistic & creative scene"
    ]
    
    medium_tourist_categories = [
        "food & dining", "shopping & local craft", "family & education"
    ]
    
    low_tourist_categories = [
        "health & medical", "transportation & access", "budget & costs",
        "visa & documentation", "logistics & planning", "safety & security"
    ]
    
    if theme_category in high_tourist_categories:
        base_relevance = 0.8
    elif theme_category in medium_tourist_categories:
        base_relevance = 0.6
    elif theme_category in low_tourist_categories:
        base_relevance = 0.3
    
    # Emotional Language Detection
    emotional_keywords = [
        "breathtaking", "stunning", "spectacular", "vibrant", "thrilling", 
        "intimate", "tranquil", "lively", "genuine", "authentic", "picturesque",
        "serene", "bustling", "hidden", "secret", "romantic", "magical"
    ]
    
    emotional_boost = sum(0.2 for keyword in emotional_keywords 
                         if keyword in combined_text) * 0.8
    
    # Experience vs Service Classification
    experiential_keywords = [
        "adventures", "exploration", "discovery", "journey", "escape", "retreat",
        "immersion", "celebration", "entertainment", "dining experience", "tours",
        "excursions", "activities", "experiences", "attractions", "sightseeing"
    ]
    
    service_keywords = [
        "fitness", "medical", "hospital", "clinic", "grocery", "utilities",
        "administrative", "residential", "office", "government"
    ]
    
    experience_boost = sum(0.15 for keyword in experiential_keywords 
                          if keyword in combined_text)
    
    service_penalty = sum(0.25 for keyword in service_keywords 
                         if keyword in combined_text)
    
    # Visual/Sensory Appeal Detection
    visual_appeal_keywords = [
        "scenic", "views", "landscapes", "sunset", "photography",
        "architecture", "art", "colorful", "beautiful", "panoramic"
    ]
    
    visual_boost = sum(0.15 for keyword in visual_appeal_keywords 
                      if keyword in combined_text)
    
    # Traveler Persona Targeting
    persona_boosts = 0.0
    
    # Romance
    romantic_keywords = ["romantic", "intimate", "couples", "sunset", "wine", "spa"]
    if any(keyword in combined_text for keyword in romantic_keywords):
        persona_boosts += 0.15
    
    # Adventure
    adventure_keywords = ["adventure", "hiking", "climbing", "extreme", "outdoor"]
    if any(keyword in combined_text for keyword in adventure_keywords):
        persona_boosts += 0.18
    
    # Culture
    culture_keywords = ["cultural", "traditional", "historical", "heritage", "authentic"]
    if any(keyword in combined_text for keyword in culture_keywords):
        persona_boosts += 0.15
    
    # Mundane penalties
    mundane_penalties = [
        "fitness center", "gym", "hospital", "clinic", "medical center",
        "grocery store", "supermarket", "pharmacy"
    ]
    
    mundane_penalty = sum(0.3 for mundane in mundane_penalties 
                         if mundane in combined_text)
    
    # Quality indicators
    quality_indicators = [
        "top-rated", "must-see", "world-class", "famous", "iconic"
    ]
    
    quality_boost = sum(0.1 for indicator in quality_indicators 
                       if indicator in combined_text)
    
    # Calculate final relevance
    final_relevance = (base_relevance + 
                      emotional_boost + 
                      experience_boost + 
                      visual_boost + 
                      persona_boosts + 
                      quality_boost - 
                      service_penalty - 
                      mundane_penalty)
    
    # Apply bounds
    final_relevance = max(0.05, min(final_relevance, 1.0))
    
    return final_relevance, {
        "base": base_relevance,
        "emotional": emotional_boost,
        "experience": experience_boost,
        "visual": visual_boost,
        "persona": persona_boosts,
        "quality": quality_boost,
        "service_penalty": -service_penalty,
        "mundane_penalty": -mundane_penalty
    }

def demo_traveler_relevance():
    """Demonstrate the enhanced traveler relevance algorithm"""
    
    # Test themes - mix of tourist-appealing vs mundane
    test_themes = [
        {
            "name": "Scenic Views", 
            "description": "Breathtaking panoramic views of mountains and valleys perfect for photography",
            "category": "Nature & Outdoor"
        },
        {
            "name": "Food Tours", 
            "description": "Authentic dining experiences exploring local cuisine and hidden culinary gems",
            "category": "Food & Dining"
        },
        {
            "name": "Adventure Hiking", 
            "description": "Thrilling outdoor adventures through spectacular landscapes and scenic trails",
            "category": "Nature & Outdoor"
        },
        {
            "name": "Cultural Heritage Sites", 
            "description": "Authentic historical landmarks showcasing traditional architecture and cultural heritage",
            "category": "Cultural Identity & Atmosphere"
        },
        {
            "name": "Romantic Sunset Spots", 
            "description": "Intimate and picturesque locations perfect for couples watching stunning sunsets",
            "category": "Entertainment & Nightlife"
        },
        {
            "name": "Fitness", 
            "description": "Local fitness centers and gym facilities for exercise routines",
            "category": "Health & Wellness"
        },
        {
            "name": "Educational", 
            "description": "Educational institutions and learning facilities in the area",
            "category": "Family & Education"
        },
        {
            "name": "Medical Services", 
            "description": "Hospital and clinic facilities providing medical care and health services",
            "category": "Health & Medical"
        },
        {
            "name": "Live Music Venues", 
            "description": "Vibrant entertainment venues featuring live music and lively performances",
            "category": "Entertainment & Nightlife"
        },
        {
            "name": "Artisan Markets", 
            "description": "Authentic local craft markets with traditional artisans and unique handmade treasures",
            "category": "Shopping & Local Craft"
        }
    ]
    
    print("🎯 ENHANCED TRAVELER RELEVANCE ALGORITHM DEMO")
    print("=" * 60)
    print()
    
    results = []
    for theme in test_themes:
        relevance, breakdown = calculate_traveler_relevance_demo(
            theme["name"], theme["description"], theme["category"]
        )
        results.append((relevance, theme, breakdown))
    
    # Sort by relevance (highest first)
    results.sort(key=lambda x: x[0], reverse=True)
    
    print("📊 RANKING BY TOURIST RELEVANCE:")
    print("-" * 60)
    
    for i, (relevance, theme, breakdown) in enumerate(results, 1):
        print(f"{i:2d}. {theme['name']:<25} | Score: {relevance:.3f}")
        print(f"    Category: {theme['category']}")
        print(f"    Breakdown: {breakdown}")
        print()
    
    print("🎉 KEY IMPROVEMENTS:")
    print("✅ Tourist-appealing themes (Scenic Views, Food Tours) rank highest")
    print("✅ Emotional language ('breathtaking', 'authentic') gets rewarded")
    print("✅ Experience keywords ('tours', 'adventures') boost relevance")
    print("✅ Mundane themes (Fitness, Medical) get penalized appropriately")
    print("✅ Persona targeting (romantic, cultural, adventure) works")

if __name__ == "__main__":
    demo_traveler_relevance() 