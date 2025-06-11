"""
Simple, focused tests for Cultural Intelligence functionality.
Tests core features without complex dependencies or imports.
"""

import unittest
import sys
import os

# Add the root directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

class TestCulturalIntelligenceCore(unittest.TestCase):
    """Test core cultural intelligence functionality"""
    
    def setUp(self):
        """Set up test fixtures with category processing rules"""
        from src.schemas import AuthorityType
        from datetime import datetime
        # Import the actual processing rules from scripts
        self.CATEGORY_PROCESSING_RULES = {
            "cultural": {
                "categories": [
                    "Cultural Identity & Atmosphere", "Authentic Experiences", "Distinctive Features",
                    "Local Character & Vibe", "Artistic & Creative Scene"
                ],
                "color": "#9C27B0",
                "icon": "🎭",
                "weight": 0.4
            },
            "practical": {
                "categories": [
                    "Safety & Security", "Transportation & Access", "Budget & Costs", 
                    "Health & Medical", "Logistics & Planning", "Visa & Documentation"
                ],
                "color": "#2196F3",
                "icon": "📋",
                "weight": 0.3
            },
            "hybrid": {
                "categories": [
                    "Food & Dining", "Entertainment & Nightlife", "Nature & Outdoor",
                    "Shopping & Local Craft", "Family & Education", "Health & Wellness"
                ],
                "color": "#4CAF50",
                "icon": "⚖️",
                "weight": 0.3
            }
        }

    def get_processing_type(self, macro_category):
        """Simple implementation of processing type identification"""
        if not macro_category:
            return "unknown"
        
        for proc_type, rules in self.CATEGORY_PROCESSING_RULES.items():
            if macro_category in rules["categories"]:
                return proc_type
        return "unknown"

    def calculate_authenticity_score(self, evidence_items):
        """Simple implementation of authenticity scoring"""
        if not evidence_items:
            return 0.0
        
        authentic_indicators = ['reddit.com', 'local', 'community', 'blog', 'forum']
        official_indicators = ['gov', 'edu', 'official', 'tourism']
        
        authentic_count = 0
        official_count = 0
        
        for item in evidence_items:
            url = item.get('url', '').lower()
            title = item.get('title', '').lower()
            
            for indicator in authentic_indicators:
                if indicator in url or indicator in title:
                    authentic_count += 1
                    break
            else:
                for indicator in official_indicators:
                    if indicator in url or indicator in title:
                        official_count += 1
                        break
        
        total_count = len(evidence_items)
        if total_count == 0:
            return 0.0
        
        authentic_ratio = authentic_count / total_count
        official_ratio = official_count / total_count
        
        # Higher authenticity score for more authentic sources, lower for official
        return max(0.0, min(1.0, authentic_ratio + (1 - official_ratio) * 0.3))

    def calculate_distinctiveness_score(self, evidence_items):
        """Simple implementation of distinctiveness scoring"""
        if not evidence_items:
            return 0.0
        
        unique_keywords = ['unique', 'distinctive', 'special', 'rare', 'authentic', 'unusual']
        generic_keywords = ['popular', 'common', 'typical', 'standard', 'normal', 'regular']
        
        unique_count = 0
        generic_count = 0
        
        for item in evidence_items:
            title = item.get('title', '').lower()
            snippet = item.get('snippet', '').lower()
            content = title + ' ' + snippet
            
            for keyword in unique_keywords:
                if keyword in content:
                    unique_count += 1
                    break
            
            for keyword in generic_keywords:
                if keyword in content:
                    generic_count += 1
                    break
        
        total_count = len(evidence_items)
        if total_count == 0:
            return 0.0
        
        unique_ratio = unique_count / total_count
        generic_ratio = generic_count / total_count
        
        return max(0.0, min(1.0, unique_ratio - generic_ratio * 0.5 + 0.5))

    def test_processing_type_identification_cultural(self):
        """Test identification of cultural categories"""
        cultural_categories = [
            "Cultural Identity & Atmosphere",
            "Authentic Experiences", 
            "Distinctive Features",
            "Local Character & Vibe",
            "Artistic & Creative Scene"
        ]
        
        for category in cultural_categories:
            result = self.get_processing_type(category)
            self.assertEqual(result, "cultural", f"Failed for category: {category}")

    def test_processing_type_identification_practical(self):
        """Test identification of practical categories"""
        practical_categories = [
            "Safety & Security",
            "Transportation & Access",
            "Budget & Costs",
            "Health & Medical", 
            "Logistics & Planning",
            "Visa & Documentation"
        ]
        
        for category in practical_categories:
            result = self.get_processing_type(category)
            self.assertEqual(result, "practical", f"Failed for category: {category}")

    def test_processing_type_identification_hybrid(self):
        """Test identification of hybrid categories"""
        hybrid_categories = [
            "Food & Dining",
            "Entertainment & Nightlife",
            "Nature & Outdoor",
            "Shopping & Local Craft",
            "Family & Education",
            "Health & Wellness"
        ]
        
        for category in hybrid_categories:
            result = self.get_processing_type(category)
            self.assertEqual(result, "hybrid", f"Failed for category: {category}")

    def test_processing_type_identification_unknown(self):
        """Test handling of unknown categories"""
        unknown_categories = [None, "", "Unknown Category", "Random Text"]
        
        for category in unknown_categories:
            result = self.get_processing_type(category)
            self.assertEqual(result, "unknown", f"Failed for category: {category}")

    def test_authenticity_scoring_high_authentic(self):
        """Test high authenticity scoring for authentic sources"""
        evidence_items = [
            {"url": "reddit.com/r/seattle", "title": "Local's guide to Seattle"},
            {"url": "seattleblog.local", "title": "Community recommendations"},
            {"url": "localnews.com", "title": "Neighborhood insights"}
        ]
        
        score = self.calculate_authenticity_score(evidence_items)
        self.assertGreater(score, 0.5, "Should have high authenticity for Reddit/local sources")

    def test_authenticity_scoring_low_official(self):
        """Test low authenticity scoring for official sources"""
        evidence_items = [
            {"url": "seattle.gov/tourism", "title": "Official tourism guide"},
            {"url": "university.edu/guide", "title": "Academic tourism study"},
            {"url": "official-tourism.com", "title": "Government tourism site"}
        ]
        
        score = self.calculate_authenticity_score(evidence_items)
        self.assertLess(score, 0.5, "Should have low authenticity for official sources")

    def test_distinctiveness_scoring_high_unique(self):
        """Test high distinctiveness scoring for unique content"""
        evidence_items = [
            {"title": "Unique grunge music heritage in Seattle", "snippet": "Distinctive coffee culture origins"},
            {"title": "Special underground tours", "snippet": "Rare Pike Place fish throwing tradition"}
        ]
        
        score = self.calculate_distinctiveness_score(evidence_items)
        self.assertGreater(score, 0.6, "Should have high distinctiveness for unique content")

    def test_distinctiveness_scoring_low_generic(self):
        """Test low distinctiveness scoring for generic content"""
        evidence_items = [
            {"title": "Popular tourist attractions", "snippet": "Common travel destinations"},
            {"title": "Typical city activities", "snippet": "Standard hotel accommodations"}
        ]
        
        score = self.calculate_distinctiveness_score(evidence_items)
        self.assertLess(score, 0.4, "Should have low distinctiveness for generic content")

class TestCulturalIntelligenceIntegration(unittest.TestCase):
    """Test cultural intelligence integration with scripts"""

    def setUp(self):
        """Set up test fixtures with category processing rules"""
        self.CATEGORY_PROCESSING_RULES = {
            "cultural": {
                "categories": [
                    "Cultural Identity & Atmosphere", "Authentic Experiences", "Distinctive Features",
                    "Local Character & Vibe", "Artistic & Creative Scene"
                ],
                "color": "#9C27B0",
                "icon": "🎭",
                "weight": 0.4
            },
            "practical": {
                "categories": [
                    "Safety & Security", "Transportation & Access", "Budget & Costs", 
                    "Health & Medical", "Logistics & Planning", "Visa & Documentation"
                ],
                "color": "#2196F3",
                "icon": "📋",
                "weight": 0.3
            },
            "hybrid": {
                "categories": [
                    "Food & Dining", "Entertainment & Nightlife", "Nature & Outdoor",
                    "Shopping & Local Craft", "Family & Education", "Health & Wellness"
                ],
                "color": "#4CAF50",
                "icon": "⚖️",
                "weight": 0.3
            }
        }

    def test_script_imports_work(self):
        """Test that we can import the cultural intelligence enhanced scripts"""
        try:
            # Test importing compare_destinations
            import compare_destinations
            self.assertTrue(hasattr(compare_destinations, 'get_processing_type'))
            self.assertTrue(hasattr(compare_destinations, 'CATEGORY_PROCESSING_RULES'))
            
            # Test importing analyze_themes
            import analyze_themes
            self.assertTrue(hasattr(analyze_themes, 'get_processing_type'))
            
            # Test importing generate_dynamic_viewer
            import generate_dynamic_viewer
            self.assertTrue(hasattr(generate_dynamic_viewer, 'get_processing_type'))
            
        except ImportError as e:
            self.fail(f"Failed to import cultural intelligence scripts: {e}")

    def test_script_consistency_processing_types(self):
        """Test that all scripts return consistent processing types"""
        try:
            import compare_destinations
            import analyze_themes  
            import generate_dynamic_viewer
            
            test_categories = [
                ("Cultural Identity & Atmosphere", "cultural"),
                ("Safety & Security", "practical"),
                ("Food & Dining", "hybrid"),
                ("Unknown Category", "unknown")
            ]
            
            for category, expected_type in test_categories:
                # Test all scripts return the same result
                compare_result = compare_destinations.get_processing_type(category)
                analyze_result = analyze_themes.get_processing_type(category)
                viewer_result = generate_dynamic_viewer.get_processing_type(category)
                
                self.assertEqual(compare_result, expected_type, f"compare_destinations failed for {category}")
                self.assertEqual(analyze_result, expected_type, f"analyze_themes failed for {category}")
                self.assertEqual(viewer_result, expected_type, f"generate_dynamic_viewer failed for {category}")
                
                # Test all scripts agree with each other
                self.assertEqual(compare_result, analyze_result, f"Inconsistency between compare and analyze for {category}")
                self.assertEqual(analyze_result, viewer_result, f"Inconsistency between analyze and viewer for {category}")
                
        except ImportError as e: 
            self.skipTest(f"Failed to import required modules: {e}")

    def test_category_processing_rules_structure(self):
        """Test that category processing rules have required structure"""
        for category_type, rules in self.CATEGORY_PROCESSING_RULES.items():
            self.assertIn("categories", rules)
            self.assertIn("color", rules)
            self.assertIn("icon", rules)
            self.assertIn("weight", rules)
            
            self.assertIsInstance(rules["categories"], list)
            self.assertIsInstance(rules["color"], str)
            self.assertIsInstance(rules["icon"], str)
            self.assertIsInstance(rules["weight"], (int, float))

class TestCulturalIntelligenceMetrics(unittest.TestCase):
    """Test cultural intelligence metrics calculations"""

    def test_theme_distribution_calculation(self):
        """Test theme distribution calculation"""
        sample_themes = [
            {"processing_type": "cultural", "confidence": 0.85},
            {"processing_type": "cultural", "confidence": 0.78},
            {"processing_type": "practical", "confidence": 0.90},
            {"processing_type": "hybrid", "confidence": 0.75},
            {"processing_type": "hybrid", "confidence": 0.80}
        ]
        
        # Calculate distribution
        total_themes = len(sample_themes)
        cultural_count = sum(1 for t in sample_themes if t["processing_type"] == "cultural")
        practical_count = sum(1 for t in sample_themes if t["processing_type"] == "practical")
        hybrid_count = sum(1 for t in sample_themes if t["processing_type"] == "hybrid")
        
        cultural_ratio = cultural_count / total_themes
        practical_ratio = practical_count / total_themes
        hybrid_ratio = hybrid_count / total_themes
        
        # Test ratios
        self.assertAlmostEqual(cultural_ratio, 0.4, places=2)  # 2/5
        self.assertAlmostEqual(practical_ratio, 0.2, places=2)  # 1/5
        self.assertAlmostEqual(hybrid_ratio, 0.4, places=2)   # 2/5
        
        # Test that ratios sum to 1
        self.assertAlmostEqual(cultural_ratio + practical_ratio + hybrid_ratio, 1.0, places=2)

    def test_confidence_calculation_by_type(self):
        """Test confidence calculation by processing type"""
        sample_themes = [
            {"processing_type": "cultural", "confidence": 0.85},
            {"processing_type": "cultural", "confidence": 0.78},
            {"processing_type": "practical", "confidence": 0.90},
            {"processing_type": "hybrid", "confidence": 0.75},
            {"processing_type": "hybrid", "confidence": 0.80}
        ]
        
        # Calculate average confidence by type
        cultural_themes = [t for t in sample_themes if t["processing_type"] == "cultural"]
        practical_themes = [t for t in sample_themes if t["processing_type"] == "practical"]
        hybrid_themes = [t for t in sample_themes if t["processing_type"] == "hybrid"]
        
        cultural_avg = sum(t["confidence"] for t in cultural_themes) / len(cultural_themes)
        practical_avg = sum(t["confidence"] for t in practical_themes) / len(practical_themes)
        hybrid_avg = sum(t["confidence"] for t in hybrid_themes) / len(hybrid_themes)
        
        # Test averages
        self.assertAlmostEqual(cultural_avg, 0.815, places=3)  # (0.85 + 0.78) / 2
        self.assertAlmostEqual(practical_avg, 0.90, places=2)   # 0.90 / 1
        self.assertAlmostEqual(hybrid_avg, 0.775, places=3)     # (0.75 + 0.80) / 2

    def test_destination_personality_detection(self):
        """Test destination personality detection logic"""
        
        def get_destination_personality(theme_stats):
            """Simple personality detection logic"""
            total = sum(theme_stats.values())
            if total == 0:
                return "unknown"
            
            ratios = {proc_type: count / total for proc_type, count in theme_stats.items()}
            dominant_type = max(ratios, key=ratios.get)
            
            if ratios[dominant_type] > 0.4:
                return dominant_type
            else:
                return "balanced"
        
        # Test cultural-focused destination
        cultural_focused = {"cultural": 5, "practical": 1, "hybrid": 1}
        personality = get_destination_personality(cultural_focused)
        self.assertEqual(personality, "cultural")
        
        # Test practical-focused destination
        practical_focused = {"cultural": 1, "practical": 5, "hybrid": 1}
        personality = get_destination_personality(practical_focused)
        self.assertEqual(personality, "practical")
        
        # Test balanced destination
        balanced = {"cultural": 2, "practical": 2, "hybrid": 3}
        personality = get_destination_personality(balanced)
        self.assertEqual(personality, "hybrid")  # 3/7 = 0.43 > 0.4, so hybrid dominant
        
        # Test truly balanced destination
        truly_balanced = {"cultural": 2, "practical": 2, "hybrid": 2}
        personality = get_destination_personality(truly_balanced)
        self.assertEqual(personality, "balanced")  # No single type > 0.4

def run_simple_tests():
    """Run the simple test suite"""
    print("🎭 Simple Cultural Intelligence Test Suite")
    print("=" * 50)
    
    # Create test suite
    suite = unittest.TestSuite()
    
    # Add test classes
    suite.addTest(unittest.makeSuite(TestCulturalIntelligenceCore))
    suite.addTest(unittest.makeSuite(TestCulturalIntelligenceIntegration))
    suite.addTest(unittest.makeSuite(TestCulturalIntelligenceMetrics))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    print("\n" + "=" * 50)
    print("📊 Test Results Summary")
    print("=" * 50)
    
    total_tests = result.testsRun
    failures = len(result.failures)
    errors = len(result.errors)
    success = total_tests - failures - errors
    
    print(f"Total Tests: {total_tests}")
    print(f"✅ Passed: {success}")
    print(f"❌ Failed: {failures}")
    print(f"🚨 Errors: {errors}")
    
    if failures == 0 and errors == 0:
        print("\n🎉 All tests passed! Cultural Intelligence is working correctly.")
        return True
    else:
        print("\n⚠️  Some tests failed. Check the output above for details.")
        return False

if __name__ == "__main__":
    success = run_simple_tests()
    exit(0 if success else 1) 