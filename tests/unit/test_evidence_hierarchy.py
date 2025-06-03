import unittest
from datetime import datetime
from src.core.evidence_hierarchy import EvidenceHierarchy, SourceCategory, EvidenceType
from src.schemas import AuthorityType
from src.core.enhanced_data_models import LocalAuthority


class TestEvidenceHierarchy(unittest.TestCase):
    
    def test_classify_source_government(self):
        urls = [
            "https://www.oregon.gov/tourism/",
            "https://tourism.ca.gov/",
            "https://www.city.portland.or.us/"
        ]
        
        for url in urls:
            result = EvidenceHierarchy.classify_source(url)
            self.assertEqual(result, SourceCategory.GOVERNMENT)
    
    def test_classify_source_academic(self):
        urls = [
            "https://www.stanford.edu/research/tourism",
            "https://scholar.google.com/paper"
        ]
        
        for url in urls:
            result = EvidenceHierarchy.classify_source(url)
            self.assertEqual(result, SourceCategory.ACADEMIC)
    
    def test_classify_source_social(self):
        urls = [
            "https://www.instagram.com/bendoregon",
            "https://facebook.com/visitbend",
            "https://twitter.com/bendoregon"
        ]
        
        for url in urls:
            result = EvidenceHierarchy.classify_source(url)
            self.assertEqual(result, SourceCategory.SOCIAL)
    
    def test_get_source_authority_with_decay(self):
        url = "https://www.oregon.gov/tourism/"
        old_date = datetime(2020, 1, 1)
        recent_date = datetime(2024, 1, 1)
        
        weight_old, type_old = EvidenceHierarchy.get_source_authority(url, old_date)
        weight_recent, type_recent = EvidenceHierarchy.get_source_authority(url, recent_date)
        
        # Recent should have higher weight due to less decay
        self.assertGreater(weight_recent, weight_old)
        self.assertEqual(type_old, type_recent)
    
    def test_calculate_evidence_diversity(self):
        diverse_sources = [
            "https://www.oregon.gov/tourism/",
            "https://www.stanford.edu/research/",
            "https://lonelyplanet.com/oregon",
            "https://instagram.com/bendoregon"
        ]
        
        diversity = EvidenceHierarchy.calculate_evidence_diversity(diverse_sources)
        self.assertGreater(diversity, 0.0)
    
    def test_is_local_source_with_country_code(self):
        ca_url = "https://visitvancouver.ca/attractions"
        is_local_ca = EvidenceHierarchy.is_local_source(ca_url, "CA")
        self.assertTrue(is_local_ca)
    
    def test_classify_local_authority_producer(self):
        url = "https://benddistillery.com"
        content = "Visit our maple farm and distillery for tours"
        
        authority = EvidenceHierarchy.classify_local_authority(url, content)
        self.assertIsInstance(authority, LocalAuthority)
        self.assertIsNotNone(authority.expertise_domain)
    
    def test_classify_local_authority_default(self):
        url = "https://genericwebsite.com"
        content = "Generic content"
        
        authority = EvidenceHierarchy.classify_local_authority(url, content)
        self.assertIsInstance(authority, LocalAuthority)
        self.assertEqual(authority.authority_type, AuthorityType.RESIDENT)


if __name__ == '__main__':
    unittest.main() 