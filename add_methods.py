#!/usr/bin/env python3

import os
import re

# Read the file
with open('src/core/web_discovery_logic.py', 'r') as f:
    content = f.read()

# Define the methods to add
methods_to_add = '''
    def _get_default_tourist_gateway_keywords(self):
        """Load tourist gateway keywords from external JSON file or return defaults."""
        import os
        import json
        
        try:
            # Try to load from external config file
            config_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "config")
            tourist_keywords_path = os.path.join(config_dir, "default_tourist_gateway_keywords.json")
            
            if os.path.exists(tourist_keywords_path):
                with open(tourist_keywords_path, 'r') as f:
                    external_keywords = json.load(f)
                    # Combine all categories from the external config
                    all_keywords = []
                    for category_keywords in external_keywords.values():
                        all_keywords.extend(category_keywords)
                    self.logger.info(f"✅ Loaded {len(all_keywords)} tourist gateway keywords from {tourist_keywords_path}")
                    return all_keywords
            else:
                self.logger.warning(f"Tourist gateway keywords file not found: {tourist_keywords_path}")
        except Exception as e:
            self.logger.error(f"Error loading tourist gateway keywords from external config: {e}")
        
        # Fallback to hardcoded defaults
        self.logger.info("Using default hardcoded tourist gateway keywords")
        return [
            "flagstaff", "sedona", "moab", "aspen", "jackson", "estes park", "mammoth lakes",
            "springdale", "gatlinburg", "bar harbor", "yosemite village", "glacier village",
            "napa", "carmel", "sausalito", "mendocino", "capitola", "pacific grove",
            "big sur", "half moon bay", "sonoma", "healdsburg", "calistoga",
            "whistler", "banff", "jasper", "lake tahoe", "steamboat springs", "vail",
            "breckenridge", "park city", "sun valley", "taos", "santa fe"
        ]

    def _get_authority_domains(self):
        """Load authority domains from external JSON file or return defaults."""
        import os
        import json
        
        try:
            # Try to load from external config file
            config_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "config")
            authority_domains_path = os.path.join(config_dir, "travel_authority_domains.json")
            
            if os.path.exists(authority_domains_path):
                with open(authority_domains_path, 'r') as f:
                    authority_data = json.load(f)
                    # Extract just the domain names from the URL field
                    authority_domains = []
                    for item in authority_data:
                        if 'url' in item:
                            # Extract domain from URL
                            domain = item['url'].replace('https://', '').replace('http://', '').replace('www.', '').split('/')[0]
                            authority_domains.append(domain)
                    self.logger.info(f"✅ Loaded {len(authority_domains)} authority domains from {authority_domains_path}")
                    return authority_domains
            else:
                self.logger.warning(f"Authority domains file not found: {authority_domains_path}")
        except Exception as e:
            self.logger.error(f"Error loading authority domains from external config: {e}")
        
        # Fallback to hardcoded defaults
        self.logger.info("Using default hardcoded authority domains")
        return [
            "wikipedia.org", "tripadvisor.com", "lonelyplanet.com",
            "timeout.com", "fodors.com", "frommers.com", 
            "visitnsw.com", "sydney.com", "australia.com",
            "gov.au", "booking.com", "expedia.com"
        ]
'''

# Find a good place to insert the methods - before the _is_tourist_gateway_destination method
insertion_point = content.find('    def _is_tourist_gateway_destination(self, destination: str) -> bool:')
if insertion_point == -1:
    print('Could not find insertion point')
    exit(1)

# Insert the methods
new_content = content[:insertion_point] + methods_to_add + '\n\n' + content[insertion_point:]

# Write back to file
with open('src/core/web_discovery_logic.py', 'w') as f:
    f.write(new_content)

print('✅ Successfully added missing methods to web_discovery_logic.py') 