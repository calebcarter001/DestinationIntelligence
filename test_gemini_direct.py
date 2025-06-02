#!/usr/bin/env python3

import os
import sys
import asyncio
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

# Load environment variables
load_dotenv()

async def test_gemini_endpoint(destination="Bend, Oregon"):
    """Test Gemini API endpoint directly"""
    
    # Get API key
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("❌ GEMINI_API_KEY not found in environment")
        return
    
    # Initialize Gemini LLM
    model_name = os.getenv("GEMINI_MODEL_NAME", "gemini-2.0-flash")
    print(f"🤖 Testing Gemini API with model: {model_name}")
    print(f"🎯 Destination: {destination}")
    
    try:
        llm = ChatGoogleGenerativeAI(
            model=model_name,
            google_api_key=api_key,
            temperature=0.1
        )
        
        # Simple test prompt
        test_prompt = f"""
        Analyze the destination "{destination}" and identify 3 key themes that make it special.
        
        For each theme, provide:
        1. Theme name
        2. Brief description 
        3. Confidence level (High/Medium/Low)
        
        Format your response as JSON with this structure:
        {{
            "destination": "{destination}",
            "themes": [
                {{"name": "Theme Name", "description": "Description", "confidence": "High"}}
            ]
        }}
        """
        
        print("🚀 Sending request to Gemini...")
        response = await llm.ainvoke(test_prompt)
        
        print("✅ Response received:")
        print("=" * 50)
        print(response.content)
        print("=" * 50)
        
        return True
        
    except Exception as e:
        print(f"❌ Error calling Gemini API: {e}")
        return False

def main():
    """Main function with argument parsing"""
    destination = "Bend, Oregon"  # Default
    
    # Parse command line arguments
    if len(sys.argv) > 1:
        destination = " ".join(sys.argv[1:])
    
    print(f"🎯 Testing Gemini endpoint for destination: {destination}")
    
    # Run async test
    result = asyncio.run(test_gemini_endpoint(destination))
    
    if result:
        print("✅ Gemini API test successful!")
    else:
        print("❌ Gemini API test failed!")
        sys.exit(1)

if __name__ == "__main__":
    main() 