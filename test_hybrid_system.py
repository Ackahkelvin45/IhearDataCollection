#!/usr/bin/env python
"""
Test script for the Hybrid Database + RAG system
"""

import os
import sys
import django

# Setup Django
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'datacollection.settings')
sys.path.insert(0, os.path.dirname(__file__))
django.setup()

from chatbot.services.intent_classifier import IntentClassifier, classify_question_intent
from chatbot.services.dataset_service import DatasetService

def test_intent_classification():
    """Test the intent classification system"""
    print("🤖 TESTING INTENT CLASSIFICATION")
    print("=" * 50)
    
    classifier = IntentClassifier()
    
    test_questions = [
        # Numeric questions
        ("How many audio recordings do we have?", "NUMERIC"),
        ("What is the average duration of recordings?", "NUMERIC"),
        ("Show me the top 5 categories by count", "NUMERIC"),
        ("Count recordings from rural areas", "NUMERIC"),
        
        # Explanatory questions
        ("Why do some recordings have low quality?", "EXPLANATORY"),
        ("Explain the pattern in community participation", "EXPLANATORY"),
        ("What trends do you see in the data?", "EXPLANATORY"),
        ("Tell me about the education program outcomes", "EXPLANATORY"),
        
        # Mixed questions
        ("How many students failed and why?", "MIXED"),
        ("What's the participation rate and what factors affect it?", "MIXED"),
    ]
    
    correct = 0
    total = len(test_questions)
    
    for question, expected_intent in test_questions:
        actual_intent = classifier.classify_intent(question)
        status = "✅" if actual_intent == expected_intent else "❌"
        
        print(f"{status} '{question}'")
        print(f"   Expected: {expected_intent} | Actual: {actual_intent}")
        
        if actual_intent == expected_intent:
            correct += 1
        print()
    
    accuracy = correct / total * 100
    print(f"🎯 Accuracy: {correct}/{total} ({accuracy:.1f}%)")
    print()

def test_dataset_service():
    """Test the dataset service with different question types"""
    print("📊 TESTING DATASET SERVICE")
    print("=" * 50)
    
    service = DatasetService()
    
    test_questions = [
        "How many audio recordings are in the system?",
        "Why might some recordings have poor quality?",
        "What are the main categories in our data?",
        "Show me recordings from the education category",
        "How many participants and what engagement patterns do you see?",
    ]
    
    for question in test_questions:
        print(f"❓ Question: {question}")
        
        try:
            result = service.query_dataset(question)
            print(f"📋 Intent: {result['intent']}")
            print(f"🎯 Answer: {result['answer'][:100]}...")
            print(f"⏱️  Processing: {result['processing_time']:.2f}s")
        except Exception as e:
            print(f"❌ Error: {e}")
        
        print("-" * 50)

def test_routing_info():
    """Test detailed routing information"""
    print("🔀 TESTING ROUTING INFORMATION")
    print("=" * 50)
    
    classifier = IntentClassifier()
    
    questions = [
        "How many recordings do we have?",
        "Explain the data quality issues",
        "How many failed recordings and why?",
    ]
    
    for question in questions:
        print(f"❓ '{question}'")
        routing = classifier.get_routing_info(question)
        
        print(f"   Intent: {routing['intent']}")
        print(f"   Confidence: {routing['confidence']}")
        print(f"   Tools: {routing['suggested_tools']}")
        print(f"   Reasoning: {routing['reasoning']}")
        print()

def main():
    """Run all tests"""
    print("🚀 TESTING HYBRID DATABASE + RAG SYSTEM")
    print("=" * 60)
    print()
    
    try:
        test_intent_classification()
        test_routing_info()
        test_dataset_service()
        
        print("🎉 ALL TESTS COMPLETED!")
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
