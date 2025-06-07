#!/usr/bin/env python3
"""
Check the status of the continuous learning feature.
"""

import os
import json
import sys
from datetime import datetime

def check_continuous_learning_status():
    """Check if continuous learning is working and provide status report."""
    
    print("=" * 60)
    print("🧠 CONTINUOUS LEARNING STATUS REPORT")
    print("=" * 60)
    
    # Check if continuous learning module exists
    continuous_learning_file = "continuous_learning.py"
    if os.path.exists(continuous_learning_file):
        print("✅ Continuous learning module found")
    else:
        print("❌ Continuous learning module NOT found")
        return
    
    # Check data files
    data_files = {
        "signal_history.json": "Signal History",
        "user_feedback.json": "User Feedback", 
        "model_performance.json": "Model Performance",
        "binary_options_model.pt": "ML Model"
    }
    
    print("\n📁 DATA FILES STATUS:")
    for file_path, description in data_files.items():
        if os.path.exists(file_path):
            file_size = os.path.getsize(file_path)
            print(f"✅ {description}: {file_path} ({file_size:,} bytes)")
        else:
            print(f"❌ {description}: {file_path} (NOT FOUND)")
    
    # Check signal history content
    print("\n📊 SIGNAL HISTORY ANALYSIS:")
    if os.path.exists("signal_history.json"):
        try:
            with open("signal_history.json", 'r') as f:
                signal_history = json.load(f)
            
            total_signals = len(signal_history)
            print(f"📈 Total signals recorded: {total_signals}")
            
            if total_signals > 0:
                # Analyze signal status
                pending_signals = sum(1 for s in signal_history if s.get('status') == 'pending')
                completed_signals = total_signals - pending_signals
                
                print(f"⏳ Pending signals: {pending_signals}")
                print(f"✅ Completed signals: {completed_signals}")
                
                # Check recent activity
                recent_signals = [s for s in signal_history if s.get('timestamp')]
                if recent_signals:
                    latest_signal = max(recent_signals, key=lambda x: x['timestamp'])
                    print(f"🕒 Latest signal: {latest_signal['timestamp']}")
                    print(f"💱 Latest pair: {latest_signal.get('pair', 'Unknown')}")
                    print(f"📊 Latest prediction: {latest_signal.get('prediction', 'Unknown')}")
                
                # Check if signals have outcomes (needed for learning)
                signals_with_outcomes = sum(1 for s in signal_history if s.get('outcome_price') is not None)
                print(f"🎯 Signals with outcomes: {signals_with_outcomes}")
                
                if signals_with_outcomes < 50:
                    print(f"⚠️  Need {50 - signals_with_outcomes} more completed signals for model retraining")
                else:
                    print("✅ Sufficient data for model retraining")
            else:
                print("📭 No signals recorded yet")
                
        except Exception as e:
            print(f"❌ Error reading signal history: {str(e)}")
    
    # Check user feedback
    print("\n💬 USER FEEDBACK ANALYSIS:")
    if os.path.exists("user_feedback.json"):
        try:
            with open("user_feedback.json", 'r') as f:
                user_feedback = json.load(f)
            
            feedback_count = len(user_feedback)
            print(f"📝 Total feedback entries: {feedback_count}")
            
            if feedback_count > 0:
                # Analyze feedback
                ratings = [f.get('rating', 0) for f in user_feedback if f.get('rating')]
                if ratings:
                    avg_rating = sum(ratings) / len(ratings)
                    print(f"⭐ Average rating: {avg_rating:.2f}/5")
                
                traded_feedback = sum(1 for f in user_feedback if f.get('traded'))
                print(f"💰 Feedback from actual trades: {traded_feedback}")
            else:
                print("📭 No user feedback recorded yet")
                
        except Exception as e:
            print(f"❌ Error reading user feedback: {str(e)}")
    
    # Check model performance
    print("\n📈 MODEL PERFORMANCE ANALYSIS:")
    if os.path.exists("model_performance.json"):
        try:
            with open("model_performance.json", 'r') as f:
                performance = json.load(f)
            
            metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'win_rate']
            for metric in metrics:
                values = performance.get(metric, [])
                if values:
                    latest_value = values[-1] if values else 0
                    print(f"📊 Latest {metric}: {latest_value:.3f}")
                else:
                    print(f"📊 {metric}: No data")
                    
        except Exception as e:
            print(f"❌ Error reading model performance: {str(e)}")
    
    # Check if continuous learning is enabled in main
    print("\n🔧 INTEGRATION STATUS:")
    try:
        # Try to import and check if it's enabled
        sys.path.append('.')
        
        # Check main.py for continuous learning initialization
        if os.path.exists("main.py"):
            with open("main.py", 'r') as f:
                main_content = f.read()
            
            if "ContinuousLearningSystem" in main_content:
                print("✅ Continuous learning integrated in main.py")
                
                if "start_learning_thread" in main_content:
                    print("✅ Learning thread startup code found")
                else:
                    print("⚠️  Learning thread startup code not found")
                    
                if "CONTINUOUS_LEARNING_ENABLED" in main_content:
                    print("✅ Continuous learning enable/disable flag found")
                else:
                    print("⚠️  Enable/disable flag not found")
            else:
                print("❌ Continuous learning NOT integrated in main.py")
        
    except Exception as e:
        print(f"❌ Error checking integration: {str(e)}")
    
    # Recommendations
    print("\n💡 RECOMMENDATIONS:")
    
    # Check if we have enough data
    signal_count = 0
    if os.path.exists("signal_history.json"):
        try:
            with open("signal_history.json", 'r') as f:
                signal_history = json.load(f)
            signal_count = len(signal_history)
        except:
            pass
    
    if signal_count == 0:
        print("🚀 Start generating signals to begin data collection")
    elif signal_count < 50:
        print(f"📊 Continue collecting data ({signal_count}/50 signals for retraining)")
    else:
        print("✅ Sufficient data available for continuous learning")
    
    # Check if feedback is being collected
    feedback_count = 0
    if os.path.exists("user_feedback.json"):
        try:
            with open("user_feedback.json", 'r') as f:
                user_feedback = json.load(f)
            feedback_count = len(user_feedback)
        except:
            pass
    
    if feedback_count == 0:
        print("💬 Encourage users to provide feedback on signals")
    
    print("\n" + "=" * 60)
    print("📋 SUMMARY:")
    
    if signal_count > 0:
        print(f"✅ Continuous learning is COLLECTING DATA ({signal_count} signals)")
    else:
        print("⚠️  Continuous learning is READY but no data collected yet")
    
    if signal_count >= 50:
        print("✅ Ready for model retraining")
    else:
        print(f"⏳ Need {max(0, 50 - signal_count)} more signals for retraining")
    
    print("=" * 60)

if __name__ == "__main__":
    check_continuous_learning_status()
