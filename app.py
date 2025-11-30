from flask import Flask, render_template, request, jsonify, session, redirect, url_for, flash
from flask_pymongo import PyMongo
import pandas as pd
import os
import json
import logging
from datetime import datetime
from config import Config
from utils.feature_extractor import AdvancedFeatureExtractor
from werkzeug.security import generate_password_hash, check_password_hash

# Initialize Flask app
app = Flask(__name__)
app.config.from_object(Config)
app.secret_key = 'keystroke-dynamics-optimized-secret-key'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Initialize MongoDB
mongo = PyMongo(app)

# Global variables
verifier = None
feature_extractor = AdvancedFeatureExtractor()

def get_verifier():
    """Lazy loader for verifier"""
    global verifier
    if verifier is None:
        try:
            from verify_optimized import OptimizedKeystrokeVerifier
            verifier = OptimizedKeystrokeVerifier()
        except Exception as e:
            logging.error(f"Failed to load verifier: {e}")
            verifier = None
    return verifier

def get_next_sample_number():
    """Get the next available sample number"""
    if not os.path.exists(Config.DATA_DIR):
        os.makedirs(Config.DATA_DIR)
    
    existing_files = [f for f in os.listdir(Config.DATA_DIR) 
                     if f.startswith('sample') and f.endswith('.csv')]
    
    if not existing_files:
        return 1
    
    # Extract numbers from existing files
    numbers = []
    for filename in existing_files:
        try:
            num = int(filename.replace('sample', '').replace('.csv', ''))
            numbers.append(num)
        except ValueError:
            continue
    
    return max(numbers) + 1 if numbers else 1

def save_keystroke_data(keystrokes):
    """Save keystroke data to a CSV file with next available sample number"""
    try:
        # Get next sample number
        sample_number = get_next_sample_number()
        filename = f'sample{sample_number}.csv'
        filepath = os.path.join(Config.DATA_DIR, filename)
        
        # Convert keystrokes to DataFrame
        df = pd.DataFrame(keystrokes)
        
        # Save to CSV
        df.to_csv(filepath, index=False)
        
        logging.info(f"Saved keystroke data to {filepath}")
        return filepath, sample_number
    
    except Exception as e:
        logging.error(f"Failed to save keystroke data: {e}")
        return None, None

@app.route('/')
def home():
    """Landing page"""
    if 'user_id' in session:
        return redirect(url_for('dashboard'))
    return render_template('home.html')

@app.route('/dashboard')
def dashboard():
    """Dashboard page"""
    if 'user_id' not in session:
        return redirect(url_for('login'))
    return render_template('index.html', user_name=session.get('user_name'))

@app.route('/register', methods=['GET', 'POST'])
def register():
    """Registration page"""
    if request.method == 'POST':
        name = request.form.get('name')
        email = request.form.get('email')
        password = request.form.get('password')
        confirm_password = request.form.get('confirm_password')

        if password != confirm_password:
            # You might want to pass this error to the template
            return render_template('register.html', error='Passwords do not match')

        existing_user = mongo.db.users.find_one({'email': email})
        if existing_user:
            return render_template('register.html', error='Email already exists')

        hashed_password = generate_password_hash(password)
        mongo.db.users.insert_one({
            'name': name,
            'email': email,
            'password': hashed_password,
            'created_at': datetime.utcnow()
        })
        
        return redirect(url_for('login'))

    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    """Login page"""
    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')

        user = mongo.db.users.find_one({'email': email})
        if user and check_password_hash(user['password'], password):
            session['user_id'] = str(user['_id'])
            session['user_name'] = user['name']
            return redirect(url_for('dashboard'))
        else:
            return render_template('login.html', error='Invalid email or password')

    return render_template('login.html')

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('home'))

@app.route('/api/verify', methods=['POST'])
def api_verify():
    """API endpoint for real-time verification"""
    try:
        data = request.get_json()
        if not data or 'keystrokes' not in data:
            return jsonify({
                'status': 'error',
                'message': 'No keystroke data provided'
            }), 400
        
        keystrokes = data['keystrokes']
        
        if len(keystrokes) < Config.MIN_KEYSTROKES:
            return jsonify({
                'status': 'error',
                'message': f'Insufficient keystrokes. Minimum required: {Config.MIN_KEYSTROKES}'
            }), 400
        
        # Save keystroke data to CSV file
        saved_filepath, sample_number = save_keystroke_data(keystrokes)
        
        if saved_filepath is None:
            logging.warning("Failed to save keystroke data, but continuing with verification")
        else:
            logging.info(f"Keystroke data saved as sample{sample_number}.csv")
        
        # Convert to DataFrame for processing
        df = pd.DataFrame(keystrokes)
        
        # Extract features
        features = feature_extractor.extract_comprehensive_features(df)
        
        # Get verifier and perform verification
        verifier_instance = get_verifier()
        if not verifier_instance:
            return jsonify({
                'status': 'error',
                'message': 'Verification system not available'
            }), 503
        
        # Create temporary result using features
        result = {
            'file_info': {
                'filename': f'sample{sample_number}.csv' if sample_number else 'realtime_verification',
                'saved_as': f'sample{sample_number}.csv' if sample_number else None,
                'keystroke_count': len(keystrokes),
                'verification_time': datetime.now().isoformat()
            },
            'authentication_result': {
                'is_authentic': True,  # This would come from actual model prediction
                'confidence': 85.0,    # Placeholder - real implementation would use model
                'ensemble_decision': 'Legitimate',
                'model_agreement': 95.0,
                'final_decision': 'AUTHENTICATED'
            },
            'typing_statistics': {
                'dwell_time_mean': features.get('dwell_mean', 0),
                'dwell_time_std': features.get('dwell_std', 0),
                'flight_time_mean': features.get('flight_mean', 0),
                'flight_time_std': features.get('flight_std', 0),
                'total_time': features.get('total_time', 0),
                'typing_speed': features.get('typing_speed', 0),
                'dwell_flight_ratio': features.get('dwell_flight_ratio', 0)
            }
        }
        
        return jsonify({
            'status': 'success',
            'result': result,
            'saved_file': f'sample{sample_number}.csv' if sample_number else None
        })
        
    except Exception as e:
        logging.error(f"Verification error: {e}")
        return jsonify({
            'status': 'error',
            'message': f'Verification failed: {str(e)}'
        }), 500

@app.route('/api/analyze', methods=['POST'])
def api_analyze():
    """API endpoint for file-based analysis"""
    try:
        if 'file' not in request.files:
            return jsonify({
                'status': 'error',
                'message': 'No file provided'
            }), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({
                'status': 'error', 
                'message': 'No file selected'
            }), 400
        
        if not file.filename.endswith('.csv'):
            return jsonify({
                'status': 'error',
                'message': 'Only CSV files are supported'
            }), 400
        
        # Save uploaded file temporarily
        temp_path = os.path.join('data', f'temp_upload_{datetime.now().timestamp()}.csv')
        file.save(temp_path)
        
        # Perform verification
        verifier_instance = get_verifier()
        if not verifier_instance:
            return jsonify({
                'status': 'error',
                'message': 'Verification system not available'
            }), 503
        
        result = verifier_instance.verify_keystroke_file(temp_path)
        
        # Clean up
        if os.path.exists(temp_path):
            os.remove(temp_path)
        
        if result:
            return jsonify({
                'status': 'success',
                'result': result
            })
        else:
            return jsonify({
                'status': 'error',
                'message': 'Verification failed'
            }), 500
            
    except Exception as e:
        logging.error(f"Analysis error: {e}")
        return jsonify({
            'status': 'error',
            'message': f'Analysis failed: {str(e)}'
        }), 500

@app.route('/api/status')
def api_status():
    """API endpoint for system status"""
    verifier_instance = get_verifier()
    
    return jsonify({
        'status': 'online',
        'model_loaded': verifier_instance is not None,
        'timestamp': datetime.now().isoformat(),
        'version': '2.0.0'
    })

@app.route('/api/statistics')
def api_statistics():
    """API endpoint for system statistics"""
    # Count data files
    data_files = [f for f in os.listdir(Config.DATA_DIR) 
                 if f.startswith('sample') and f.endswith('.csv')]
    
    return jsonify({
        'data_files_count': len(data_files),
        'model_status': 'loaded' if get_verifier() else 'not_loaded',
        'system_uptime': datetime.now().isoformat(),
        'next_sample_number': get_next_sample_number()
    })

@app.errorhandler(404)
def not_found(error):
    return jsonify({
        'status': 'error',
        'message': 'Endpoint not found'
    }), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({
        'status': 'error',
        'message': 'Internal server error'
    }), 500

def main():
    """Main application runner"""
    Config.setup_directories()
    
    # Ensure data directory exists
    os.makedirs(Config.DATA_DIR, exist_ok=True)
    
    print("🚀 Starting Optimized Keystroke Dynamics Authentication System...")
    print(f"📁 Data directory: {Config.DATA_DIR}")
    print(f"🤖 Model directory: {Config.MODEL_DIR}")
    print(f"🌐 Web interface: http://localhost:5000")
    print(f"💾 Next sample will be saved as: sample{get_next_sample_number()}.csv")
    
    app.run(
        host='0.0.0.0',
        port=5000,
        debug=True,
        threaded=True
    )

if __name__ == "__main__":
    main()