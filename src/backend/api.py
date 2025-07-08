from flask import Flask, request, jsonify, make_response
from flask_cors import CORS
import logging
from promptomatix.main import (
    process_input,
    save_feedback,
    feedback_store,
    optimize_with_feedback,
    optimization_sessions,
    optimize_with_synthetic_feedback
)
from promptomatix.utils.paths import SESSIONS_DIR
from flask_cors import cross_origin
import json
import os
from datetime import datetime
import traceback
from pathlib import Path

app = Flask(__name__)
CORS(app)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@app.route('/optimize', methods=['POST'])
def optimize_prompt_endpoint():
    try:
        data = request.json
        human_input = data.get('input')
        session_id = data.get('sessionId')
        config = data.get('config', {})
        
        logger.info(f"API received input for session {session_id}")
        logger.info(f"human input: {human_input}")
        
        try:
            result = process_input(raw_input=human_input, config=config)
            logger.info(f"Process input result: {result}")  # Add this line
        except Exception as e:
            logger.error(f"Error in process_input: {str(e)}")
            logger.error(f"Traceback: {traceback.format_exc()}")  # Add this line
            raise
        
        response = {
            'optimizedPrompt': result.get('result', ''),
            'sessionId': result.get('session_id', session_id),
            'metrics': result.get('metrics', None),
            'synthetic_data': result.get('synthetic_data', [])
        }
        
        # Save session data
        if response['sessionId']:
            save_session_data(response['sessionId'], {
                'input': human_input,
                'optimizedPrompt': response['optimizedPrompt'],
                'metrics': response['metrics'],
                'config': config,
                'timestamp': datetime.now().isoformat()
            })
        
        return jsonify(response)
    except Exception as e:
        logger.error(f"API error: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/optimize-with-feedback', methods=['POST'])
@cross_origin()
def optimize_with_feedback_endpoint():
    try:
        data = request.json
        session_id = data.get('session_id')
        print(f"Received optimize-with-feedback request for session: {session_id}")  # Debug log
        
        if not session_id:
            return jsonify({
                'error': 'No session_id provided',
                'result': None,
                'metrics': None
            }), 400
            
        # Get the session
        session = optimization_sessions.get(session_id)
        if not session:
            print(f"Session not found: {session_id}")  # Debug log
            return jsonify({
                'error': f'Session {session_id} not found',
                'result': None,
                'metrics': None
            }), 404
            
        # Get the latest feedback for this session using the proper method
        session_feedbacks = feedback_store.get_feedback_for_prompt(session_id)
        if not session_feedbacks:
            print(f"No feedback found for session: {session_id}")  # Debug log
            return jsonify({
                'error': 'No feedback found for this session',
                'result': None,
                'metrics': None
            }), 400
            
        # Call optimize_with_feedback with the session_id
        result = optimize_with_feedback(session_id)
        print(f"Optimization result: {result}")  # Debug log
        
        return jsonify(result)
        
    except Exception as e:
        print(f"Error in optimize_with_feedback_endpoint: {str(e)}")  # Debug log
        return jsonify({
            'error': str(e),
            'session_id': session_id if 'session_id' in locals() else None,
            'result': None,
            'metrics': None
        }), 500

@app.route('/comments', methods=['POST'])
@cross_origin()
def add_comment():
    try:
        data = request.json
        print(f"Received data in /comments endpoint: {data}")  # Debug log
        
        # Validate required fields
        required_fields = ['text', 'startOffset', 'endOffset', 'feedback', 'promptId']
        missing_fields = [field for field in required_fields if field not in data]
        
        if missing_fields:
            print(f"Missing required fields: {missing_fields}")  # Debug log
            return jsonify({
                "success": False, 
                "error": f"Missing required fields: {', '.join(missing_fields)}"
            }), 400
            
        result = save_feedback(
            text=data['text'],
            start_offset=data['startOffset'],
            end_offset=data['endOffset'],
            feedback=data['feedback'],  # Changed from comment to feedback
            prompt_id=data['promptId']
        )
        return jsonify({"success": True, "comment": result})
    except Exception as e:
        print(f"Error in add_comment endpoint: {str(e)}")  # Debug log
        return jsonify({"success": False, "error": str(e)}), 400

@app.route('/comments', methods=['GET'])
@cross_origin()
def get_comments():
    try:
        # Get all feedback using the proper method
        comments_json = feedback_store.get_all_feedback()
        return jsonify({"success": True, "comments": comments_json})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 400

@app.route('/session/<session_id>', methods=['GET'])
@cross_origin()
def get_session(session_id):
    try:
        session_data = load_session_data(session_id)
        if session_data:
            return jsonify(session_data)
        return jsonify({'error': 'Session not found'}), 404
    except Exception as e:
        logger.error(f"Error getting session: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/session/<session_id>/download', methods=['GET'])
@cross_origin()
def download_session(session_id):
    try:
        session_data = load_session_data(session_id)
        if session_data:
            response = make_response(json.dumps(session_data, indent=2))
            response.headers['Content-Type'] = 'application/json'
            response.headers['Content-Disposition'] = f'attachment; filename=session_{session_id}.json'
            return response
        return jsonify({'error': 'Session not found'}), 404
    except Exception as e:
        logger.error(f"Error downloading session: {str(e)}")
        return jsonify({'error': str(e)}), 500

def save_session_data(session_id, data):
    """Save session data to a file"""
    # Ensure the sessions directory exists
    SESSIONS_DIR.mkdir(parents=True, exist_ok=True)
    
    file_path = SESSIONS_DIR / f'session_{session_id}.json'
    try:
        with open(file_path, 'w') as f:
            json.dump(data, f, indent=2)
    except Exception as e:
        logger.error(f"Error saving session data: {str(e)}")
        raise

def load_session_data(session_id):
    """Load session data from a file"""
    file_path = SESSIONS_DIR / f'session_{session_id}.json'
    try:
        if os.path.exists(file_path):
            with open(file_path, 'r') as f:
                return json.load(f)
        return None
    except Exception as e:
        logger.error(f"Error loading session data: {str(e)}")
        raise

@app.route('/session/<session_id>/log', methods=['GET'])
@cross_origin()
def get_session_log(session_id):
    try:
        logger.info(f"Attempting to get log for session: {session_id}")
        
        if session_id not in optimization_sessions:
            logger.error(f"Session not found: {session_id}")
            return jsonify({
                'error': 'Session not found'
            }), 404
            
        session = optimization_sessions[session_id]
        logger.info("Found session, formatting log...")
        
        try:
            log_content = session.logger.format_log()
            logger.info("Log formatted successfully")
            
            # Even if there was an error in the optimization process,
            # we still want to return the log
            response = make_response(log_content)
            response.headers['Content-Type'] = 'text/plain'
            response.headers['Content-Disposition'] = f'attachment; filename=session_{session_id}_log.txt'
            logger.info("Response prepared successfully")
            return response
            
        except Exception as format_error:
            logger.error(f"Error formatting log: {str(format_error)}")
            return jsonify({
                'error': f'Error formatting log: {str(format_error)}'
            }), 500
        
    except Exception as e:
        logger.error(f"Unexpected error in get_session_log: {str(e)}")
        return jsonify({
            'error': str(e)
        }), 500

@app.route('/synthetic-data-feedback', methods=['POST'])
def handle_synthetic_data_feedback():
    """
    Handle feedback for synthetic dataset.
    
    Returns:
        Dict: Response with success/error status
    """
    try:
        data = request.get_json()
        session_id = data.get('session_id')
        feedback = data.get('feedback')
        
        if not session_id or not feedback:
            return jsonify({
                'error': 'Missing required fields: session_id and feedback'
            }), 400
        
        # Call optimize_with_synthetic_feedback with the feedback
        result = optimize_with_synthetic_feedback(session_id, feedback)
        
        if result.get('error'):
            return jsonify({
                'error': result['error'],
                'traceback': result.get('traceback')
            }), 500
        
        return jsonify({
            'success': True,
            'message': 'Feedback processed successfully',
            'result': result
        })
        
    except Exception as e:
        error_msg = str(e)
        trace = traceback.format_exc()
        
        return jsonify({
            'error': error_msg,
            'traceback': trace
        }), 500

@app.route('/load-session', methods=['POST'])
@cross_origin()
def load_session_endpoint():
    try:
        data = request.json
        session_file_path = data.get('session_file_path')
        
        if not session_file_path:
            return jsonify({
                'error': 'No session file path provided',
                'result': None
            }), 400
            
        # Load the session using the session manager
        session = optimization_sessions.session_manager.load_session_from_file(session_file_path)
        
        if not session:
            return jsonify({
                'error': f'Failed to load session from {session_file_path}',
                'result': None
            }), 404
            
        # Return the session data
        return jsonify({
            'session_id': session.session_id,
            'input': session.initial_human_input,
            'optimizedPrompt': session.latest_optimized_prompt,
            'config': session.config.__dict__,
            'result': 'Session loaded successfully'
        })
        
    except Exception as e:
        logger.error(f"Error loading session: {str(e)}")
        return jsonify({
            'error': str(e),
            'result': None
        }), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)