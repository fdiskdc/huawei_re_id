"""
Flask development server entry point.

For production, use wsgi.py with Gunicorn instead.
"""
from mrmodn_backend.app import create_app
from mrmodn_backend.core.config import config, get_logger

logger = get_logger('server')
app = create_app()

if __name__ == '__main__':
    logger.info(f"Starting Flask server on {config.FLASK_HOST}:{config.FLASK_PORT}")
    app.run(debug=config.FLASK_DEBUG, host=config.FLASK_HOST, port=config.FLASK_PORT)
