import time
import logging
from functools import wraps

logger = logging.getLogger(__name__)

def retry_on_network_error(max_retries=3, delay=5):
    # Retry database operations on network errors with exponential backoff
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    error_msg = str(e).lower()
                    
                    # List of keywords that indicate network errors
                    network_keywords = [
                        'network', 'connection', 'timeout', 'broken pipe', 
                        'reset by peer', 'timed out', '10060', 'winerror 10060',
                        'connection attempt failed', 'host has failed to respond'
                    ]
                    
                    # Check if error message contains network error keywords
                    is_network_error = any(keyword in error_msg for keyword in network_keywords)
                    
                    if is_network_error and attempt < max_retries - 1:
                        # Calculate wait time with exponential backoff
                        wait_time = delay * (2 ** attempt)
                        logger.warning(
                            f"Network error in {func.__name__}: {e}. "
                            f"Retrying in {wait_time}s (attempt {attempt + 1}/{max_retries})..."
                        )
                        time.sleep(wait_time)
                        continue
                    else:
                        # Log detailed error information
                        logger.error(f"Error in {func.__name__}: {type(e).__name__}: {e}")
                        raise
            
            return None
        return wrapper
    return decorator


def verify_connection(cur):
    # Test if database connection is still alive
    try:
        cur.execute("SELECT 1")
        return True
    except Exception as e:
        logger.warning(f"Connection test failed: {e}")
        return False