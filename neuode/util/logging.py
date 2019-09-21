"""
Logging and Formats
"""

import logging
logging.basicConfig(format='%(levelname)s [%(asctime)s]: %(message)s',
					level=logging.INFO)
logger = logging.getLogger(__name__)