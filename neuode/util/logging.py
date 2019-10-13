"""
Logging and Formats
"""

import numpy as np
import cv2
import logging

import neuode.util.util as util

# logs
logging.basicConfig(format='%(levelname)s [%(asctime)s]: %(message)s',
					level=logging.INFO)
logger = logging.getLogger(__name__)


# render np array [frame, channel, height, width] to video
def render_video(frames, fps=24.0, path='dump/dummy.mp4'):
	# initialize video writer
	writer = cv2.VideoWriter(
		path,
		cv2.VideoWriter_fourcc(*'MP4V'),
		fps,
		(frames.shape[-1], frames.shape[-2]))

	# write frames to video
	frames = (util.normalize_range(frames) * 255.0).astype(np.uint8)
	for frame in frames:
		frame = np.transpose(frame, [1, 2, 0])
		writer.write(frame)
	writer.release()