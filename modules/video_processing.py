import os, cv2

class Video:

    def __init__(self, codec:str='mp4v', fps:int=3, shape:tuple=(854, 480), overwrite=False):
        self.codec = codec; self.fps = fps; self.shape = shape
        self.overwrite = overwrite

    def writer(self, path):
        if not self.overwrite and os.path.exists(path):
            print(f'ANNOTATE VIDEO TIMESTAMP FAILED. FILE ALREADY EXISTS · FILE-PATH: {path}')
            logging.error(f'ANNOTATE VIDEO TIMESTAMP FAILED. FILE ALREADY EXISTS · FILE-PATH: {path}')
            return False
        return cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*self.codec), self.fps, self.shape)

def get_video_metadata(video_path=None, cap=None):
    if cap is None and video_path is not None:
        cap = cv2.VideoCapture(video_path)
    # Get the frames per second (fps)
    fps = cap.get(cv2.CAP_PROP_FPS)
    # Get the frame dimensions (shape)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    shape = (w, h) # get the shape
    if cap is None and video_path is not None:
        cap.release(); cv2.destroyAllWindows()
    return fps, shape