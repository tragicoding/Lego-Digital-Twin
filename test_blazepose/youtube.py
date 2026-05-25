from pytubefix import YouTube
from moviepy.editor import VideoFileClip
import os

url = "https://youtu.be/XxsnzyvzLBc?si=q-XCnLV8XJajsv-4"

yt = YouTube(url)
stream = yt.streams.filter(progressive=True, file_extension='mp4').first()

out_dir = "videos"
os.makedirs(out_dir, exist_ok=True)
video_path = stream.download(output_path=out_dir)

# # 트리밍 (0~2초)
# clip = VideoFileClip(video_path).subclip('01:10', '01:30')
# trimmed_path = os.path.join(out_dir, f"{yt.title}_trimmed.mp4")
# clip.write_videofile(trimmed_path, codec="libx264")
# print("✅ Trimmed video saved at:", trimmed_path)
