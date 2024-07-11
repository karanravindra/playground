import os
import json
import torch
import torch.nn.functional as F
import torchvision


# Function to load metadata
def load_metadata(metadata_path):
    with open(metadata_path, "r") as f:
        metadata = json.load(f)
    return metadata


# Load video files
video_dir = "blog/12-dotcloud/vids"
metadata_dir = "blog/12-dotcloud/metadata"
video_files = [f for f in os.listdir(video_dir) if f.endswith(".mp4")]
metadata_files = [f.replace(".mp4", ".json") for f in video_files]

# Load videos and metadata
videos = []
metadata_list = []
for video_file, metadata_file in zip(video_files, metadata_files):
    video_path = os.path.join(video_dir, video_file)
    metadata_path = os.path.join(metadata_dir, metadata_file)

    # Read video
    video_frames, _, _ = torchvision.io.read_video(video_path, pts_unit="sec")
    video_frames = video_frames.permute(0, 3, 1, 2)
    videos.append(video_frames)

    # Read metadata
    metadata = load_metadata(metadata_path)
    metadata_list.append(metadata)

# Concatenate all videos
videos_tensor = torch.cat(videos)
videos_tensor = F.interpolate(videos_tensor, size=(64, 64))
print("Videos tensor shape:", videos_tensor.shape)


# Function to process metadata into a tensor
def process_metadata(metadata_list, num_frames):
    class_labels = []
    for metadata in metadata_list:
        for frame_data in metadata:
            ball_data = []
            for ball in frame_data["balls"]:
                ball_data.append(
                    [
                        ball["center"][0],
                        ball["center"][1],  # x, y center
                        ball["radius"],  # radius
                        ball["color"][0],
                        ball["color"][1],
                        ball["color"][2],  # color
                    ]
                )
            while len(ball_data) < num_frames:
                ball_data.append([0, 0, 0, 0, 0, 0])  # Pad with zeros if fewer balls
            class_labels.append(ball_data)
    return torch.tensor(class_labels)


# Assume a maximum of 5 balls per frame for simplicity
num_balls = 1

# Process metadata
metadata_tensor = process_metadata(metadata_list, num_balls)
metadata_tensor = metadata_tensor.view(-1, num_balls, 6)  # Reshape to (N, num_balls, 6)
print("Metadata tensor shape:", metadata_tensor.shape)

# Save tensors
torch.save(videos_tensor, "blog/12-dotcloud/videos.pt")
torch.save(metadata_tensor, "blog/12-dotcloud/metadata.pt")
