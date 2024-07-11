from PIL import Image, ImageDraw
import numpy as np
from moviepy.editor import ImageSequenceClip
import json


class Ball:
    def __init__(
        self,
        x,
        y,
        vx,
        vy,
        radius,
        color,
        mass,
        gravity=0.3,
        bounce_factor=0.8,
        friction=0.9999,
    ):
        self.x = x
        self.y = y
        self.vx = vx
        self.vy = vy
        self.radius = radius
        self.color = color
        self.mass = mass
        self.gravity = gravity
        self.bounce_factor = bounce_factor
        self.friction = friction

    def update(self, screen_width, screen_height):
        self.vy += self.gravity
        self.x += self.vx
        self.y += self.vy

        # Apply friction
        self.vx *= self.friction
        self.vy *= self.friction

        # Bounce off the walls
        if self.x - self.radius < 0:
            self.x = self.radius
            self.vx = -self.vx * self.bounce_factor
        elif self.x + self.radius > screen_width:
            self.x = screen_width - self.radius
            self.vx = -self.vx * self.bounce_factor
        if self.y - self.radius < 0:
            self.y = self.radius
            self.vy = -self.vy * self.bounce_factor
        elif self.y + self.radius > screen_height:
            self.y = screen_height - self.radius
            self.vy = -self.vy * self.bounce_factor

    def to_dict(self):
        return {"center": [self.x, self.y], "radius": self.radius, "color": self.color}


class BouncingBallsAnimation:
    def __init__(self, screen_width, screen_height, balls_per_color, capture=False):
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.balls = self.create_balls(balls_per_color)
        self.frames = []
        self.metadata = []
        self.capture = capture

    def create_balls(self, balls_per_color):
        balls = []
        colors = list(balls_per_color.keys())
        for color in colors:
            count = balls_per_color[color]
            for _ in range(count):
                x = np.random.randint(16, self.screen_width - 16)
                y = np.random.randint(16, self.screen_height - 16)
                vx = np.random.uniform(-4, 4)
                vy = np.random.uniform(-4, 4)
                radius = np.random.randint(10, 20)
                mass = (4 / 3) * np.pi * radius**3
                balls.append(Ball(x, y, vx, vy, radius, color, mass))
        return balls

    def run(self, num_frames=300):
        for frame_count in range(num_frames):
            frame_metadata = {
                "frame": frame_count,
                "ball_count": len(self.balls),
                "balls": [],
            }

            # Update all balls
            for ball in self.balls:
                ball.update(self.screen_width, self.screen_height)
                frame_metadata["balls"].append(ball.to_dict())

            # Check for collisions
            for i in range(len(self.balls)):
                for j in range(i + 1, len(self.balls)):
                    if self.balls[i].check_collision(self.balls[j]):
                        self.balls[i].resolve_collision(self.balls[j])

            # Capture the frame
            if self.capture:
                self.capture_frame(frame_metadata)

            self.metadata.append(frame_metadata)

    def capture_frame(self, frame_metadata):
        image = Image.new(
            "RGB", (self.screen_width, self.screen_height), (255, 255, 255)
        )
        draw = ImageDraw.Draw(image)
        for ball_data in frame_metadata["balls"]:
            x, y = ball_data["center"]
            radius = ball_data["radius"]
            color = tuple(ball_data["color"])
            draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=color)
        self.frames.append(np.array(image))

    def save_video(self, filename, metadata_filename, fps=60):
        if self.capture:
            print("Saving video...")
            clip = ImageSequenceClip(self.frames, fps=fps)
            clip.write_videofile(filename, codec="libx264")

            with open(metadata_filename, "w") as f:
                json.dump(self.metadata, f, indent=4)


if __name__ == "__main__":
    for i in range(20):
        # Define the number of balls per color
        balls_per_color = {
            (255, 0, 0): 1,  # Red balls
            (0, 255, 0): 0,  # Green balls
            (0, 0, 255): 0,  # Blue balls
            (255, 255, 0): 0,  # Yellow balls
        }

        animation = BouncingBallsAnimation(
            screen_width=256,
            screen_height=256,
            balls_per_color=balls_per_color,
            capture=True,
        )
        animation.run(num_frames=300)
        animation.save_video(
            f"blog/12-dotcloud/vids/bouncing_balls{i}.mp4",
            f"blog/12-dotcloud/metadata/bouncing_balls{i}.json",
            fps=60,
        )
