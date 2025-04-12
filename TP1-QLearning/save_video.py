import cv2
import os

from dotenv import load_dotenv

def create_video_from_images(image_list, output_path, frame_duration=0.5):
    if not image_list:
        raise ValueError("The image list is empty.")
    
    # Read the first image to get dimensions
    first_image = cv2.imread(image_list[0])
    if first_image is None:
        raise ValueError(f"Cannot read the image: {image_list[0]}")
    
    height, width, layers = first_image.shape
    fps = int(1 / frame_duration)
    
    # Define the codec and create VideoWriter object
    
    fourcc = cv2.VideoWriter_fourcc(*'avc1') # Codec for .mp4 files
    video = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    for image_path in image_list:
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Cannot read the image: {image_path}")
        video.write(img)
    
    video.release()
    print(f"Video saved to {output_path}")

# Example usage
if __name__ == "__main__":
    


    load_dotenv()
    base_path = "/Users/fabricio.denardi/Documents/CEIA/AR1/repos/MIA_01c_AR1/TP1-QLearning/taxi_env_qlearning/"
    results_path = os.path.join(base_path,"results")
    best_result_path = os.path.join(results_path,"best_result")
    img_results = os.path.join(best_result_path,"img")

    images = sorted(
        [os.path.join(img_results, f) for f in os.listdir(img_results) if f.endswith('.png')]
    )

    output_video_path = os.path.join(best_result_path, "step_by_step.mov")

    create_video_from_images(images, output_video_path)