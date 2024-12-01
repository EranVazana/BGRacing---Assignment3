from ultralytics import YOLO
import torch

def main():
   # Dataset used: https://universe.roboflow.com/yuval-k/cones-detection-k0i6h/dataset/2

   # Ensures safe execution on Windows
   torch.multiprocessing.freeze_support()

   # Check if CUDA installation exists:
   if (torch.cuda.is_available()):
      # Setting the trainer to run on default CUDA device (Currently on: NVIDIA RTX 4060TI)
      torch.cuda.set_device(0)
   else:
       print('WARNING: Cuda is not installed on this system.\nThe model trainer will run significantly slower.')

   # Train the cones model.
   YOLO('model_trainer\\yolov8n.pt').train(  
      data='model_trainer\\data.yaml',
      imgsz=640,
      epochs=300, 
      batch=8,
      name='cones'
   )
    
if __name__ == '__main__':
    main()