from ultralytics import YOLO

# load the model
model = YOLO("yolov8n.yaml") # build a new model from scratch
#model = YOLO("yolov8n.pt") # load pretrained model

# use the model
results = model.train(data="config.yaml", epochs=1)
#results = model.val() # evaluate the model
#results = model("") # predict an images
#success = model.export(format="onnx") # export the model to ONNX format